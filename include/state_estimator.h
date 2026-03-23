#ifdef ARDUINO
#include <ArduinoEigen.h>
#else
#include <Eigen/Dense>
#endif

#ifdef ARDUINO
#include "../../controller/include/controller.h"
#else
#include "controller.h"
#endif

#ifndef STATE_ESTIMATOR_HEADER
#define STATE_ESTIMATOR_HEADER

#include <vector>
#include <cmath>


// RTS smoother history storage
struct HistoryNode {
    Eigen::VectorXd x_pred;
    Eigen::MatrixXd P_pred;
    Eigen::VectorXd x_upd;
    Eigen::MatrixXd P_upd;
    Eigen::MatrixXd F_matrix;
};

class StateEstimator {
private:
    State state;
    Eigen::MatrixXd P;
    Eigen::MatrixXd Q;
    
    // Sensor Covariances
    Eigen::MatrixXd R_gps;
    Eigen::MatrixXd R_imu;
    Eigen::MatrixXd R_pressure;
    
    float dt;
    
    std::vector<HistoryNode> history;
    Eigen::VectorXd prev_updated_state_vec; // Needed for IMU velocity differential

public:
    StateEstimator(State initial_state = State(), float timestep = 0.1) 
        : state(initial_state), dt(timestep) {
        
        P = Eigen::MatrixXd::Identity(8, 8) * 0.1;
        Q = Eigen::MatrixXd::Identity(8, 8) * 0.01;
        
        // 1.5 m accuracy GPS
        R_gps = Eigen::MatrixXd::Identity(2, 2) * 3.5; 
        
        // 0.5 m/s^2 accuracy IMU
        R_imu = Eigen::MatrixXd::Identity(3, 3) * 0.25; 
        
        // 0.1 m accuracy Pressure Sensor
        R_pressure = Eigen::MatrixXd::Identity(1, 1) * 0.01; 

        prev_updated_state_vec = stateToVec(state);
    }

    /**
     * @brief Prediction Step
     */
    void Predict(Input input) {
        prev_updated_state_vec = stateToVec(state);

        Eigen::MatrixXd F_matrix = calculateJacobian(state, input);
        state = dynamics(state, input, dt);
        P = F_matrix * P * F_matrix.transpose() + Q;

        // RTS Smoother storage
        HistoryNode node;
        node.x_pred = stateToVec(state);
        node.P_pred = P;
        node.F_matrix = F_matrix;
        node.x_upd = node.x_pred;
        node.P_upd = node.P_pred;
        history.push_back(node);
    }

    /**
     * @brief GPS Update: X and Y. Triggers RTS Smoother if surfaced.
     */
    void UpdateGPS(float gps_x, float gps_y) {
        if (state.pose.z > 0.5) return; 

        Eigen::MatrixXd H = Eigen::MatrixXd::Zero(2, 8);
        H(0, 0) = 1.0; 
        H(1, 1) = 1.0;

        Eigen::VectorXd z_obs(2);
        z_obs << gps_x, gps_y;

        Eigen::VectorXd z_pred(2);
        z_pred << state.pose.x, state.pose.y;

        applyUpdate(z_obs, z_pred, H, R_gps);

        // Surface GPS lock! Runs the RTS smoother to correct underwater drift
        if (history.size() > 1) {
            runRTSSmoother();
        }
    }

    /**
     * @brief IMU Update: XYZ Acceleration
     * a = (v_k - v_k-1) / dt
     */
    void UpdateIMU(float ax, float ay, float az) {
        Eigen::MatrixXd H = Eigen::MatrixXd::Zero(3, 8);
        H(0, 4) = 1.0 / dt; // d(ax)/d(vx)
        H(1, 5) = 1.0 / dt; // d(ay)/d(vy)
        H(2, 6) = 1.0 / dt; // d(az)/d(vz)

        Eigen::VectorXd z_obs(3);
        z_obs << ax, ay, az;

        Eigen::VectorXd current_vec = stateToVec(state);
        Eigen::VectorXd z_pred(3);
        z_pred(0) = (current_vec(4) - prev_updated_state_vec(4)) / dt;
        z_pred(1) = (current_vec(5) - prev_updated_state_vec(5)) / dt;
        z_pred(2) = (current_vec(6) - prev_updated_state_vec(6)) / dt;

        applyUpdate(z_obs, z_pred, H, R_imu);
    }

    /**
     * @brief Pressure Sensor Update: Z depth
     */
    void UpdatePressure(float depth_z) {
        Eigen::MatrixXd H = Eigen::MatrixXd::Zero(1, 8);
        H(0, 2) = 1.0;

        Eigen::VectorXd z_obs(1);
        z_obs << depth_z;

        Eigen::VectorXd z_pred(1);
        z_pred << state.pose.z;

        applyUpdate(z_obs, z_pred, H, R_pressure);
    }

    State GetState() const { 
        return state; 
    }
    
    // Get RTS smoother corrected history
    std::vector<Eigen::VectorXd> GetSmoothedTrajectory() const {
        std::vector<Eigen::VectorXd> traj;
        for(const auto& node : history) traj.push_back(node.x_upd);
        return traj;
    }

private:

    /**
     * @brief Generalized Kalman Update to avoid code repetition
     */
    void applyUpdate(const Eigen::VectorXd& z_obs, const Eigen::VectorXd& z_pred, 
                      const Eigen::MatrixXd& H, const Eigen::MatrixXd& R) {
        
        Eigen::VectorXd y = z_obs - z_pred;
        
        Eigen::MatrixXd S = H * P * H.transpose() + R;
        Eigen::MatrixXd K = P * H.transpose() * S.inverse();

        Eigen::VectorXd x_vec = stateToVec(state);
        x_vec = x_vec + K * y;
        
        Eigen::MatrixXd I = Eigen::MatrixXd::Identity(8, 8);
        P = (I - K * H) * P;

        state = vecToState(x_vec);

        if (!history.empty()) {
            history.back().x_upd = x_vec;
            history.back().P_upd = P;
        }
    }

    /**
     * @brief Rauch-Tung-Striebel (RTS) Backward Pass
     */
    void runRTSSmoother() {
        int N = history.size();
        
        for (int k = N - 2; k >= 0; --k) {
            Eigen::MatrixXd P_pred_next = history[k + 1].P_pred;
            Eigen::MatrixXd F_next = history[k + 1].F_matrix;

            Eigen::MatrixXd C = history[k].P_upd * F_next.transpose() * P_pred_next.inverse();

            // Smooth state
            history[k].x_upd = history[k].x_upd + C * (history[k + 1].x_upd - history[k + 1].x_pred);

            // Smooth covariance
            history[k].P_upd = history[k].P_upd + C * (history[k + 1].P_upd - P_pred_next) * C.transpose();
        }

        history.clear(); 
    }

    Eigen::MatrixXd calculateJacobian(State s, Input u) {
        Eigen::MatrixXd F_matrix = Eigen::MatrixXd::Identity(8, 8);
        F_matrix(0, 4) = dt; F_matrix(1, 5) = dt; F_matrix(2, 6) = dt; F_matrix(3, 7) = dt; 

        float speed = sqrt(pow(s.velocity.x, 2) + pow(s.velocity.y, 2));
        F_matrix(0, 3) = -speed * sin(s.pose.theta) * dt;
        F_matrix(1, 3) =  speed * cos(s.pose.theta) * dt;
        return F_matrix;
    }

    Eigen::VectorXd stateToVec(const State& s) const {
        Eigen::VectorXd vec(8);
        vec << s.pose.x, s.pose.y, s.pose.z, s.pose.theta,
               s.velocity.x, s.velocity.y, s.velocity.z, s.velocity.theta;
        return vec;
    }

    State vecToState(const Eigen::VectorXd& vec) const {
        State s;
        s.pose.x = vec(0); s.pose.y = vec(1); s.pose.z = vec(2); s.pose.theta = vec(3);
        s.velocity.x = vec(4); s.velocity.y = vec(5); s.velocity.z = vec(6); s.velocity.theta = vec(7);
        return s;
    }
};

#endif