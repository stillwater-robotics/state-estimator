#include "state_estimator.h"
#include <cmath>

using namespace Eigen;

StateEstimator::StateEstimator() {
    x.setZero();
    P.setZero();
    P.diagonal().setConstant(0.1f);
}

void StateEstimator::Predict(float aL, float aR, float aZ, float dt) {
    Matrix<float, 7, 7> F = Matrix<float, 7, 7>::Identity();
    
    float vL = x(IDX_VL);
    float vR = x(IDX_VR);
    float th = x(IDX_THETA);
    
    float cos_th = cosf(th);
    float sin_th = sinf(th);

    // Linearized state transition
    F(IDX_X, IDX_VL) = 0.5f * cos_th * dt;
    F(IDX_X, IDX_VR) = 0.5f * cos_th * dt;
    F(IDX_Y, IDX_VL) = 0.5f * sin_th * dt;
    F(IDX_Y, IDX_VR) = 0.5f * sin_th * dt;
    F(IDX_Z, IDX_VZ) = dt;
    
    F(IDX_X, IDX_THETA) = -0.5f * (vL + vR) * sin_th * dt;
    F(IDX_Y, IDX_THETA) =  0.5f * (vL + vR) * cos_th * dt;

    F(IDX_THETA, IDX_VL) = (-1.0f / L_BASE) * dt;
    F(IDX_THETA, IDX_VR) = (1.0f / L_BASE) * dt;
    
    // State prediciton
    x(IDX_X)     += (0.5f * cos_th * (vL + vR)) * dt;
    x(IDX_Y)     += (0.5f * sin_th * (vL + vR)) * dt;
    x(IDX_Z)     += x(IDX_VZ) * dt;
    x(IDX_THETA) += (1.0f / L_BASE * (-vL + vR)) * dt;
    
    x(IDX_VL) += aL * dt;
    x(IDX_VR) += aR * dt;
    x(IDX_VZ) += aZ * dt;

    // Covariance prediciton
    Matrix<float, 7, 7> Q = Matrix<float, 7, 7>::Identity(); 
    Q *= (std_accel * dt * std_accel * dt); 
    P = F * P * F.transpose() + Q;

    saveToHistory(F);
}

void StateEstimator::UpdatePressure(float z_meas) {
    Matrix<float, 1, 7> H = Matrix<float, 1, 7>::Zero();
    H(0, IDX_Z) = 1.0f;
    
    float R = std_press * std_press;
    float z = z_meas;
    float y = z - (H * x)(0); 
    
    // Kalman Gain calculation
    // S = H*P*H^T + R (Result is 1x1)
    float S = (H * P * H.transpose())(0, 0) + R;
    Matrix<float, 7, 1> K = P * H.transpose() * (1.0f / S);

    x += K * y;
    Matrix<float, 7, 7> I = Matrix<float, 7, 7>::Identity();
    P = (I - K * H) * P;
}

void StateEstimator::UpdateGPS(float gps_x, float gps_y) {
    Matrix<float, 2, 7> H = Matrix<float, 2, 7>::Zero();
    H(0, IDX_X) = 1.0f; 
    H(1, IDX_Y) = 1.0f;

    Vector2f z(gps_x, gps_y);
    Vector2f y = z - (H * x); 

    // Calculate convergence metric to decide if we should throw away outliers
    float current_uncertainty = P(IDX_X, IDX_X) + P(IDX_Y, IDX_Y);
    float convergence_threshold = 0.5f; 

    if (current_uncertainty >= convergence_threshold) {
        float gate_radius = 2.0f; // Throw out gps measurements that deviate from current estimate by this much
        float error_mag = y.norm();
        
        if (error_mag > gate_radius) {
            // Pull the estimate toward GPS
            y = y * (gate_radius / error_mag);
        }
    }

    // Kalman gain
    Matrix<float, 2, 2> R = Matrix<float, 2, 2>::Identity() * (std_gps * std_gps);
    Matrix2f S = H * P * H.transpose() + R;
    Matrix<float, 7, 2> K = P * H.transpose() * S.ldlt().solve(Matrix2f::Identity());

    x += K * y;
    Matrix<float, 7, 7> I = Matrix<float, 7, 7>::Identity();
    P = (I - K * H) * P;
    
    // Forced symmetry for stability
    P = (P + P.transpose()) * 0.5f;
}

void StateEstimator::RunRTSSmoother() {
    int current = (history_idx - 1 + HISTORY_SIZE) % HISTORY_SIZE;

    for(int i = 0; i < HISTORY_SIZE - 1; i++) {
        int next = current;
        int prev = (current - 1 + HISTORY_SIZE) % HISTORY_SIZE;

        StateSnapshot& s_next = history[next];
        StateSnapshot& s_prev = history[prev];

        // Formal RTS Gain: C = P_k * F^T * (P_pred_k+1)^-1
        Matrix<float, 7, 7> P_pred = s_prev.F * s_prev.P * s_prev.F.transpose(); 
        // Small epsilon on diagonal improves inversion stability
        P_pred.diagonal().array() += 1e-6f; 
        
        Matrix<float, 7, 7> C = s_prev.P * s_prev.F.transpose() * P_pred.inverse();
        
        // Smoothing state: x_k|N = x_k|k + C * (x_k+1|N - x_k+1|k)
        s_prev.x = s_prev.x + C * (s_next.x - (s_prev.F * s_prev.x));
        
        current = prev;
    }
}

void StateEstimator::saveToHistory(const Matrix<float, 7, 7>& F_k) {
    history[history_idx].x = x;
    history[history_idx].P = P;
    history[history_idx].F = F_k;
    history_idx = (history_idx + 1) % HISTORY_SIZE;
    if (history_idx == 0) buffer_full = true;
}