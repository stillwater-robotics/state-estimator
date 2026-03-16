#ifdef ARDUINO
#include <ArduinoEigenDense.h>
#else 
#include <Eigen/Dense>
#endif

#ifndef STATE_ESTIMATOR_H
#define STATE_ESTIMATOR_H

const float L_BASE = 0.5f; 
const int HISTORY_SIZE = 50; 

enum StateIdx {
    IDX_X = 0,
    IDX_Y = 1,
    IDX_Z = 2,
    IDX_THETA = 3,
    IDX_VL = 4,
    IDX_VR = 5,
    IDX_VZ = 6,
    STATE_DIM = 7
};

struct StateSnapshot { // This holds info for the RTS smoother
    Eigen::Matrix<float, STATE_DIM, 1> x; // State
    Eigen::Matrix<float, STATE_DIM, STATE_DIM> P; // Covariance
    Eigen::Matrix<float, STATE_DIM, STATE_DIM> F; // Jacobian
    bool hasGPS;
};

class StateEstimator {
public:
    Eigen::Matrix<float, STATE_DIM, 1> x; 
    Eigen::Matrix<float, STATE_DIM, STATE_DIM> P;

    // Sensor noise
    float std_gps = 1.5f;
    float std_press = 0.2f;
    float std_accel = 0.2f;

    // History buffer for RTS Smoother
    StateSnapshot history[HISTORY_SIZE];
    int history_idx = 0;
    bool buffer_full = false;

public:
    StateEstimator();

    void Predict(float aL, float aR, float aZ, float dt);
    
    void UpdatePressure(float z_meas);
    void UpdateGPS(float gps_x, float gps_y);

    void RunRTSSmoother();

private:
    void saveToHistory(const Eigen::Matrix<float, STATE_DIM, STATE_DIM>& F_k);

    // Ensure 16-byte alignment for Eigen fixed-size types if using on certain architectures
    // (idk it breaks without this)
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

#endif