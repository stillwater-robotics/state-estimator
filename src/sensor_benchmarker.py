import json
import itertools
import numpy as np

# ======================================
# UUV SENSOR SUITE EVALUATION SCRIPT
# ======================================

def load_sensor_specs(json_path):
    with open(json_path, "r") as f:
        return json.load(f)

def imu_noise_from_spec(noise_density_ug, rate_hz):
    """
    Convert IMU noise density (µg/√Hz) to per-sample std (m/s²)
    """
    noise_density_mps2 = noise_density_ug * 1e-6 * 9.80665  # µg → m/s²
    sigma_a = noise_density_mps2 * np.sqrt(rate_hz)
    return sigma_a

def simulate_uuv(sensor_suite, T=200, dt=0.1, surface_interval=50):
    """
    Simple EKF-like propagation with intermittent GPS updates.
    State: [x, y, z, theta, z_dot]
    """
    imu_noise = sensor_suite['imu_noise']
    gps_noise = sensor_suite['gps_noise']
    press_noise = sensor_suite['press_noise']
    imu_rate = sensor_suite['imu_rate']

    # Initialize covariance
    P = np.eye(5) * 0.01

    # Process noise (IMU acceleration uncertainty into z_dot)
    q_zdot = imu_noise ** 2
    Q = np.diag([0.01, 0.01, 0.01, 0.001, q_zdot])

    # Measurement matrices
    H_gps = np.array([[1, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0]])
    H_press = np.array([[0, 0, 1, 0, 0]])

    R_gps = np.diag([gps_noise ** 2, gps_noise ** 2])
    R_press = np.diag([press_noise ** 2])

    # Transition matrix (simple constant-velocity model)
    F = np.eye(5)
    F[2, 4] = dt  # z integrates z_dot

    total_steps = int(T / dt)
    uncertainty_trace = []

    for k in range(total_steps):
        # ---- Prediction ----
        P = F @ P @ F.T + Q

        # ---- Measurement update (Pressure always) ----
        S = H_press @ P @ H_press.T + R_press
        K = P @ H_press.T @ np.linalg.inv(S)
        P = (np.eye(5) - K @ H_press) @ P

        # ---- GPS update only when surfaced ----
        if k % int(surface_interval / dt) == 0:
            S = H_gps @ P @ H_gps.T + R_gps
            K = P @ H_gps.T @ np.linalg.inv(S)
            P = (np.eye(5) - K @ H_gps) @ P

        uncertainty_trace.append(np.trace(P))

    return np.mean(uncertainty_trace), np.trace(P)

def evaluate_all_combinations(specs):
    results = []
    for imu_name, imu in specs["IMUs"].items():
        for gps_name, gps in specs["GPS"].items():
            for press_name, press in specs["Pressure"].items():

                imu_sigma = imu_noise_from_spec(
                    imu["noise_density_ug_per_sqrtHz"], imu["rate_hz"]
                )

                suite = {
                    "imu_noise": imu_sigma,
                    "imu_rate": imu["rate_hz"],
                    "gps_noise": gps["pos_noise_std"],
                    "press_noise": press["depth_noise_std"],
                }

                mean_trace, final_trace = simulate_uuv(suite)
                total_cost = imu["cost"] + gps["cost"] + press["cost"]

                results.append({
                    "IMU": imu_name,
                    "GPS": gps_name,
                    "Pressure": press_name,
                    "MeanTrace": mean_trace,
                    "FinalTrace": final_trace,
                    "Cost": total_cost
                })

    # Normalize metrics for weighted cost function
    traces = np.array([r["FinalTrace"] for r in results])
    costs  = np.array([r["Cost"] for r in results])
    traces_norm = (traces - np.min(traces)) / (np.max(traces) - np.min(traces) + 1e-9)
    costs_norm  = (costs  - np.min(costs))  / (np.max(costs)  - np.min(costs)  + 1e-9)

    for i, r in enumerate(results):
        r["Score"] = (2/3) * traces_norm[i] + (1/3) * costs_norm[i]

    results.sort(key=lambda r: r["Score"])
    return results

if __name__ == "__main__":
    specs = load_sensor_specs("./sensor_specs.json")
    results = evaluate_all_combinations(specs)

    print("\n=== Sensor Suite Ranking (lower Score = better) ===")
    for r in results:
        print(f"{r['IMU']:<15} | {r['GPS']:<12} | {r['Pressure']:<12} "
              f"| Cost=${r['Cost']:<6.0f} | FinalTrace={r['FinalTrace']:.4f} | Score={r['Score']:.4f}")
