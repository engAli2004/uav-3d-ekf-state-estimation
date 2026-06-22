import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Simulation Parameters
DT = 0.1  # time step (s)
SIM_TIME = 60.0  # total simulation time (s)

# Noise profiles
# Control noise: Body velocities [vx, vy, vz, yaw_rate]
Q = np.diag([0.1, 0.1, 0.1, np.deg2rad(2.0)])**2 
# Measurement noise: GPS [x, y, z]
R = np.diag([1.0, 1.0, 1.5])**2 

def motion_model(x_state, u_control):
    """
    Non-linear UAV Kinematics (Body frame velocities to Global position)
    x_state: [x, y, z, yaw]
    u_control: [v_body_x, v_body_y, v_body_z, yaw_rate]
    """
    yaw = x_state[3, 0]
    
    # Rotation matrix from body to global (2D rotation for XY, Z is direct)
    cos_y = np.cos(yaw)
    sin_y = np.sin(yaw)
    
    # Extract controls
    v_bx = u_control[0, 0]
    v_by = u_control[1, 0]
    v_bz = u_control[2, 0]
    omega = u_control[3, 0]
    
    # State update
    x_next = np.zeros((4, 1))
    x_next[0, 0] = x_state[0, 0] + (v_bx * cos_y - v_by * sin_y) * DT
    x_next[1, 0] = x_state[1, 0] + (v_bx * sin_y + v_by * cos_y) * DT
    x_next[2, 0] = x_state[2, 0] + v_bz * DT
    x_next[3, 0] = x_state[3, 0] + omega * DT
    
    return x_next

def observation_model(x_state):
    """Observation Model: UAV only observes X, Y, Z from GPS/Barometer"""
    H = np.array([[1.0, 0.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0, 0.0],
                  [0.0, 0.0, 1.0, 0.0]])
    return H @ x_state

def jacobian_f(x_state, u_control):
    """Jacobian of the non-linear UAV motion model"""
    yaw = x_state[3, 0]
    v_bx = u_control[0, 0]
    v_by = u_control[1, 0]
    
    jF = np.array([
        [1.0, 0.0, 0.0, (-v_bx * np.sin(yaw) - v_by * np.cos(yaw)) * DT],
        [0.0, 1.0, 0.0, ( v_bx * np.cos(yaw) - v_by * np.sin(yaw)) * DT],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    return jF

def ekf_estimation(x_est, P_est, z, u, Q, R):
    # Predict Step
    x_pred = motion_model(x_est, u)
    jF = jacobian_f(x_est, u)
    P_pred = jF @ P_est @ jF.T + Q

    # Update Step
    H = np.array([[1.0, 0.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0, 0.0],
                  [0.0, 0.0, 1.0, 0.0]])
    
    z_pred = observation_model(x_pred)
    y = z - z_pred  # Innovation error
    S = H @ P_pred @ H.T + R  # Innovation covariance
    K = P_pred @ H.T @ np.linalg.inv(S)  # Kalman Gain
    
    x_est = x_pred + K @ y
    P_est = (np.eye(len(x_est)) - K @ H) @ P_pred
    return x_est, P_est

def main():
    time = 0.0
    # Initial States: [x, y, z, yaw]
    x_true = np.zeros((4, 1))
    x_est = np.zeros((4, 1))
    P_est = np.eye(4)
    
    # History logs for 3D plotting
    h_x_true, h_x_est, h_z = x_true, x_est, np.zeros((3, 1))

    while time <= SIM_TIME:
        time += DT
        # UAV Control input: [v_forward, v_lateral, v_up, yaw_rate]
        # This creates a climbing spiral
        u = np.array([[2.0], [0.0], [0.5], [0.2]]) 
        
        # Ground Truth Trajectory
        x_true = motion_model(x_true, u)
        
        # Add noise to simulate real-world corrupted GPS / Barometer data
        z = observation_model(x_true) + np.random.randn(3, 1) * np.diag(np.sqrt(R))
        
        # Run EKF
        x_est, P_est = ekf_estimation(x_est, P_est, z, u, Q, R)

        # Store logs
        h_x_true = np.hstack((h_x_true, x_true))
        h_x_est = np.hstack((h_x_est, x_est))
        h_z = np.hstack((h_z, z))

    # Plotting the 3D results
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(h_z[0, :], h_z[1, :], h_z[2, :], c='g', marker='.', label="Noisy GPS", alpha=0.2)
    ax.plot(h_x_true[0, :], h_x_true[1, :], h_x_true[2, :], 'b-', label="Ground Truth Flight Path", linewidth=2)
    ax.plot(h_x_est[0, :], h_x_est[1, :], h_x_est[2, :], 'r-', label="EKF Estimated State", linewidth=2)
    
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_zlabel('Altitude Z (m)')
    ax.legend()
    plt.title("UAV 3D Extended Kalman Filter State Estimation")
    plt.savefig("uav_ekf_3d.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()