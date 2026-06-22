# uav-3d-ekf-state-estimation
A NumPy-based 3D Extended Kalman Filter (EKF) for UAV state estimation fusing body-frame velocities and noisy GPS data.
# Autonomous UAV 3D Extended Kalman Filter (EKF)

This repository contains a lightweight, high-performance Python implementation of a 3D Extended Kalman Filter (EKF) designed specifically for Unmanned Aerial Vehicles (UAVs). It estimates the true 3D spatial trajectory of a drone by fusing noisy body-frame velocity setpoints (similar to PX4 offboard control inputs) with highly corrupted global GPS measurements.

**No external robotics libraries (like ROS or FilterPy) were used. The Jacobian matrices, non-linear state transitions, and covariance updates are derived and calculated completely from scratch using NumPy.**

## Performance Evaluation
The simulation tests the filter against an ascending 3D spiral trajectory. Heavy Gaussian noise is injected into the observation model to simulate real-world sensor degradation (e.g., GPS multi-path errors or barometer drift).

<img width="3000" height="2400" alt="uav_ekf_3d" src="https://github.com/user-attachments/assets/27481685-91ed-4ee3-ac3b-a2593227f986" />


As demonstrated by the tracking results, despite the high variance in raw sensor inputs, the custom EKF architecture dynamically calculates optimal Kalman gains to adhere directly to the actual ground-truth flight pathway.

---

## Mathematical Architecture

The EKF models a non-linear flight system where the drone receives body-frame velocity commands ($v_x, v_y, v_z$) and a yaw rate ($\omega$), which must be continuously rotated into the global coordinate frame.

### 1. Kinematic Prediction Model
The state vector is defined as $x=[x,y,z,\psi]^T$ (where $\psi$ is yaw). The non-linear transition dynamics based on control inputs $u=[v_{bx},v_{by},v_{bz},\omega]^T$ are mapped via:

$$x_k=f(x_{k-1},u_k)=\begin{bmatrix}x_{k-1}+(v_{bx}\cos\psi-v_{by}\sin\psi)\Delta t\\y_{k-1}+(v_{bx}\sin\psi+v_{by}\cos\psi)\Delta t\\z_{k-1}+v_{bz}\Delta t\\\psi_{k-1}+\omega\Delta t\end{bmatrix}$$

To propagate the state covariance matrix $P$, the non-linear motion dynamics are linearized at each time-step using the Jacobian matrix $G_k$:

$$G_k=\frac{\partial f}{\partial x}=\begin{bmatrix}1&0&0&(-v_{bx}\sin\psi-v_{by}\cos\psi)\Delta t\\0&1&0&(v_{bx}\cos\psi-v_{by}\sin\psi)\Delta t\\0&0&1&0\\0&0&0&1\end{bmatrix}$$

### 2. Measurement Update Loop
The measurement model assumes the UAV receives corrupted $X, Y, Z$ coordinates from an onboard GPS and Barometer module ($z=[x_{gps},y_{gps},z_{baro}]^T$). The measurement matrix $H$ remains strictly linear:

$$H=\begin{bmatrix}1&0&0&0\\0&1&0&0\\0&0&1&0\end{bmatrix}$$

---

## How to Run Locally

This project was built to be highly portable and requires no heavy robotics frameworks.

1. Clone the repository:
```bash
git clone [https://github.com/engAli2004/uav-3d-ekf-state-estimation.git](https://github.com/engAli2004/uav-3d-ekf-state-estimation.git)
cd uav-3d-ekf-state-estimation
