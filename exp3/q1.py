"""
EE430 Introduction to Robot Dynamics and Control - Experiment 3
Task 1 (Individual Lab Component): Dynamic Modeling and Simulation of a 2-DoF Planar RR Robot
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import roboticstoolbox as rtb
from spatialmath import SE3

# =====================================================================
# 1. PHYSICAL PARAMETERS (ALUMINUM 2-DOF PLANAR RR ROBOT)
# =====================================================================
rho = 2700.0          # Density of Aluminum [kg/m^3]
width = 0.05          # Cross-section width [m] (5 cm)
height = 0.05         # Cross-section height [m] (5 cm)
L1, L2 = 0.5, 0.5     # Link lengths [m]
g = 9.81              # Acceleration due to gravity [m/s^2]

volume = width * height * L1                 # 0.00125 m^3
m1 = m2 = rho * volume                       # 3.375 kg per link
lc1, lc2 = L1 / 2.0, L2 / 2.0                # 0.25 m

Ixx = (1.0 / 12.0) * m1 * (width**2 + height**2)   # 0.00140625 kg.m^2
Iyy = (1.0 / 12.0) * m1 * (L1**2 + width**2)       # 0.071015625 kg.m^2
Izz = (1.0 / 12.0) * m1 * (L1**2 + height**2)      # 0.071015625 kg.m^2


# =====================================================================
# 2. ROBOTICS TOOLBOX MODEL DEFINITION (DHRobot & RevoluteDH)
# =====================================================================
def build_rtb_2dof_robot():
    link1 = rtb.RevoluteDH(
        d=0.0, a=L1, alpha=0.0, m=m1,
        r=[-lc1, 0.0, 0.0],
        I=[Ixx, Iyy, Izz, 0.0, 0.0, 0.0],
        B=0.0, G=1.0
    )
    link2 = rtb.RevoluteDH(
        d=0.0, a=L2, alpha=0.0, m=m2,
        r=[-lc2, 0.0, 0.0],
        I=[Ixx, Iyy, Izz, 0.0, 0.0, 0.0],
        B=0.0, G=1.0
    )
    return rtb.DHRobot([link1, link2], name="Planar_2DoF_RR_Aluminum", gravity=[0.0, g, 0.0])


def step_torque_input(t):
    return np.array([15.0, 5.0]) if t >= 0.0 else np.array([0.0, 0.0])


def forward_dynamics_rhs(t, state, robot_rtb):
    q = state[0:2]
    dq = state[2:4]
    tau = step_torque_input(t)
    ddq = robot_rtb.accel(q, dq, tau)
    return np.concatenate([dq, ddq])


# =====================================================================
# 3. SIMULATION & ANIMATION
# =====================================================================
def main():
    robot_rtb = build_rtb_2dof_robot()
    print(robot_rtb)

    t_span = (0.0, 3.0)
    t_eval = np.linspace(t_span[0], t_span[1], 600)
    initial_state = np.zeros(4)

    sol = solve_ivp(
        fun=lambda t, y: forward_dynamics_rhs(t, y, robot_rtb),
        t_span=t_span, y0=initial_state, method='RK45',
        t_eval=t_eval, rtol=1e-6, atol=1e-8
    )

    t = sol.t
    q_hist = sol.y[0:2, :]
    dq_hist = sol.y[2:4, :]
    ddq_hist = np.column_stack([
        forward_dynamics_rhs(t[i], sol.y[:, i], robot_rtb)[2:4] for i in range(len(t))
    ])

    # 1. Plot Joint Variables, Velocities, and Accelerations
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    fig.suptitle("2-DoF Planar RR Robot (Aluminum) - Step Torque Response ($\\tau = [15, 5]$ N$\\cdot$m)",
                 fontsize=13, fontweight="bold")

    labels = [("Angle [rad]", q_hist, "Joint Positions", "$q_1(t)$", "$q_2(t)$"),
              ("Velocity [rad/s]", dq_hist, "Joint Velocities", "$\\dot{q}_1(t)$", "$\\dot{q}_2(t)$"),
              ("Acceleration [rad/s$^2$]", ddq_hist, "Joint Accelerations", "$\\ddot{q}_1(t)$", "$\\ddot{q}_2(t)$")]

    for ax, (ylabel, data, title, l1, l2) in zip(axes, labels):
        ax.plot(t, data[0, :], label=l1, color="#1f77b4", lw=2)
        ax.plot(t, data[1, :], label=l2, color="#ff7f0e", lw=2, ls="--")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, linestyle=":", alpha=0.7)
        ax.legend(loc="upper right")

    axes[2].set_xlabel("Time [s]")
    plt.tight_layout()
    plt.savefig("task1_2dof_step_response.png", dpi=300)

    # 2. Sequential Stick Figure Animation (erasing previous frame, 200 ms delay)
    fig_stick, ax_stick = plt.subplots(figsize=(8, 8))

    x1_all = L1 * np.cos(q_hist[0, :])
    y1_all = L1 * np.sin(q_hist[0, :])
    x2_all = x1_all + L2 * np.cos(q_hist[0, :] + q_hist[1, :])
    y2_all = y1_all + L2 * np.sin(q_hist[0, :] + q_hist[1, :])

    sample_indices = np.linspace(0, len(t) - 1, 30, dtype=int)

    plt.ion()
    flag = False
    for idx in sample_indices:
        ax_stick.cla()  # Erase previous stick figure

        # Progressive end-effector trajectory trail
        ax_stick.plot(x2_all[:idx+1], y2_all[:idx+1], color="gray", ls="--",
                      lw=1.5, alpha=0.7, label="End-Effector Trail")

        # Current robot stick figure
        ax_stick.plot([0.0, x1_all[idx], x2_all[idx]], [0.0, y1_all[idx], y2_all[idx]],
                      marker="o", markersize=8, color="#1f77b4", lw=4.0, label="Robot Links")
        ax_stick.plot(0.0, 0.0, marker="s", markersize=10, color="black", label="Base (0,0)")

        ax_stick.set_title(f"2-DoF Planar RR Robot Animation (t = {t[idx]:.2f} s)",
                           fontsize=13, fontweight="bold")
        ax_stick.set_xlim(-1.15, 1.15)
        ax_stick.set_ylim(-1.15, 1.15)
        ax_stick.set_aspect("equal", adjustable="box")
        ax_stick.set_xlabel("X Position [m]")
        ax_stick.set_ylabel("Y Position [m]")
        ax_stick.grid(True, linestyle="--", alpha=0.6)
        ax_stick.legend(loc="upper right")

        plt.draw()
        if not flag:
            plt.show()
            _ = input("Press Enter to start the animation...")
            flag = True
        plt.pause(0.2)  # 200 ms delay

    plt.ioff()
    plt.savefig("task1_2dof_stick_figure.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()