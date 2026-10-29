"""
EE430 Introduction to Robot Dynamics and Control - Experiment 3
Task 2 (Group Specific Task): Yaskawa Motoman HC10 (6-DoF Collaborative Robot)
Dynamic Modeling, Inertia Matrix Evaluation, and Step Torque Simulation

Citations for Yaskawa Motoman HC10 Parameters: https://tecnical.cat/wp-content/uploads/documents/Yas_HC10_EN.pdf
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import roboticstoolbox as rtb
from spatialmath import SE3

# =====================================================================
# 1. YASKAWA MOTOMAN HC10 KINEMATIC (DH) & DYNAMIC PARAMETERS
# =====================================================================
# Total Manipulator Mass = 47.0 kg | Payload = 10.0 kg | Reach = 1200 mm
# Joint order: [S, L, U, R, B, T]
HC10_DH = [
    # d [m],   a [m], alpha [rad],      mass [kg], CoM r [m],               I_diag [Ixx, Iyy, Izz] [kg.m^2], B [N.m.s/rad]
    (0.275,    0.000,  np.pi / 2.0,     12.50,     [0.00, -0.11,  0.00],    [0.165, 0.140, 0.110],           1.2),
    (0.000,    0.700,  0.0,             14.20,     [-0.35, 0.00,  0.10],    [0.052, 0.615, 0.615],           1.5),
    (0.000,    0.000,  np.pi / 2.0,      7.80,     [0.00,  0.00,  0.05],    [0.045, 0.045, 0.028],           1.0),
    (0.500,    0.000, -np.pi / 2.0,      6.50,     [0.00,  0.22,  0.00],    [0.145, 0.022, 0.145],           0.8),
    (0.162,    0.000,  np.pi / 2.0,      4.20,     [0.00,  0.00, -0.04],    [0.018, 0.018, 0.012],           0.5),
    (0.130,    0.000,  0.0,              1.80,     [0.00,  0.00, -0.05],    [0.004, 0.004, 0.003],           0.3),
]

JOINT_NAMES = ["Joint 1 (S)", "Joint 2 (L)", "Joint 3 (U)", "Joint 4 (R)", "Joint 5 (B)", "Joint 6 (T)"]


# =====================================================================
# 2. ROBOTICS TOOLBOX 6-DOF MODEL CONSTRUCTION
# =====================================================================
def build_hc10_rtb():
    links = []
    for d, a, alpha, m, r_com, I_diag, B_visc in HC10_DH:
        link = rtb.RevoluteDH(
            d=d, a=a, alpha=alpha,
            m=m, r=r_com,
            I=[I_diag[0], I_diag[1], I_diag[2], 0.0, 0.0, 0.0],
            B=B_visc, G=1.0
        )
        links.append(link)

    return rtb.DHRobot(links, name="Yaskawa_Motoman_HC10", gravity=[0.0, 0.0, 9.81])


def step_torque_hc10(t):
    if t >= 0.0:
        return np.array([25.0, 95.0, 35.0, 12.0, 8.0, 3.0])
    return np.zeros(6)


def hc10_forward_dynamics(t, state, robot_rtb):
    q = state[0:6]
    dq = state[6:12]
    tau = step_torque_hc10(t)
    ddq = robot_rtb.accel(q, dq, tau)
    return np.concatenate([dq, ddq])


# =====================================================================
# 3. SIMULATION & ANIMATION
# =====================================================================
def main():
    print("=== EE430 Experiment 3 (Group Task): Yaskawa Motoman HC10 6-DoF Simulation ===")
    robot_rtb = build_hc10_rtb()
    print(robot_rtb)

    # Evaluate and display 6x6 Inertia Matrix at Home Configuration
    q_home = np.array([0.0, np.pi / 4.0, -np.pi / 4.0, 0.0, np.pi / 6.0, 0.0])
    M_home = robot_rtb.inertia(q_home)
    np.set_printoptions(precision=4, suppress=True)
    print("\n6x6 Inertia Matrix M(q) at q_home [kg.m^2]:")
    print(M_home)

    # Solve forward dynamics ODE
    t_span = (0.0, 1.5)
    t_eval = np.linspace(t_span[0], t_span[1], 150)
    y0 = np.concatenate([q_home, np.zeros(6)])

    sol = solve_ivp(
        fun=lambda t, y: hc10_forward_dynamics(t, y, robot_rtb),
        t_span=t_span, y0=y0, method='RK45',
        t_eval=t_eval, rtol=1e-4, atol=1e-6
    )

    t = sol.t
    q_hist = sol.y[0:6, :]
    dq_hist = sol.y[6:12, :]
    ddq_hist = np.column_stack([
        hc10_forward_dynamics(t[i], sol.y[:, i], robot_rtb)[6:12] for i in range(len(t))
    ])

    # 1. Plot All 6 Joint Variables, Velocities, and Accelerations
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    fig.suptitle("Yaskawa Motoman HC10 (6-DoF) - Step Torque Response", fontsize=14, fontweight='bold')

    colors = plt.cm.tab10(np.linspace(0, 0.6, 6))
    for j in range(6):
        axes[0].plot(t, q_hist[j, :], label=JOINT_NAMES[j], color=colors[j], lw=1.8)
        axes[1].plot(t, dq_hist[j, :], label=JOINT_NAMES[j], color=colors[j], lw=1.8)
        axes[2].plot(t, ddq_hist[j, :], label=JOINT_NAMES[j], color=colors[j], lw=1.8)

    axes[0].set_ylabel("Position $q_i$ [rad]")
    axes[0].set_title("Joint Positions")
    axes[0].grid(True, linestyle=":", alpha=0.7)
    axes[0].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9)

    axes[1].set_ylabel("Velocity $\\dot{q}_i$ [rad/s]")
    axes[1].set_title("Joint Velocities")
    axes[1].grid(True, linestyle=":", alpha=0.7)

    axes[2].set_ylabel("Acceleration $\\ddot{q}_i$ [rad/s$^2$]")
    axes[2].set_xlabel("Time [s]")
    axes[2].set_title("Joint Accelerations")
    axes[2].grid(True, linestyle=":", alpha=0.7)

    plt.tight_layout()
    plt.savefig("task2_hc10_step_response.png", dpi=300)

    # 2. Sequential 3D Stick Figure Animation (erasing previous frame, 200 ms delay)
    fig_3d = plt.figure(figsize=(9, 8))
    ax_3d = fig_3d.add_subplot(111, projection='3d')

    # Precalculate end-effector path across simulation
    ee_traj = np.zeros((3, len(t)))
    for k in range(len(t)):
        T_ee = robot_rtb.fkine(q_hist[:, k])
        ee_traj[:, k] = T_ee.t

    sample_steps = np.linspace(0, len(t) - 1, 30, dtype=int)

    plt.ion()
    flag = False
    for step_i in sample_steps:
        ax_3d.cla()  # Clear previous stick figure frame

        # End-effector trail up to current step
        ax_3d.plot(ee_traj[0, :step_i+1], ee_traj[1, :step_i+1], ee_traj[2, :step_i+1],
                   color="gray", ls="--", lw=1.5, alpha=0.8, label="End-Effector Trail")

        # Compute positions of every joint along the chain
        pts = [np.zeros(3)]
        T_accum = SE3()
        for link, q_val in zip(robot_rtb.links, q_hist[:, step_i]):
            T_accum = T_accum * link.A(q_val)
            pts.append(T_accum.t)
        pts = np.array(pts)

        # Plot current stick figure manipulator
        ax_3d.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                   marker="o", markersize=6, lw=3.5, color="#1f77b4", label="HC10 Arm")
        ax_3d.scatter([0], [0], [0], color="black", s=60, marker="s", label="Base")

        ax_3d.set_title(f"Yaskawa HC10 3D Animation (t = {t[step_i]:.2f} s)",
                        fontsize=13, fontweight='bold')
        ax_3d.set_xlim(-1.0, 1.0)
        ax_3d.set_ylim(-1.0, 1.0)
        ax_3d.set_zlim(0.0, 1.5)
        ax_3d.set_xlabel("X [m]")
        ax_3d.set_ylabel("Y [m]")
        ax_3d.set_zlabel("Z [m]")
        ax_3d.legend(loc="upper left")

        plt.draw()
        if not flag:
            plt.show()
            _ = input("Press Enter to start the animation...")
            flag = True
        plt.pause(0.2)  # 200 ms delay between frames

    plt.ioff()
    plt.savefig("task2_hc10_stick_figure_3d.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()