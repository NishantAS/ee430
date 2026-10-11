import math
import matplotlib.pyplot as plt
import numpy as np

# Define link lengths
L1 = 1.0
L2 = 1.0

# Initial guess
theta1 = np.radians(0)
theta2 = np.radians(1.5708)
q = np.array([theta1, theta2], dtype=float)

# Tolerance and No of iterations
tol = 0.001
iterations = 20

# Set up dynamic plot
plt.ion()
fig, ax = plt.subplots(figsize=(6, 6))

# Input handling
while True:
    try:
        inputflag = int(input("1 to choose target coordinate, 2 to choose target joint angles\n").strip())
        if inputflag not in (1, 2):
            print("Please enter 1 or 2.")
            continue

        inputvalueraw = input("Input target co-ordinate in x,y format or target angle in q1,q2 format (in degrees)\n")
        inputvalue = [float(val.strip()) for val in inputvalueraw.split(",")]
        
        if len(inputvalue) != 2:
            print("Please enter exactly two values.")
            continue

        if inputflag == 2:
            q1_rad = math.radians(inputvalue[0])
            q2_rad = math.radians(inputvalue[1])
            mutarget = [
                L1 * math.cos(q1_rad) + L2 * math.cos(q1_rad + q2_rad),
                L1 * math.sin(q1_rad) + L2 * math.sin(q1_rad + q2_rad)
            ]
        else:
            mutarget = inputvalue

        mutarget = np.array(mutarget, dtype=float)
        break
    except ValueError:
        print("Invalid entry. Try again.")

# Check workspace boundary
target_dist = np.linalg.norm(mutarget)
if target_dist > (L1 + L2) or target_dist < abs(L1 - L2):
    print(f"Warning: Target {mutarget} is outside reachable workspace range [{abs(L1-L2)}, {L1+L2}]!")

# Newton-Raphson Iteration
for i in range(iterations):
    # Joint positions calculation
    x0, y0 = 0.0, 0.0
    x1 = L1 * math.cos(q[0])
    y1 = L1 * math.sin(q[0])
    x2 = x1 + L2 * math.cos(q[0] + q[1])
    y2 = y1 + L2 * math.sin(q[0] + q[1])

    # Plotting current robot configuration
    ax.clear()
    ax.plot([x0, x1, x2], [y0, y1, y2], "-o", linewidth=3, markersize=8, color="blue")
    ax.plot(mutarget[0], mutarget[1], "rx", markersize=12, markeredgewidth=2, label="Target")

    ax.set_xlim([-(L1 + L2) - 0.5, (L1 + L2) + 0.5])
    ax.set_ylim([-(L1 + L2) - 0.5, (L1 + L2) + 0.5])
    ax.set_aspect("equal")
    ax.grid(True)
    ax.set_title(f"Iter {i}: q1 = {math.degrees(q[0]):.1f}°, q2 = {math.degrees(q[1]):.1f}°")
    ax.legend(loc="upper right")

    plt.draw()
    plt.pause(0.3)

    # Check for convergence
    muguess = np.array([x2, y2])
    delta = mutarget - muguess

    if np.linalg.norm(delta) < tol:
        print(f"\nConverged in {i} iterations!")
        print(f"Joint Angles: q1 = {math.degrees(q[0]):.2f}°, q2 = {math.degrees(q[1]):.2f}°")
        break

    # Analytical Jacobian matrix
    J11 = -L1 * math.sin(q[0]) - L2 * math.sin(q[0] + q[1])
    J12 = -L2 * math.sin(q[0] + q[1])
    J21 = L1 * math.cos(q[0]) + L2 * math.cos(q[0] + q[1])
    J22 = L2 * math.cos(q[0] + q[1])

    jacobian = np.array([[J11, J12], [J21, J22]])

    # Newton Update
    q += np.linalg.pinv(jacobian) @ delta
    
    # Wrap angles back to range [-pi, pi]
    q = np.arctan2(np.sin(q), np.cos(q))

plt.ioff()
plt.show()