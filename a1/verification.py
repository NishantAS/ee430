import numpy as np
import roboticstoolbox as rtb
from symbolic_fk import compute_symbolic_fk

LENGTH_VALUES = {
    'd1': 0.3330,
    'd3': 0.3160,
    'd5': 0.3840,
    'd7': 0.1070,
    'a3': 0.0825,
    'a4': -0.0825,
    'a6': 0.0880
}

CONFIGS = {
    "(a) All joint angles zero": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "(b) theta2 = pi/2, other angles = 0": [0.0, np.pi/2, 0.0, 0.0, 0.0, 0.0, 0.0],
    "(c) theta2 = theta4 = pi/2, other angles = 0": [0.0, np.pi/2, 0.0, np.pi/2, 0.0, 0.0, 0.0]
}

def run_verification():
    t_0_7, syms = compute_symbolic_fk()
    th1, th2, th3, th4, th5, th6, th7 = syms['thetas']
    d1, d3, d5, d7, a3, a4, a6 = syms['lengths']

    length_subs = {
        d1: LENGTH_VALUES['d1'],
        d3: LENGTH_VALUES['d3'],
        d5: LENGTH_VALUES['d5'],
        d7: LENGTH_VALUES['d7'],
        a3: LENGTH_VALUES['a3'],
        a4: LENGTH_VALUES['a4'],
        a6: LENGTH_VALUES['a6']
    }

    # 3. Update the Corke Toolbox links definition
    links = [
        rtb.RevoluteDH(d=LENGTH_VALUES['d1'], a=0.0,                  alpha=-np.pi/2),
        rtb.RevoluteDH(d=0.0,                  a=0.0,                  alpha= np.pi/2),
        rtb.RevoluteDH(d=LENGTH_VALUES['d3'], a=LENGTH_VALUES['a3'],  alpha= np.pi/2),
        rtb.RevoluteDH(d=0.0,                  a=LENGTH_VALUES['a4'],  alpha=-np.pi/2),
        rtb.RevoluteDH(d=LENGTH_VALUES['d5'], a=0.0,                  alpha= np.pi/2),
        rtb.RevoluteDH(d=0.0,                  a=LENGTH_VALUES['a6'],  alpha= np.pi/2),
        rtb.RevoluteDH(d=LENGTH_VALUES['d7'], a=0.0,                  alpha= 0.0)
    ]
    panda = rtb.DHRobot(links, name="Franka Emika Panda")

    for config_name, q in CONFIGS.items():
        print(f"\nConfiguration: {config_name}")
        print(f"Joint Angles: {q}")

        angle_subs = {
            th1: q[0], th2: q[1], th3: q[2],
            th4: q[3], th5: q[4], th6: q[5], th7: q[6]
        }
        t_eval = t_0_7.subs({**length_subs, **angle_subs})
        t_v1 = np.array(t_eval, dtype=np.float64)

        print("\nVerification #1 (SymPy subs):")
        print(np.round(t_v1, 4))

        t_v2 = panda.fkine(q).A

        print("\nVerification #2 (Corke fkine):")
        print(np.round(t_v2, 4))

        max_error = np.max(np.abs(t_v1 - t_v2))
        print(f"Max Absolute Error: {max_error:.2e}")
        assert max_error < 1e-5, f"Mismatch in {config_name}!"
        print("Result: MATCH CONFIRMED")

if __name__ == '__main__':
    run_verification()