import sympy as sp


def get_transformation_matrix(theta, d, a, alpha):
    """
    Computes the homogeneous transformation matrix between frames i-1 and i
    using standard Denavit-Hartenberg parameters.
    """
    return sp.Matrix(
        [
            [
                sp.cos(theta),
                -sp.cos(alpha) * sp.sin(theta),
                sp.sin(alpha) * sp.sin(theta),
                a * sp.cos(theta),
            ],
            [
                sp.sin(theta),
                sp.cos(alpha) * sp.cos(theta),
                -sp.sin(alpha) * sp.cos(theta),
                a * sp.sin(theta),
            ],
            [0, sp.sin(alpha), sp.cos(alpha), d],
            [0, 0, 0, 1],
        ]
    )


def compute_symbolic_fk():
    # Define symbolic variables for thetas and link lengths
    th1, th2, th3, th4, th5, th6, th7 = sp.symbols(
        "theta1 theta2 theta3 theta4 theta5 theta6 theta7", real=True
    )
    d1, d3, d5, d7 = sp.symbols('d1 d3 d5 d7', real=True)
    a3, a4, a6 = sp.symbols('a3 a4 a6', real=True)

    dh_table = [
        (th1, d1,   0, -sp.pi/2),
        (th2,  0,   0,  sp.pi/2),
        (th3, d3,  a3,  sp.pi/2),
        (th4,  0,  a4, -sp.pi/2),
        (th5, d5,   0,  sp.pi/2),
        (th6,  0,  a6,  sp.pi/2),
        (th7, d7,   0,        0)
    ]

    t_matrices = []
    t_0_7 = sp.eye(4)

    for i, (theta_val, d_val, a_val, alpha_val) in enumerate(dh_table, start=1):
        t_i = get_transformation_matrix(theta_val, d_val, a_val, alpha_val)
        t_matrices.append(t_i)
        t_0_7 = t_0_7 * t_i
        t_0_7 = t_0_7.applyfunc(sp.trigsimp)
        print(f"\nTransformation Matrix T_{i - 1}^{i}:")
        sp.pprint(t_i)

    t_0_7 = sp.simplify(t_0_7)
    print("FINAL HOMOGENEOUS TRANSFORMATION MATRIX (T_0^7)")
    sp.pprint(t_0_7)

    symbols_dict = {
        'thetas': (th1, th2, th3, th4, th5, th6, th7),
        'lengths': (d1, d3, d5, d7, a3, a4, a6)
    }
    return t_0_7, symbols_dict


if __name__ == "__main__":
    sp.init_printing(wrap_line=False)
    compute_symbolic_fk()
