"""EE430 Assignment 1: symbolic forward kinematics for Franka Emika Panda.

This program uses only SymPy matrix multiplication for the required forward
kinematics calculation.  It does not use a robotics/kinematics library.
"""

from __future__ import annotations

import sympy as sp


def dh_transform(theta: sp.Expr, d: sp.Expr, a: sp.Expr, alpha: sp.Expr) -> sp.Matrix:
    """Return the standard-DH transformation T_i_(i-1)."""
    c_theta, s_theta = sp.cos(theta), sp.sin(theta)
    c_alpha, s_alpha = sp.cos(alpha), sp.sin(alpha)
    return sp.Matrix(
        [
            [c_theta, -s_theta * c_alpha, s_theta * s_alpha, a * c_theta],
            [s_theta, c_theta * c_alpha, -c_theta * s_alpha, a * s_theta],
            [0, s_alpha, c_alpha, d],
            [0, 0, 0, 1],
        ]
    )


def panda_symbolic_model() -> tuple[tuple[sp.Symbol, ...], tuple[sp.Symbol, ...], list[sp.Matrix], sp.Matrix]:
    """Build the seven Panda standard-DH transforms and their product T_7_0."""
    theta = sp.symbols("theta1:8", real=True)
    d1, d3, d5, d7 = sp.symbols("d1 d3 d5 d7", real=True)
    a3, a4, a6 = sp.symbols("a3 a4 a6", real=True)

    # Table rows: theta_i, d_i, a_i, alpha_i.
    dh_rows = (
        (theta[0], d1, 0, -sp.pi / 2),
        (theta[1], 0, 0, sp.pi / 2),
        (theta[2], d3, a3, sp.pi / 2),
        (theta[3], 0, a4, -sp.pi / 2),
        (theta[4], d5, 0, sp.pi / 2),
        (theta[5], 0, a6, sp.pi / 2),
        (theta[6], d7, 0, 0),
    )
    transforms = [dh_transform(*row) for row in dh_rows]
    end_effector = sp.eye(4)
    for transform in transforms:
        end_effector *= transform
    return theta, (d1, d3, d5, d7, a3, a4, a6), transforms, end_effector


def numeric_substitutions(theta: tuple[sp.Symbol, ...], lengths: tuple[sp.Symbol, ...]) -> dict[sp.Symbol, float]:
    """Return SI geometric values from the assignment's Panda drawing."""
    d1, d3, d5, d7, a3, a4, a6 = lengths
    return {
        d1: 0.3330,
        d3: 0.3160,
        d5: 0.3840,
        d7: 0.1070,
        a3: 0.0825,
        a4: -0.0825,
        a6: 0.0880,
    }


def print_matrix(label: str, matrix: sp.Matrix, numeric: bool = False) -> None:
    """Print a readable exact or six-decimal matrix."""
    print(f"\n{label}")
    print(sp.N(matrix, 6) if numeric else matrix)


def main() -> None:
    theta, lengths, transforms, t_7_0 = panda_symbolic_model()
    dimensions = numeric_substitutions(theta, lengths)

    print("Panda forward kinematics using the standard DH convention")
    print("All distances are in metres.")
    for index, transform in enumerate(transforms, start=1):
        print_matrix(f"T_{index}_{index - 1} (symbolic)", transform)
    print_matrix("T_7_0 = T_1_0*T_2_1*...*T_7_6 (symbolic)", t_7_0)

    configurations = (
        ("Case A: theta1...theta7 = 0", {}),
        ("Case B: theta2 = pi/2; all other angles = 0", {theta[1]: sp.pi / 2}),
        (
            "Case C: theta2 = theta4 = pi/2; all other angles = 0",
            {theta[1]: sp.pi / 2, theta[3]: sp.pi / 2},
        ),
    )
    zero_angles = {joint: 0 for joint in theta}
    for name, changed_angles in configurations:
        substitutions = dimensions | zero_angles | changed_angles
        print_matrix(name, t_7_0.subs(substitutions), numeric=True)


if __name__ == "__main__":
    main()
