
import numpy as np


def translation(x, y, z):
    T = np.eye(4)
    T[0, 3] = x
    T[1, 3] = y
    T[2, 3] = z
    return T


def rotation_z(q):
    c, s = np.cos(q), np.sin(q)

    return np.array([
        [c, -s, 0, 0],
        [s,  c, 0, 0],
        [0,  0, 1, 0],
        [0,  0, 0, 1]
    ])


def rotation_y(q):
    c, s = np.cos(q), np.sin(q)

    return np.array([
        [ c, 0, s, 0],
        [ 0, 1, 0, 0],
        [-s, 0, c, 0],
        [ 0, 0, 0, 1]
    ])


def forward_kinematics(q):
    """
    q = [q1, q2, q3, q4] in radians.

    Returns the end-effector homogeneous transformation.
    """

    q1, q2, q3, q4 = q

    # Base to joint 1
    T1 = translation(0.012, 0.0, 0.017) @ rotation_z(q1)

    # Joint 1 to joint 2
    T2 = translation(0.0, 0.0, 0.0595) @ rotation_y(q2)

    # Joint 2 to joint 3
    T3 = translation(0.024, 0.0, 0.128) @ rotation_y(q3)

    # Joint 3 to joint 4
    T4 = translation(0.124, 0.0, 0.0) @ rotation_y(q4)

    # Joint 4 to end-effector
    T5 = translation(0.126, 0.0, 0.0)

    T_EE = T1 @ T2 @ T3 @ T4 @ T5

    return T_EE


def get_position(q):
    T = forward_kinematics(q)
    return T[:3, 3]
    
    

# Joint limits in radians (from the URDF)

JOINT_LOWER = np.array([
    -np.pi * 0.9,
    -np.pi * 0.57,
    -np.pi * 0.3,
    -np.pi * 0.57
])

JOINT_UPPER = np.array([
    np.pi * 0.9,
    np.pi * 0.5,
    np.pi * 0.44,
    np.pi * 0.65
])


def numerical_jacobian(q, epsilon=1e-6):
    """
    Calculate the 3x4 position Jacobian numerically.
    """

    J = np.zeros((3, 4))

    for i in range(4):

        q_plus = q.copy()
        q_minus = q.copy()

        q_plus[i] += epsilon
        q_minus[i] -= epsilon

        p_plus = get_position(q_plus)
        p_minus = get_position(q_minus)

        J[:, i] = (p_plus - p_minus) / (2 * epsilon)

    return J


def inverse_kinematics(target, initial_q=None,
                       max_iterations=1000, damping=0.03,
                       tolerance=1e-5):

    """
    Damped Least Squares inverse kinematics.

    target: desired [x, y, z] in metres
    initial_q: initial joint configuration in radians
    """

    if initial_q is None:
        q = np.zeros(4)
    else:
        q = np.array(initial_q, dtype=float)

    target = np.array(target, dtype=float)

    if target.shape != (3,) or not np.all(np.isfinite(target)):
        raise ValueError("Target must contain three finite coordinates.")

    if q.shape != (4,) or not np.all(np.isfinite(q)):
        raise ValueError("Initial joint configuration must contain four finite angles.")

    if np.any(q < JOINT_LOWER) or np.any(q > JOINT_UPPER):
        raise ValueError("Initial joint configuration violates joint limits.")

    for iteration in range(max_iterations):

        current_position = get_position(q)

        error = target - current_position

        error_norm = np.linalg.norm(error)

        if error_norm < tolerance:
            return q, True, error_norm, iteration

        J = numerical_jacobian(q)

        # Damped Least Squares update
        delta_q = J.T @ np.linalg.solve(
            J @ J.T + damping**2 * np.eye(3),
            error
        )

        # Prevent excessively large updates
        step_norm = np.linalg.norm(delta_q)

        if step_norm > 0.2:
            delta_q = delta_q * (0.2 / step_norm)

        # Apply joint limits
        q = np.clip(
            q + delta_q,
            JOINT_LOWER,
            JOINT_UPPER
        )

    final_error = np.linalg.norm(target - get_position(q))

    return q, final_error < tolerance, final_error, max_iterations









if __name__ == "__main__":

    print("OpenMANIPULATOR IK TEST")

    while True:

        command = input(
            "\nEnter target x y z in metres (or exit): "
        ).strip()

        if command.lower() == "exit":
            break

        try:
            target = np.array(
                [float(value) for value in command.split()]
            )

            if target.shape != (3,):
                print("Please enter exactly three values.")
                continue

            q, success, error, iterations = inverse_kinematics(
                target,
                initial_q=np.zeros(4)
            )

            if success:

                print("\nTarget reached mathematically!")

                print("Joint angles (degrees):")
                print(np.degrees(q))

                print("Achieved position:")
                print(get_position(q))

                print("Position error (mm):")
                print(error * 1000)

            else:
                print("\nIK could not reach the target.")
                print("Final position error (mm):")
                print(error * 1000)

        except ValueError:
            print("Invalid input. Enter three numeric coordinates.")

