"""Convert AI Cartesian commands into absolute base-frame XYZ targets.

The existing motion planner accepts absolute XYZ positions. This small module
bridges the AI's absolute/relative command formats to that interface.
"""

import numpy as np

from kinematics import get_position, JOINT_LOWER, JOINT_UPPER


_ZERO = np.zeros(3, dtype=float)


def command_to_absolute_target(command, current_q):
    """Return (absolute_target_xyz, description) for a movement command.

    `current_q` must be the measured current joint configuration, ordered as
    joint1..joint4, in radians. A relative command is converted using
    FK(current_q) + requested_displacement. An absolute command is passed
    through unchanged.
    """
    q = np.asarray(current_q, dtype=float)
    if q.shape != (4,) or not np.all(np.isfinite(q)):
        raise ValueError("Current joint state must contain four finite angles.")
    if np.any(q < JOINT_LOWER) or np.any(q > JOINT_UPPER):
        raise ValueError(
            "Measured joint state is outside the joint limits configured in kinematics.py."
        )

    action = getattr(command, "action", None)

    if action == "move_absolute":
        target = np.array(
            [command.target_x_m, command.target_y_m, command.target_z_m],
            dtype=float,
        )
        unused = np.array([command.dx_m, command.dy_m, command.dz_m], dtype=float)
        if not np.allclose(unused, _ZERO, atol=1e-9):
            raise ValueError("An absolute command must have zero relative displacement fields.")
        description = "Absolute base-frame target"

    elif action == "move_relative":
        displacement = np.array(
            [command.dx_m, command.dy_m, command.dz_m],
            dtype=float,
        )
        unused = np.array(
            [command.target_x_m, command.target_y_m, command.target_z_m],
            dtype=float,
        )
        if not np.allclose(unused, _ZERO, atol=1e-9):
            raise ValueError("A relative command must have zero absolute target fields.")
        current_xyz = get_position(q)
        target = current_xyz + displacement
        description = (
            "Relative displacement converted to absolute target; "
            f"current XYZ={current_xyz.tolist()} m, "
            f"displacement={displacement.tolist()} m"
        )

    else:
        raise ValueError(f"Action '{action}' is not a Cartesian movement command.")

    if target.shape != (3,) or not np.all(np.isfinite(target)):
        raise ValueError("Resolved Cartesian target must contain three finite coordinates.")

    return target, description
