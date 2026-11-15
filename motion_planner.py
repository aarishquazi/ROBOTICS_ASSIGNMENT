
import numpy as np

from kinematics import (
    inverse_kinematics,
    get_position,
    JOINT_LOWER,
    JOINT_UPPER
)


def plan_motion(target, initial_q=None):

    target = np.asarray(target, dtype=float)

    # Validate Cartesian target
    if target.shape != (3,) or not np.all(np.isfinite(target)):
        return {
            "success": False,
            "message": "Invalid target coordinates."
        }

    # Solve inverse kinematics
    try:
        q, success, error, iterations = inverse_kinematics(
            target,
            initial_q=initial_q
        )
    except ValueError as e:
        return {
            "success": False,
            "message": str(e)
        }

    if not success:
        return {
            "success": False,
            "message": "IK failed: target could not be reached."
        }

    # Check joint limits
    if np.any(q < JOINT_LOWER) or np.any(q > JOINT_UPPER):
        return {
            "success": False,
            "message": "Calculated joint angles violate limits."
        }

    # Verify using forward kinematics
    achieved_position = get_position(q)

    verification_error = np.linalg.norm(
        target - achieved_position
    )

    if verification_error >= 1e-5:
        return {
            "success": False,
            "message": "FK verification failed."
        }

    return {
        "success": True,
        "message": "Motion plan generated successfully.",
        "joint_angles": q,
        "target": target,
        "achieved_position": achieved_position,
        "error_mm": verification_error * 1000,
        "iterations": iterations
    }


if __name__ == "__main__":

    print("OPENMANIPULATOR MOTION PLANNER")

    while True:

        command = input(
            "\nEnter x y z in metres (or exit): "
        ).strip()

        if command.lower() == "exit":
            break

        try:
            target = [float(v) for v in command.split()]

            result = plan_motion(target)

            print("\nMotion planning result:")
            print(result["message"])

            if result["success"]:
                print("Joint angles (radians):")
                print(result["joint_angles"])

                print("Joint angles (degrees):")
                print(np.degrees(result["joint_angles"]))

                print("Verified position:")
                print(result["achieved_position"])

                print("Position error (mm):")
                print(result["error_mm"])

                print("Iterations:")
                print(result["iterations"])

        except ValueError:
            print("Enter exactly three valid numeric coordinates.")

