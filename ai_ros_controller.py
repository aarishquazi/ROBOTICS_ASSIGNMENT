"""Natural-language OpenMANIPULATOR-X controller (ROS 1).

Run this file from a terminal where the ROS 1 distribution and the project's
catkin workspace have been sourced. All movement requires a manual YES.
"""

import numpy as np
import rospy
from sensor_msgs.msg import JointState
from open_manipulator_msgs.srv import SetJointPosition, SetJointPositionRequest
from open_manipulator_msgs.msg import JointPosition

from command_bridge import command_to_absolute_target
from kinematics import JOINT_LOWER, JOINT_UPPER
from main import interpret_command
from motion_planner import plan_motion
from speech_input import listen_and_translate


JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4"]
JOINT_STATE_TOPIC = "/joint_states"
JOINT_GOAL_SERVICE = "/goal_joint_space_path"
DEFAULT_PATH_TIME_SECONDS = 3.0
JOINT_STATE_TIMEOUT_SECONDS = 3.0

# Configured home joint pose, as selected for this project by the user.
# Values are radians. This is a joint-space target, not XYZ=(0, 0, 0).
HOME_Q = np.zeros(4, dtype=float)


class AirobotController:
    def __init__(self):
        rospy.init_node("openmanipulator_ai_controller")

        rospy.loginfo("Waiting for ROS service: %s", JOINT_GOAL_SERVICE)
        try:
            rospy.wait_for_service(JOINT_GOAL_SERVICE, timeout=30.0)
        except rospy.ROSException as error:
            raise RuntimeError(
                f"ROS service {JOINT_GOAL_SERVICE} was not available within 30 seconds. "
                "Start the OpenMANIPULATOR controller and check the service name."
            ) from error

        self.joint_service = rospy.ServiceProxy(
            JOINT_GOAL_SERVICE,
            SetJointPosition,
        )
        rospy.loginfo("AI + ROS controller is ready.")

    def get_current_joint_positions(self):
        """Read measured joint positions and return q in radians, ordered by name."""
        try:
            message = rospy.wait_for_message(
                JOINT_STATE_TOPIC,
                JointState,
                timeout=JOINT_STATE_TIMEOUT_SECONDS,
            )
        except rospy.ROSException as error:
            raise RuntimeError(
                f"No JointState message received on {JOINT_STATE_TOPIC}. "
                "Check that the robot state publisher/controller is running."
            ) from error

        if len(message.name) != len(message.position):
            raise RuntimeError("JointState name and position arrays have different lengths.")

        position_by_name = dict(zip(message.name, message.position))
        missing = [name for name in JOINT_NAMES if name not in position_by_name]
        if missing:
            raise RuntimeError(
                "JointState is missing required joints: " + ", ".join(missing)
            )

        q_current = np.array(
            [position_by_name[name] for name in JOINT_NAMES],
            dtype=float,
        )
        if q_current.shape != (4,) or not np.all(np.isfinite(q_current)):
            raise RuntimeError("ROS returned invalid current joint positions.")
        if np.any(q_current < JOINT_LOWER) or np.any(q_current > JOINT_UPPER):
            raise RuntimeError(
                "Measured joint positions fall outside the limits in kinematics.py. "
                "Check joint ordering, zero offsets, and configured limits."
            )
        return q_current

    def send_joint_command(self, q_goal, path_time=DEFAULT_PATH_TIME_SECONDS):
        """Send an absolute joint-angle goal (radians) to the ROS service."""
        q_goal = np.asarray(q_goal, dtype=float)
        if q_goal.shape != (4,) or not np.all(np.isfinite(q_goal)):
            raise ValueError("Joint goal must contain four finite angles.")
        if np.any(q_goal < JOINT_LOWER) or np.any(q_goal > JOINT_UPPER):
            raise ValueError("Joint goal is outside the configured joint limits.")
        if not np.isfinite(path_time) or path_time <= 0:
            raise ValueError("Path time must be a positive finite number of seconds.")

        request = SetJointPositionRequest()
        request.planning_group = "arm"

        joint_position = JointPosition()
        joint_position.joint_name = JOINT_NAMES
        joint_position.position = q_goal.tolist()

        request.joint_position = joint_position
        request.path_time = float(path_time)

        response = self.joint_service(request)
        if response.is_planned:
            print("ROS controller accepted the joint-space path request.")
            print("Note: acceptance is not a feedback signal that motion has finished.")
        else:
            print("ROS controller did not plan the requested joint-space path.")
        return bool(response.is_planned)

    def confirm_and_send(self, q_goal, description):
        """Preview a joint goal and send it only after explicit confirmation."""
        print("\nMovement preview:")
        print(description)
        print("Target joint angles (radians):", np.round(q_goal, 5).tolist())
        print("Target joint angles (degrees):", np.round(np.degrees(q_goal), 3).tolist())
        confirmation = input("Type YES to send this joint goal: ").strip()
        if confirmation != "YES":
            print("Movement cancelled. No joint command was sent.")
            return False
        return self.send_joint_command(q_goal)

    def execute_text_command(self, text):
        """Interpret one sentence and safely route it to the existing planner."""
        command = interpret_command(text)

        if command.action == "clarify" or command.needs_clarification:
            print("I need clarification:", command.question)
            return

        if command.action == "stop":
            print(
                "STOP is not implemented by the supplied ROS interface. "
                "No joint target was sent. Do not rely on this text command as an "
                "emergency stop; use the robot/controller's actual stop mechanism."
            )
            return

        if command.action == "go_home":
            self.confirm_and_send(
                HOME_Q.copy(),
                "Return to the configured home joint pose q=[0, 0, 0, 0] radians.",
            )
            return

        if command.action not in ("move_absolute", "move_relative"):
            print(f"Unsupported command action: {command.action}")
            return

        # Read measured joints for both motion types. For absolute commands this
        # is the IK starting guess; for relative commands it is also used by FK
        # to determine the current end-effector XYZ position.
        q_current = self.get_current_joint_positions()
        target_xyz, description = command_to_absolute_target(command, q_current)

        # plan_motion accepts an ABSOLUTE Cartesian XYZ target in metres.
        # initial_q is only the starting guess for the IK iterations.
        result = plan_motion(target_xyz, initial_q=q_current)
        if not result.get("success", False):
            print("Motion planning failed:", result.get("message", "Unknown planning error."))
            return

        q_goal = np.asarray(result["joint_angles"], dtype=float)
        if q_goal.shape != (4,) or not np.all(np.isfinite(q_goal)):
            print("Planner returned an invalid joint goal. No command was sent.")
            return

        print("\nMotion plan verified by the project's kinematics model.")
        print("Absolute target XYZ (metres):", np.round(target_xyz, 6).tolist())
        print(
            "FK-achieved XYZ (metres):",
            np.round(result["achieved_position"], 6).tolist(),
        )
        print("FK position error (mm):", round(float(result["error_mm"]), 6))
        print("IK iterations:", result["iterations"])

        self.confirm_and_send(q_goal, description)

    @staticmethod
    def print_help():
        """Show command examples, coordinates, and supported actions."""
        print(
            "\nCOMMAND GUIDE\n"
            "\n"
            "1) ABSOLUTE POSITION — give the exact final XYZ destination:\n"
            "   Move to x=0.20 m, y=0.05 m, z=0.15 m\n"
            "   (You can also say 20 cm, 5 cm, and 15 cm with the axis names.)\n"
            "\n"
            "2) RELATIVE MOVEMENT — say how far to move from the current position:\n"
            "   Move 5 cm forward\n"
            "   Move 3 cm left\n"
            "   Move 2 cm up\n"
            "   Move 5 cm backward\n"
            "\n"
            "3) HOME POSITION:\n"
            "   Go home\n"
            "   Return to home position\n"
            "\n"
            "4) STOP:\n"
            "   Stop\n"
            "   NOTE: Stop is recognized but is NOT connected to a real controller\n"
            "   stop or emergency-stop function. Use the robot's actual stop control.\n"
            "\n"
            "COORDINATE DIRECTIONS (robot base frame):\n"
            "   +X = forward     -X = backward\n"
            "   +Y = left        -Y = right\n"
            "   +Z = up          -Z = down\n"
            "\n"
            "UNITS: metres (m), centimetres (cm), or millimetres (mm).\n"
            "For absolute movement, include all three coordinates X, Y, and Z.\n"
            "For relative movement, include a direction AND a distance.\n"
            "\n"
            "NOT IMPLEMENTED: gripper open/close, pick-and-place, obstacle avoidance,\n"
            "and a software emergency-stop command.\n"
            "\n"
            "Every planned movement is previewed. Type YES to send it to ROS;\n"
            "anything else cancels the movement.\n"
        )

    def _run_text_input(self):
        user_text = input("\nType your robot command (or BACK for menu): ").strip()
        if user_text.lower() in {"back", "menu"}:
            return
        if not user_text:
            print("No command entered.")
            return
        self.execute_text_command(user_text)

    def _run_voice_input(self):
        """Capture speech, display Whisper's language/English text, then ask permission."""
        speech = listen_and_translate()
        print("\nWhisper speech recognition result")
        print(f"Detected spoken language: {speech.language_name} ({speech.language_code})")
        print(f"English translation: {speech.english_text}")
        print("\nCheck the English translation carefully, especially distances and directions.")
        approval = input("Type USE to interpret this command, or press Enter to discard: ").strip()
        if approval != "USE":
            print("Voice command discarded. No robot command was sent.")
            return
        self.execute_text_command(speech.english_text)

    def run(self):
        """Run a friendly terminal menu with typed and optional voice input."""
        print("\n" + "=" * 66)
        print("       OPENMANIPULATOR-X  |  AI ROBOT COMMAND INTERFACE")
        print("=" * 66)
        print("You can control the robot by typing a command or speaking.")
        print("Whisper voice mode: automatically detects the spoken language and")
        print("translates it into English before the AI interprets the command.")
        print("\nFor examples and the coordinate directions, choose Help.")

        while not rospy.is_shutdown():
            print("\n" + "-" * 66)
            print("CHOOSE INPUT MODE")
            print("  [1] Type a command")
            print("  [2] Speak a command (Whisper; optional installation)")
            print("  [H] Show command guide and examples")
            print("  [Q] Quit")
            choice = input("\nYour choice [1/2/H/Q]: ").strip().lower()

            if choice in {"q", "quit", "exit"}:
                print("Exiting AI robot command interface.")
                break
            if choice in {"h", "help", "?"}:
                self.print_help()
                continue
            if choice in {"1", "t", "text"}:
                try:
                    self._run_text_input()
                except (rospy.ROSException, rospy.ServiceException, ValueError, RuntimeError) as error:
                    print("\nCommand was not executed:", error)
                except Exception as error:
                    print("\nUnexpected error; no further command was sent:", error)
                continue
            if choice in {"2", "v", "voice"}:
                try:
                    self._run_voice_input()
                except (rospy.ROSException, rospy.ServiceException, ValueError, RuntimeError) as error:
                    print("\nVoice command was not executed:", error)
                except Exception as error:
                    print("\nUnexpected voice-input error; no command was sent:", error)
                continue

            print("Please choose 1, 2, H, or Q.")


def main():
    try:
        controller = AirobotController()
        controller.run()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()
