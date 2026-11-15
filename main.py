"""Natural-language command interpreter for OpenMANIPULATOR-X.

This module converts text into a validated RobotCommand. It does not perform
kinematics and it does not send commands to ROS. The ROS integration entry
point is ai_ros_controller.py.
"""

from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

GROQ_API_KEY = "PASTE_YOUR_GROQ_API_KEY_HERE"

class RobotCommand(BaseModel):
    """One structured intent returned by the language model.

    move_absolute uses target_x_m/target_y_m/target_z_m.
    move_relative uses dx_m/dy_m/dz_m.
    For other actions all six coordinate values are zero.
    """

    model_config = ConfigDict(extra="forbid")

    action: Literal[
        "move_absolute",
        "move_relative",
        "go_home",
        "stop",
        "clarify",
    ] = Field(description="The single action requested by the user.")
    target_x_m: float = Field(description="Absolute base-frame X target in metres; otherwise zero.")
    target_y_m: float = Field(description="Absolute base-frame Y target in metres; otherwise zero.")
    target_z_m: float = Field(description="Absolute base-frame Z target in metres; otherwise zero.")
    dx_m: float = Field(description="Relative X displacement in metres; otherwise zero.")
    dy_m: float = Field(description="Relative Y displacement in metres; otherwise zero.")
    dz_m: float = Field(description="Relative Z displacement in metres; otherwise zero.")
    needs_clarification: bool = Field(description="True only when the user's request is unclear or unsupported.")
    question: str = Field(description="A concise clarification question, or an empty string.")


PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You interpret natural-language commands for an OpenMANIPULATOR-X robot.
Your job is ONLY to identify the user's intent and return the required fields.
Do not calculate joint angles, perform inverse kinematics, or claim that the
robot has moved.

Coordinate frame: robot base frame. +X = forward, +Y = left, +Z = up.
All distances must be converted to metres.

Choose exactly one action:
1. move_absolute: the user gives a final destination in the base frame, e.g.
   'move to x 0.20, y 0.05, z 0.15 metres'. Put it in target_x_m,
   target_y_m, target_z_m. Set dx_m, dy_m, dz_m to 0.
2. move_relative: the user gives a displacement from the CURRENT end-effector
   position, e.g. 'move 10 cm forward'. Put it in dx_m, dy_m, dz_m. Set all
   target_* fields to 0. Do not turn a relative displacement into a final XYZ.
3. go_home: the user clearly requests the configured home joint pose. Set all
   six coordinate fields to 0.
4. stop: the user clearly asks to stop. Set all six coordinate fields to 0.
   The software may report that a real controller stop function is not configured.
5. clarify: the request is ambiguous, missing information, or unsupported. Set
   all six coordinate fields to 0, set needs_clarification=true, and ask one
   concise question.

For move_absolute, move_relative, go_home, and stop, set
needs_clarification=false and question="".
Never invent missing distances or coordinates. If a movement direction is
specified without a distance, ask for the distance. If a coordinate target is
missing one or more coordinates, ask for the missing values.
""",
        ),
        ("human", "{command}"),
    ]
)


@lru_cache(maxsize=1)
def _get_command_chain():
    """Build the Groq/LangChain pipeline on first use."""
    if (
        not GROQ_API_KEY.strip()
        or GROQ_API_KEY.strip() == "PASTE_YOUR_GROQ_API_KEY_HERE"
    ):
        raise RuntimeError(
            "Groq API key is not configured. "
            "Open main.py and add your Groq API key."
        )

    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0,
    )
    structured_llm = llm.with_structured_output(
        RobotCommand,
        method="json_schema",
        strict=True,
    )
    return PROMPT | structured_llm


def interpret_command(command: str) -> RobotCommand:
    """Interpret one user sentence and return a validated RobotCommand."""
    if not command or not command.strip():
        raise ValueError("Command cannot be empty.")
    result = _get_command_chain().invoke({"command": command.strip()})
    if not isinstance(result, RobotCommand):
        # Defensive check in case the installed integration changes its return type.
        result = RobotCommand.model_validate(result)
    return result


def main() -> None:
    """Test AI command interpretation without ROS or robot movement."""
    print("\n" + "=" * 62)
    print("       OPENMANIPULATOR-X  |  AI COMMAND TEST")
    print("=" * 62)
    print("This test interprets commands only. It does NOT connect to ROS or move the robot.")
    print("\nTry commands such as:")
    print("  Absolute:  Move to x=0.20 m, y=0.05 m, z=0.15 m")
    print("  Relative:  Move 5 cm forward")
    print("  Relative:  Move 3 cm left")
    print("  Relative:  Move 2 cm up")
    print("  Home:      Go home")
    print("  Stop:      Stop (recognized only; no real stop is configured)")
    print("\nType 'help' to show these examples again, or 'exit' to quit.")

    while True:
        user_input = input("\nYour command > ").strip()
        if user_input.lower() in {"exit", "quit", "q"}:
            print("Goodbye.")
            break
        if user_input.lower() in {"help", "h", "?"}:
            print("Absolute: Move to x=0.20 m, y=0.05 m, z=0.15 m")
            print("Relative: Move 5 cm forward / Move 3 cm left / Move 2 cm up")
            print("Control:  Go home / Stop")
            continue
        if not user_input:
            continue
        try:
            result = interpret_command(user_input)
            print("\nAI interpretation:")
            print(result.model_dump_json(indent=2))
        except Exception as error:  # API/network/configuration errors are shown to the user.
            print(f"Could not interpret command: {error}")


if __name__ == "__main__":
    main()
