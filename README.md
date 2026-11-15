# OpenMANIPULATOR-X | AI + ROS + Whisper Voice Input

**Project package version:** 3 (integrated command interface)

This project connects a natural-language command interpreter to the existing OpenMANIPULATOR-X inverse-kinematics planner and ROS 1 joint-space service. The terminal supports typed commands and optional microphone input using OpenAI Whisper.

## 1. What the project can do

| User intent | Example | What the software does |
|---|---|---|
| Absolute XYZ destination | `Move to x=0.20 m, y=0.05 m, z=0.15 m` | Treats the coordinates as the final end-effector position in the robot base frame. |
| Relative movement | `Move 5 cm forward` | Reads the current joint positions, calculates the current end-effector position using FK, adds the requested displacement, and plans to the resulting absolute XYZ target. |
| Relative movement | `Move 3 cm left` | Converts the leftward displacement into an absolute target using the current position. |
| Relative movement | `Move 2 cm up` | Converts the upward displacement into an absolute target using the current position. |
| Home joint pose | `Go home` | Requests the configured joint-space pose `[0, 0, 0, 0]` radians. |
| Stop request | `Stop` | Recognizes the request but does not send a controller stop command; no stop service is configured. |

Robot base-frame directions:

- `+X` = forward; `-X` = backward
- `+Y` = left; `-Y` = right
- `+Z` = up; `-Z` = down

Distances are represented internally in metres. For an absolute target, provide all three coordinates X, Y, and Z. For a relative movement, provide both a direction and a distance.

**Not implemented:** gripper open/close, pick-and-place, obstacle avoidance, collision checking, workspace safety checking, or a software emergency-stop function.

## 2. Project files and their roles

### Main integrated application

| File | Role |
|---|---|
| `ai_ros_controller.py` | Main integrated entry point. Displays the terminal menu, reads current joint states, coordinates AI interpretation and motion planning, previews motion, and sends confirmed joint goals to ROS. |
| `main.py` | Uses Groq and LangChain to convert English text into a validated `RobotCommand`. It does not perform kinematics or send ROS commands. |
| `speech_input.py` | Optional microphone recording and local Whisper language detection / translation into English. |
| `command_bridge.py` | Converts AI movement commands into an absolute Cartesian XYZ target. Absolute targets pass through; relative displacements are added to the current FK position. |
| `motion_planner.py` | Calls inverse kinematics and verifies the achieved endpoint position using forward kinematics. |
| `kinematics.py` | Implements forward kinematics (FK), a numerical Jacobian, Damped Least Squares (DLS) inverse kinematics (IK), and configured joint limits. |

### Supporting files

| File | Role |
|---|---|
| `requirements.txt` | Python dependencies for typed AI command interpretation and numerical planning. |
| `requirements-voice.txt` | Optional Whisper and microphone dependencies. Install this only when using speech input. |
| `.gitignore` | Excludes local secrets, virtual environments, and Python cache files from version control. |
| `tests/` | Unit tests for the command bridge, motion planner, and speech-input configuration. |
| `VERSION.txt` | Package version and build-verification notes. |
| `main (1).py` | Original numeric-only ROS command-line controller. It is kept for reference and is not the integrated AI entry point. |

Use `ai_ros_controller.py` for the integrated AI interface. `main.py` can be run separately to test AI interpretation without connecting to ROS. `main (1).py` is the older numerical-only ROS interface.

## 3. Installation and quick start

### 3.1 Check the ROS/Python environment first

This project uses ROS 1 Python APIs: `rospy`, `sensor_msgs`, and `open_manipulator_msgs`. The Python interpreter used to run the project must be able to import those ROS packages.

Check the active ROS distribution and Python version:

```bash
 echo "$ROS_DISTRO"
 python3 --version
 python3 -c "import rospy, sensor_msgs, open_manipulator_msgs; print('ROS Python packages are available')"
```

If the import check fails, source the correct ROS distribution and catkin workspace before continuing. For example, adjust the distribution and workspace path to match your installation:

```bash
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash
```

Do not assume that a new virtual environment automatically has access to your ROS packages. A virtual environment is optional; what matters is that the Python interpreter running the program can import the ROS packages and is compatible with the project dependencies. On operating systems that manage Python packages externally, follow the OS guidance rather than forcing a global pip install with `--break-system-packages`.

**Version compatibility warning:** `requirements.txt` specifies `langchain-groq>=0.3.8`, which requires Python 3.9 or newer. Standard ROS Noetic installations commonly use Python 3.8. If your ROS distribution uses Python 3.8, do not install the listed requirements blindly; the dependency versions or runtime arrangement must be adjusted to a compatible combination first.

### 3.2 Start ROS and the robot controller

Start the ROS master and the OpenMANIPULATOR-X description/controller and Gazebo bringup (or the appropriate robot bringup for your setup). In the same terminal used to run the application, source the ROS distribution and workspace.

Check that the required joint-state topic and joint-space service are available:

```bash
rostopic echo -n 1 /joint_states
rosservice list | grep goal_joint_space_path
rossrv show open_manipulator_msgs/SetJointPosition
```

The code expects `/joint_states` to contain the joints named `joint1`, `joint2`, `joint3`, and `joint4`. It sends joint goals through `/goal_joint_space_path` using `open_manipulator_msgs/SetJointPosition` and the planning group `arm`. Confirm these match your installed robot controller.

### 3.3 Configure the Groq API key

### Groq API Key Configuration

The Groq API key is configured directly inside main.py.

1. Open main.py.
2. Find the GROQ_API_KEY variable.
3. Paste your actual Groq API key.

Example:

```python
GROQ_API_KEY = "your_actual_groq_api_key"
```

### 3.4 Install the core Python requirements

Install `requirements.txt` into the same ROS-compatible Python environment that will run the application:

```bash
python3 -m pip install -r requirements.txt
```

ROS packages such as `rospy`, `sensor_msgs`, and `open_manipulator_msgs` are supplied by the ROS installation/workspace; they are not installed by this pip requirements file.

### 3.5 Run the integrated application

```bash
python3 ai_ros_controller.py
```

The terminal menu offers:

```text
[1] Type a command
[2] Speak a command (Whisper; optional installation)
[H] Show command guide and examples
[Q] Quit
```

In the movement preview, type the exact word `YES` to send the joint goal. Any other input cancels the movement.

## 4. Optional speech input with Whisper

Voice input uses the open-source multilingual Whisper model locally. It attempts to identify the spoken language automatically and translates the recording into English. The English translation is then passed to the Groq command interpreter.

### 4.1 Install audio system packages

On Ubuntu:

```bash
sudo apt update
sudo apt install ffmpeg libportaudio2
```

Then install the optional Python dependencies into the same Python environment used by the ROS application:

```bash
python3 -m pip install -r requirements-voice.txt
```

Typed input does not require Whisper or the voice dependencies.

### 4.2 Whisper model and language behaviour

The default model is `small`, configured by `WHISPER_MODEL=small`.The supported model choices in this project are `tiny`, `base`, `small`, `medium`, and `large`.

The code calls Whisper with:

```python
model.transcribe(
    audio_path,
    task="translate",
    language=None,
    verbose=False,
)
```

`language=None` asks Whisper to detect the spoken language. `task="translate"` requests an English translation. Use a multilingual model for this workflow; do not select `turbo`, because it is not suitable for the translation task described here.

The first use downloads the selected model weights. Later runs normally reuse the cached model files. The model is loaded into memory when needed; it is cached and reused during the running application session. Downloading another model or clearing the cache requires another download.

The microphone is opened at the default input device's reported native sample rate. Whisper's audio loader/FFmpeg resamples the saved WAV internally as required.

### 4.3 Use voice mode

1. Start `python3 ai_ros_controller.py`.
2. Choose `[2] Speak a command`.
3. Press Enter when ready, speak one command, then press Enter to stop recording.
4. The terminal displays the detected language and Whisper's English translation.
5. Check the translation carefully, especially distances and directions. Type `USE` to pass the translation to the AI interpreter; press Enter to discard it.
6. If a movement is planned, inspect the XYZ and joint-angle preview. Type `YES` to send the joint goal to ROS.

The `USE` and `YES` prompts are separate on purpose: the first approves the transcript for AI interpretation; the second approves the planned robot movement.

### 4.4 Accuracy, privacy, and network use

Automatic language identification and translation are not perfect, especially for very short speech, background noise, strong accents, code-switching, and spoken numbers or directions. Always review the English translation.

The audio is recorded and processed locally by Whisper after the model is installed. The resulting English command text is sent to the configured Groq API for structured command interpretation. Do not use this setup for private speech unless that data flow is acceptable.

If the microphone is not detected, check operating-system microphone permissions, the selected input device, and PortAudio installation.

## 5. How the integrated command flow works

### 5.1 Absolute movement

Example user command:

```text
Move to x=0.20 m, y=0.05 m, z=0.15 m
```

1. `main.py` sends the text to Groq through LangChain and validates the structured response as a `RobotCommand` with action `move_absolute`.
2. The absolute XYZ values are stored in `target_x_m`, `target_y_m`, and `target_z_m`.
3. `ai_ros_controller.py` reads the measured joint positions from `/joint_states`.
4. `command_bridge.py` passes the absolute XYZ target through unchanged.
5. `plan_motion(target_xyz, initial_q=q_current)` calls IK, using the measured current joint angles as the solver's starting guess.
6. The planner verifies the achieved position with FK and returns the candidate joint angles.
7. The controller displays the target and joint-angle preview. Only an exact `YES` sends the joint goal to ROS.

### 5.2 Relative movement

Example user command:

```text
Move 5 cm forward
```

1. `main.py` returns action `move_relative` and a displacement such as `dx_m=0.05`.
2. `ai_ros_controller.py` reads the measured current joint configuration from `/joint_states`.
3. `command_bridge.py` calls FK through `get_position(q_current)` to estimate the current end-effector XYZ position from the project's kinematics model.
4. It adds the requested displacement:

   ```text
   target_xyz = current_xyz + [dx_m, dy_m, dz_m]
   ```

5. That resulting absolute XYZ target is passed to the same motion planner used for absolute commands.
6. The planned joint angles are previewed and require an exact `YES` before being sent to ROS.

This conversion is why the current measured joint positions are needed for relative movement. Relative movement accuracy depends on the FK model matching the real robot model and coordinate frames.

### 5.3 Home command and the meaning of zero

The configured home joint vector is:

```python
HOME_Q = np.zeros(4, dtype=float)
```

This represents the joint-space target `[0, 0, 0, 0]` radians. It does **not** represent Cartesian position XYZ `(0, 0, 0)`. The home command sends this joint target after the operator confirms with `YES`.

Zero joint angles have been selected as the project's intended home configuration, but the pose must be checked against the robot URDF, joint zero offsets, controller configuration, and Gazebo before physical use. A zero vector in the IK solver can also be used as an initial guess; that is a separate concept from sending the home joint target.

### 5.4 Stop command

The AI recognizes `stop`, but the integrated code does not connect that intent to an actual ROS stop or hold service. The software prints a warning and sends no joint target. This text command is **not** an emergency stop. Use the actual controller or hardware stop mechanism.

## 6. ROS interface and configuration assumptions

The integrated controller currently expects:

| Item | Configured value |
|---|---|
| ROS version/API | ROS 1 (`rospy`) |
| Joint-state topic | `/joint_states` |
| Joint names and order | `joint1`, `joint2`, `joint3`, `joint4` |
| Joint goal service | `/goal_joint_space_path` |
| Service type | `open_manipulator_msgs/SetJointPosition` |
| Planning group | `arm` |
| Joint angle unit | Radians |
| Default path time | 3.0 seconds |
| Home joint target | `[0, 0, 0, 0]` radians |

If your ROS setup uses different topic, service, joint, or planning-group names, inspect the actual messages and services before changing the constants and request fields near the top of `ai_ros_controller.py`.

Useful inspection commands:

```bash
rostopic echo -n 1 /joint_states
rostopic info /joint_states
rosservice list | grep goal_joint_space_path
rossrv show open_manipulator_msgs/SetJointPosition
```

## 7. Safety and limitations before physical hardware use

- Keep the manual movement confirmation enabled during testing.
- Start in Gazebo/simulation with a clear workspace and conservative target positions.
- Verify the FK link dimensions, base and end-effector frames, axis directions, joint ordering, zero offsets, and joint limits in `kinematics.py` against the actual robot URDF and controller. Relative movement depends on these values being consistent.
- Verify that `[0, 0, 0, 0]` is the intended safe home pose in the simulator before using the home command on physical hardware.
- The planner checks the requested endpoint against the project's FK model and configured joint limits. It does not do collision checking, obstacle avoidance, or full workspace safety analysis.
- The ROS service response indicates whether a path was planned/accepted. It is not a feedback signal proving that the robot has finished moving; this project does not currently wait for a motion-completion signal.
- The `stop` action is not an emergency stop. Use a real controller/hardware stop mechanism.
- Whisper can mishear words, distances, and directions. Review its English translation, then separately review the planned XYZ and joint-angle preview.
