# This is a library for controlling the MG-400 Dobot arm.
#
# It has a lot of features and code in it specific to the Motor Tester G3, but
# there is also a lot of stuff in it that is generally useful to any project.
# (Maybe those two types of code should be in different classes.)
#
# Note: The Dobot's "Remote Control" mode must be set to
# "TCP/IP Secondary development" in the settings.
#
# Arm params documentation (arm_params.json):
# - standard_z (float): This is the Z coordinate we want to use for most of our
#   standard points, so we know there is no height change when moving between
#   those points, which makes it easier to reason about collisions.
#
# - tester_max_x (float)
#   The maximum X coordinate of the tester zone (i.e. the area where the arm
#   could be commanded to when dealing with motors on the tester).
#   In some cases, we raise an exception when we see an X coordinate larger
#   than this, to help prevent the arm from hitting the tester.
#
# - tester_min_x (float):
#   The minimum X coordinate of the tester zone.  We use this to split
#   the space of possible coordinates up into multiple zones and assign
#   waypoints to enable safe navigation without the arm hitting anything.
#
# - input_waypoint (coords):
#   The point to move the arm to when entering or exiting the input tray area.
#
# - output_waypoint (coords):
#   The point to move the arm to when entering or exiting the output tray area.
#
# - learned: Map of ID to coords that were manually learned for each
#   motor position.
#   For motor ID 1, these coordinates define the standard grasp of the
#   motor on the arm.
#   For other motor IDs, these are the coordinates you would use to place a
#   motor in the standard grasp in the most-forward position possible.

import json
import os
import socket
import time
import inspect
from functools import wraps
import numpy as np

robot_modes = [0, 'init', 'open', 'status', 'disabled', 'enable', 'backdrive',
    'running', 'recording', 'error', 'pause', 'jog' ]

# run a few checks before calling f()
def check_movement_args(f):

    @wraps(f)
    def wrapper(*args, **kwargs):
        sig = inspect.signature(f)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()

        self = args[0]

        for name, value in bound.arguments.items():
            match name:

                # defaults to set top_[var] if argument not provided or above top_[var]
                case "speed":
                    if value is None or value > self.top_speed:
                        bound.arguments[name] = self.top_speed

                    if bound.arguments[name] > 100:
                        raise ValueError(f"Speed = {value} > 100.")

                case "acc":
                    if value is None or value > self.top_acc:
                        bound.arguments[name] = self.top_acc

                case "cp":
                    if value is None or value > self.top_cp:
                        bound.arguments[name] = self.top_cp

                # if the function uses coordinates, check if they are in a valid format
                case _ if name.startswith("coords"):
                    if type(value) == np.ndarray:
                        value = value.tolist()

                    match value:
                        case [x, y, z, r]:
                            is_right_type = all(isinstance(val, (int, float)) for val in [x, y, z, r])

                            if not is_right_type:
                                raise ValueError(f"Invalid coord: {item}.")

                        case _:
                            raise ValueError(f"Invalid coords length: {len(value)}.")

        return f(*bound.args, **bound.kwargs)

    return wrapper

class DobotArm:
    def __init__(self, log_level: int = 1):
        self.top_speed = 10
        self.top_acc = 100
        self.top_cp = 100

        # Set to true to simplify the way network communication is done, which
        # could be useful for debugging but slows everything down a lot.
        self.sync_mode = False

        # 5: Log all network communication
        # 4: Log all asyncronously sent commands
        # 3: Log all commands/response exchanges
        # 2: Log high-level movement info (none exist yet).
        # 1: Log warnings (none exist yet).
        self.log_level = log_level

        self.ip = "192.168.1.6"

        self.dashboard = None
        self.motion = None

    def _new_conn(self, socket):
        conn = { "socket": socket }
        if self.sync_mode:
            conn["responses"] = []
        else:
            conn["cmds_in_flight"] = []
            conn["rx_buffer"] = bytearray(0)
        return conn

    def connect(self):
        try:
            if not self.dashboard:
                dashboard_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                dashboard_socket.settimeout(10)
                dashboard_socket.connect((self.ip, 29999))

                self.dashboard = self._new_conn(dashboard_socket)

            if not self.motion:
                motion_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                motion_socket.settimeout(10)
                motion_socket.connect((self.ip, 30003))

                self.motion = self._new_conn(motion_socket)

        except OSError as e:
            self.disconnect()

            # No route to host
            if e.errno == 113:
                ke = RuntimeError("Could not connect to Dobot arm.  " \
                    "Make sure arm's power switch is on, LAN1 is connected " \
                    "to the ethernet cable, and the arm's LED is done flashing.")

                ke.known_error = True

                raise ke from e

            raise e

        except Exception as e:
            self.disconnect()

            raise e

    def disconnect(self):
        if self.dashboard:
            self.dashboard["socket"].close()
            self.dashboard = None

        if self.motion:
            self.motion["socket"].close()
            self.motion = None

    def is_connected(self):
        return self.motion != None

    def parse_response(self, cmd_bytes, response_bytes):
        if response_bytes == None:

            raise RuntimeError(f"Timeout for command {cmd_bytes}.")

        expected_ending = b"," + b"".join(cmd_bytes.split(b" "))

        if not response_bytes.endswith(expected_ending):

            raise RuntimeError("Command/response mismatch: "
                f"{cmd_bytes} -> {bytes(response_bytes)}")

        response_bytes = response_bytes[:len(response_bytes) - len(expected_ending)]

        if self.log_level >= 3:
            print("Dobot exchange:", cmd_bytes, "->", bytes(response_bytes))

        response_bytes = response_bytes.replace(b"{", b"")
        response_bytes = response_bytes.replace(b"}", b"")

        parts = response_bytes.split(b",")

        if parts[0] != b"0":
            if cmd_bytes == b"Pause()" and parts[0] == b"-1":
                # The Pause command always gives this error.
                pass

            elif cmd_bytes == b"StopScript()" and parts[0] == b"-1":
                # StopScript seems to give this error if the
                # robot is already in error mode.
                pass

            else:
                raise RuntimeError(f"Error code from Dobot: "
                    f"{cmd_bytes} -> {bytes(response_bytes)}")

        parts = parts[1:]

        return parts

    def cmd(self, conn, cmds):
        socket = conn["socket"]

        if self.sync_mode:
            if isinstance(cmds, list):
                for cmd in cmds:
                    self.cmd(conn, cmd)

                    return None

            cmd_bytes = cmds.encode() if isinstance(cmds, str) else cmds

            if self.log_level >= 4 and not self.sync_mode:
                print("Dobot TX:", cmd_bytes)

            socket.sendall(cmd_bytes)
            response_bytes = socket.recv(4096)

            if self.log_level >= 5:
                print("Dobot RX:", response_bytes)

            response_data = parse_response(cmd_bytes, response_bytes)
            conn["responses"].append(response_data)

        else:
            if isinstance(cmds, list):
                cmd_bytes_list = []
                for cmd in cmds:
                    if isinstance(cmd, str):
                        cmd = cmd.encode()

                    cmd_bytes_list.append(cmd)

            elif isinstance(cmds, str):
                cmd_bytes_list = [cmds.encode()]

            else:
                cmd_bytes_list = [cmds]

            cmd_bytes = b"".join(cmd_bytes_list)

            if self.log_level >= 4 and not self.sync_mode:
                print("Dobot TX:", cmd_bytes)

            socket.sendall(cmd_bytes)
            conn["cmds_in_flight"] += cmd_bytes_list

    # Reads a response from the Dobot or returns None if there was a timeout.
    def read_response(self, conn):
        assert(not self.sync_mode)

        sock = conn["socket"]
        rx_buffer = conn["rx_buffer"]
        tries = 0

        while tries < 8:
            if (index := rx_buffer.find(b';')) >= 0:
                response = rx_buffer[0:index]
                del rx_buffer[0:(index+1)]

                return response

            try:
                r = sock.recv(4096)

            except TimeoutError:

                return None

            if self.log_level >= 5:
                print("Dobot RX:", r)

            rx_buffer += r

        return None

    # Reads reponses for all the in-flight commands.  Raises an exception
    # if there was a timeout or bad response.
    def read_responses(self, conn):
        if self.sync_mode:
            r = conn["responses"]
            conn["responses"] = []

            return r

        else:
            responses = []
            try:
                for cmd in conn["cmds_in_flight"]:
                    response_bytes = self.read_response(conn)
                    response_data = self.parse_response(cmd, response_bytes)

                    responses.append(response_data)

            except Exception as e:

                # Something went wrong.  Reset our RX state so we don't
                # get the same error later.
                conn["cmds_in_flight"] = []
                conn["rx_buffer"] = bytearray(0)

                raise e

            finally:
                del conn["cmds_in_flight"][:]

            return responses

    def dashboard_cmd(self, cmd: str):
        return self.cmd(self.dashboard, cmd)

    def read_dashboard_responses(self):
        return self.read_responses(self.dashboard)

    def motion_cmd(self, cmd: str):
        return self.cmd(self.motion, cmd)

    def read_motion_responses(self):
        return self.read_responses(self.motion)

    def clear_error(self):
        return self.enable_defaults(top_speed = self.top_speed)

    # Enable the robot and also change some settings back to good defaults.
    # speed should be between 1 and 100.  We will use 100 in production.
    # Default is 10.
    def enable_defaults(self):
        self.dashboard_cmd([
            b"ClearError()",
            b"wait(10)",           # Necessary if there was an error
            b"EnableRobot(0.22)",  # Set payload to 220 g
            b"DO(1,0)",
            b"DO(2,0)",
            b"DO(3,0)",
            f"CP({self.top_cp})",            # Enable smoothing at corners
            f"SpeedJ({self.top_speed})",
            f"SpeedL({self.top_speed})",
            f"AccJ({self.top_acc})",
            f"AccL({self.top_acc})",
        ])

        return self.read_dashboard_responses()

    # Gets the "robot mode" as a string:
    # - 'running' means the robot is busy executing queued commands.
    def get_robot_mode(self):
        self.motion_cmd(b"RobotMode()")
        mode_num = int(self.read_motion_responses()[-1][0])

        try:

            return robot_modes[mode_num]

        except IndexError:

            return mode_num

    def wait_for_movement_completion(self, counter: int = 3):

        # These two lines are necessary in case we recently issued a movl command.
        # (The Dobot protocol is badly designed!)
        self.read_motion_responses()
        time.sleep(0.15)

        not_running_count = 0

        while not_running_count < counter:
            mode = self.get_robot_mode()

            if mode in ["running", "error", "pause"]:
                not_running_count = 0

            else:
                not_running_count += 1

            time.sleep(0.1)

        return mode

    # Gets the coordinates of the robot's curreent position (X,Y,Z,R).
    # Warning: If the robot is currently executing a queued movement, this is
    # probably not what you want.  Call wait_for_movement_compleition() first.
    def get_coords(self, user = 0, tool = 0):
        self.motion_cmd(f"GetPose({user},{tool})")
        parts = self.read_motion_responses()[-1]

        # The command actually returns 6 numbers but we don't know what the
        # last two are supposed to be.
        if len(parts) < 4:
            raise RuntimeError(f"Unexpected GetPose response: {parts}.")

        return [float(p) for p in parts[0:4]]

    # Cancels all queued commands.
    def stop(self):
        self.motion_cmd(b"StopScript()")

        return self.read_motion_responses()

    def pause(self):
        self.motion_cmd(b"Pause()")

        return self.read_motion_responses()

    def unpause(self):
        self.motion_cmd(b"Continue()")

        return self.read_motion_responses()

    # speed: 0 to 100.  Default is None, which means to move at whatever speed
    #   was set by a previous SpeedJ command (100 by default).
    @check_movement_args
    def mov(self, 
            move_type: str, 
            coords, 
            speed: int | float | None = None, 
            acc: int | None = None, 
            cp: int | None = None):

        args = f"{coords[0]:.2f},{coords[1]:.2f},{coords[2]:.2f},{coords[3]:.2f}"

        match move_type:
            case "MovJ":
                args += f",SpeedJ={speed}"
                args += f",AccJ={acc}"

            case _:
                args += f",SpeedL={speed}"
                args += f",AccL={acc}"

        args += f",CP={cp}"

        return self.motion_cmd(f"{move_type}({args})")

    # Queues a linear movement.
    def movl(self, coords, speed = None, acc = None, cp = None):
        self.mov('MovL', coords, speed, acc, cp)
        self.wait_for_movement_completion()

        return None

    # Queues a joint movement (non-linear motion).
    def movj(self, coords, speed = None, acc = None, cp = None):
        self.mov('MovJ', coords, speed, acc, cp)
        self.wait_for_movement_completion()

        return None

    # implementation of relmovl without using RelMovLUser() from dobot
    # this doesn't have to redefine speed, acc, and cp
    def relmovl(self, coords, speed = None, acc = None, cp = None):
        current_pose = self.get_coords()
        self.mov("MovL", [x[0] + x[1] for x in zip(coords, current_pose)], speed, acc, cp)
        self.wait_for_movement_completion()

        return None

    # implementation of relmovj without using RelMovJUser() from dobot
    # this doesn't have to redefine speed, acc, and cp
    def relmovj(self, coords, speed = None, acc = None, cp = None):
        current_pose = self.get_coords()
        self.mov("MovJ", [x[0] + x[1] for x in zip(coords, current_pose)], speed, acc, cp)
        self.wait_for_movement_completion()

        return None

    # Circle() from dobot
    # move in a circle from current position -> 1st coord -> 2nd coord 'count' times
    @check_movement_args
    def circle(self, count, coords1, coords2):

        # for some reason the dobot developers decided to use {}?
        coords1_str = "{" + ",".join([str(x) for x in coords1]) + "}"
        coords2_str = "{" + ",".join([str(x) for x in coords2]) + "}"

        self.motion_cmd(f"Circle({count},{coords1_str},{coords2_str})")
        self.wait_for_movement_completion()

        return None

    # Arc() from dobot
    # move in an arc from current position -> 1st coord -> 2nd coord
    @check_movement_args
    def arc(self, coords1, coords2, user = 0, tool = 0, speed = None, acc = None, cp = None):
        dobot_kwargs = ["User", "Tool", "SpeedL", "AccL", "CP"]
        kwargs = [user, tool, speed, acc, cp]

        coords1_str = ",".join([str(x) for x in coords1])
        coords2_str = ",".join([str(x) for x in coords2])

        args = ",".join([coords1_str, coords2_str])

        for dobot_kwarg, kwarg in zip(dobot_kwargs, kwargs):
            args += f",{dobot_kwarg}={kwarg}"

        self.motion_cmd(f"Arc({args})")
        self.wait_for_movement_completion()

        return None

    # DI() from dobot
    # pins can either be an int or a list with # of elements > 1
    def di(self, pins: int | list):
        match pins:
            case int():
                command = f"DI({pins})"

            case [x, *rest]:
                command = [f"DI({i})" for i in pins]

            case _:
                raise ValueError(f"{pins} not 'int' or 'list' of length > 1")

        self.dashboard_cmd(command)
        res = self.read_dashboard_responses()

        decoded_res = [res[i][0].decode() for i in range(len(res))]

        match decoded_res:
            case [x]:
                return int(x)

            case [x, *rest]:
                return [int(val) for val in decoded_res]

    # DO() from dobot
    # pins can either be an int or a list with # of elements > 1
    # if pins is a list, len(status) == len(pins)
    # if pins is an int, type(status) == int
    def do(self, pins: int | list, status: int | list):
        match pins:
            case int():
                if type(status) != int:
                    raise ValueError(f"{status} not 'int'.")

                command = f"DO({pins},{status})"

            case [x, *rest]:
                if len(pins) != len(status):
                    raise ValueError(f"{status} does not have length {len(pins)}.")

                arg = ",".join([str(x) for pair in zip(pins, status) for x in pair])
                command = f"DOGroup({arg})"

            case _:
                raise ValueError(f"{pins} not 'int' nor 'list' of length > 1.")

        self.motion_cmd(command)

        return None

    # SetUser() from dobot
    # defines a new user with origin at current point
    @check_movement_args
    def set_user(self, index, coords):
        coords_str = "{" + ",".join([str(x) for x in coords]) + "}"
        
        command = f"SetUser({index},{coords_str})"
        self.dashboard_cmd(command)

        return None

    # CalcUser() from dobot
    @check_movement_args
    def calc_user(self, index, matmul_method, coords):
        coords_str = "{" + ",".join([str(x) for x in coords]) + "}"
        
        command = f"CalcUser({index},{matmul_method},{coords_str})"
        self.dashboard_cmd(command)

        return None