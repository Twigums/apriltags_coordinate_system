from dobot_arm import DobotArm


class Gripper:
    def __init__(self, arm, gripper_type: str):
        self.arm = arm

        self.gripper_type = gripper_type # 'pnp' or 'npn'
        if gripper_type not in ["npn", "pnp"]:
            raise ValueError(f"{gripper_type} not 'pnp' or 'npn'")

        # the di/do pins the gripper uses
        self.gripper_inputs = [1, 2, 3]
        self.gripper_outputs = [1, 2, 3]

        # initiate gripper as open on init
        self.is_gripper_closed = False

        
        self.valid_gripper_params = {
            "distance": [2, 4, "limit"], # in mm; limit is either 0 or max depending on open or close
            "force": [25, 50, 75, 100] # in %
        }

    # checks if the provided arguments are in correct format for the gripper
    # this can be changed to a wrapper eventually
    def check_valid_gripper_args(self, distance: int | str, force: int) -> None:
        if distance not in self.valid_gripper_params["distance"] or force not in self.valid_gripper_params["force"]:
            e = ""

            if distance not in self.valid_gripper_params["distance"]:
                e += f"{distance} is not a valid gripper distance. "

            if force not in self.valid_gripper_params["force"]:
                e += f"{force} is not a valid gripper force. "

            raise ValueError(e)

        return None

    # checks if the gripper is currently moving
    def is_gripper_moving(self) -> bool:
        mode_list = self.arm.di(self.gripper_inputs)

        match mode_list:
            case [1, 1, 1]:
                return True

            case _:
                return False

    # checks if the gripper is currently holding an object
    def is_gripper_holding_object(self) -> bool:
        mode_list = self.arm.di(self.gripper_inputs)

        match mode_list:
            case [0, 0, 1]:
                is_gripper_force_feedback = True

            case [1, 1, 0]:
                is_gripper_force_feedback = True

            case _:
                is_gripper_force_feedback = False

        return self.is_gripper_closed and is_gripper_force_feedback

    # opens gripper using provided parameters
    def open_gripper(self, distance: int | str, force: int) -> None:
        self.check_valid_gripper_args(distance, force)
        self.is_gripper_closed = False

        match (distance, force):
            case (2, 100):
                settings = [1, 0, 1]

            case (4, 100):
                settings = [1, 1, 0]

            case ("limit", 100):
                settings = [1, 1, 1]

            case ("limit", 50):
                settings = [1, 0, 0]

            case _:
                raise ValueError("Not a valid (distance, force) combination.")

        # my setup was using npn, and pnp is flipped
        if self.gripper_type == "pnp":
            settings = [1 - x for x in settings]

        self.arm.do(self.gripper_outputs, settings)

        return None

    # opens gripper using provided parameters
    def close_gripper(self, distance: str, force: int) -> None:
        self.check_valid_gripper_args(distance, force)
        self.is_gripper_closed = True

        match (distance, force):
            case ("limit", 100):
                settings = [0, 1, 1]

            case ("limit", 75):
                settings = [0, 0, 1]

            case ("limit", 50):
                settings = [0, 1, 0]

            case ("limit", 25):
                settings = [0, 0, 0]

            case _:
                raise ValueError("Not a valid (distance, force) combination.")

        # my setup was using npn, and pnp is flipped
        if self.gripper_type == "pnp":
            settings = [1 - x for x in settings]

        self.arm.do(self.gripper_outputs, settings)

        return None