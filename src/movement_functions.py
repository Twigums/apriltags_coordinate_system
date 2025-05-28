import math

from dobot_arm import DobotArm
from tester import Tester
from packager import Packager
from gripper import Gripper


# This module has all the modules that actually commands the robot arm to move from position to position for performing different actions.

save_point = [300, 0, 80, 0]

# Locations of input UCS (user coordinate system)
input1 = np.array([46.661264,-224.029679,-76.966362,-47.104790])
input2 = np.array([196.710034, -394.037147, -79.410782, -47.603844])

# Locations of tester UCS (user coordinate system)
tester1 = np.array([207.753087, 184.911893, -82.447044, 43.807716])
tester2 = np.array([322.488184, 286.343721, -86.83371, 43.357578])

# Locations for reject UCS (user coordinate system)
reject1 = np.array([-208.678251, 232.01004, -78.981209, 133.631042])
reject2 = np.array([-259.587041, 289.226145, -82.615219, 133.452271])

#Packaging location (found by using calibration grippers touching packaging machine)
packaging_point = np.array([-362.958844, -182.974811, -157.034393, -46.725723])

# Converts coordinates in a user coordinate system to robot coordinates
def ucs_to_robot(coord_system, q):

    # picks which ucs to use
    match coord_system:
        case "input":
            p1 = input1
            p2 = input2

        case "tester":
            p1 = tester1
            p2 = tester2

        case "reject":
            p1 = reject1
            p2 = reject2

    # angle of ucs
    theta = math.atan((p1[1] - p2[1]) / (p1[0] - p2[0]))

    # compensaing for quadrants
    x_axis_d = theta if p1[0] >= 0 else theta + math.pi
    
    transformation_matrix = np.array([[np.cos(x_axis_d), -np.sin(x_axis_d)], [np.sin(x_axis_d), np.cos(x_axis_d)]]) 

    # rotation transformation
    q[:2] = np.dot(transformation_matrix,[q[0], q[1]])

    # movement transformation
    q = q + p1
    
    return q

# Converts robot coordinates to coordinates in a ucs
def robot_to_ucs(coord_system, q):

    # picks which ucs to use
    match coord_system:
        case "input":
            p1 = input1
            p2 = input2
            
        case "tester":
            p1 = tester1
            p2 = tester2

        case "reject":
            p1 = reject1
            p2 = reject2

    # movement transformation
    q = q - p1 

    # angle of ucs
    theta = math.atan((p1[1] - p2[1]) / (p1[0] - p2[0]))

    # negative angle, compensating for quadrants
    x_axis_d = -theta if p1[0] >= 0 else -theta - math.pi
    
    transformation_matrix = np.array([[np.cos(x_axis_d), -np.sin(x_axis_d)], [np.sin(x_axis_d), np.cos(x_axis_d)]]) 

    # rotation transformation
    q[:2] = np.dot(transformation_matrix,[q[0], q[1]])

    return q

class Move:
    def __init__(self, arm, gripper_type):
        self.arm = arm
        self.tester = Tester(arm)
        self.packager = Packager(arm)
        self.gripper = Gripper(arm, gripper_type)
    
    def pick_board(self, p, pick_height, last_try = False):
        self.arm.open_gripper("limit", 50)
    
        # Move over the input bin position
        self.arm.movj(p, speed = 100, acc = 100, cp = 100)
    
        # Try to pick up the board
        self.arm.relmovj([0, 0, -pick_height, 0], speed = 10, acc = 20, cp = 0)
        self.gripper.close_gripper("limit", 50)
        time.sleep(0.5)
        self.arm.relmovj([0, 0, pick_height, 0], speed = 10, acc = 20, cp = 0)
    
        #  If we did pick up a board, lift and return true
        if self.gripper.is_gripper_holding_object(): 
            return True
    
        # If we failed to grab the board, try again going down 1mm more
        elif last_try == False:
            return self.pick_board(p, pick_height + 1, True)
    
        # Return false if we did not pick up a board
        return False

    def replace_board(self, p, pick_height):
    
        # Move over the input bin position
        self.arm.movj(p, speed = 100, acc = 100, cp=100)
      
        # Lower and release the board
        self.arm.relmovj([0, 0, -pick_height, 0], speed = 10, acc = 20, cp = 0)
        self.gripper.open_gripper("limit", 50)
        time.sleep(0.5)
        self.arm.relmovj([0, 0, pick_height, 0], speed = 10, acc = 20, cp = 0)
    
        return None

    def test_board(self,
                   p,
                   place_height,
                   time_out,
                   pick_position,
                   max_attempts = 1,
                   repick_height = 0,
                   r = 1,
                   original_max_attempts = None):

        # Save original max attempts value for situations where we try repicking the board
        if original_max_attempts == None:
            original_max_attempts = max_attempts

        test_pass = False
    
        # Move to the tester
        self.arm.movj(p, speed = 100, acc = 100, cp = 100)
     
        while not self.tester.wait_for_test_ready_state([0]):
            print("Waiting for READY signal from tester to be low.")
    
        # Tighten grip on board
        self.gripper.close_gripper("limit", 75)
    
        # Lower onto tester
        self.arm.relmovj([0, 0, -place_height, 0], speed = 5, acc = 20, cp = 0)
    
        # Make a little circle to make sure the pogo pins set into the board correctly 
        if r != 0:

            # The robot will move the board to 6:00 then circle clockwise when max_attempts is odd
            # This will make the robot move the board to 12:00 first then circle counter-clockwise when max_attempts is even
            rx = -r if max_attempts % 2 == 1 else r

            # First, shift out the radius amount
            self.arm.relmovj([rx, 0, 0, 0], speed = 20, acc = 20, cp = 0)
    
            # Make the circle
            self.arm.circle(1, p + np.array([0, r, -place_height, 0]), p + np.array([-rx, 0, -place_height, 0]))
    
            # go back to the center
            self.arm.relmovj([-rx, 0, 0, 0], speed = 20, acc = 20, cp = 0)
            time.sleep(0.1)
    
        # Starts test
        self.tester.pulse_tester()
    
        if self.tester.wait_for_test_ready_state([1], time_out):
            test_pass = self.tester.get_test_result()
    
        else:
            print("Tester timeout!")
    
        # Lift from tester
        self.arm.relmovj([0, 0, place_height, 0], speed = 5, acc = 20, cp = 0)
    
        if test_pass:
            return True  
    
        elif max_attempts > 1:
    
            # acknowledge fail result
            self.tester.pulse_tester()
    
            return self.test_board(p, place_height, time_out, pick_position, max_attempts - 1, repick_height, r, original_max_attempts)
    
        elif max_attempts <= 1 and repick_height > 0:
            self.replace_board(pick_position, repick_height)
    
            # acknowledge fail result
            self.tester.pulse_tester()
    
            if self.pick_board(pick_position, repick_height):
                return self.test_board(p, place_height, time_out, pick_position, original_max_attempts, 0, r)
    
            return False

    def package_board(self, p, place_height, jolt, joltTurn, wait_for_operator = False):
    
        # Wait until the packaging machine is ready
        self.packager.wait_for_packaging_machine()
    
        # Move to the packaging machine
        self.arm.movj(packaging_point + p, speed = 100, acc = 50, cp = 100)
    
        # Move straight down towards machine 
        self.arm.relmovj([0, 0, -place_height, 0], speed = 10, acc = 20, cp = 0)
    
        self.gripper.open_gripper("limit", 100)
    
        if jolt == 1:
            self.arm.relmovj([0, 0, 10, 0], speed = 100, acc = 100, cp = 0)
    
            if joltTurn == 1:
                self.arm.relmovj([0, 0, -10, 90], speed = 100, acc = 100, cp = 0)
    
            else:
                self.arm.relmovj([0, 0, -10, 0], speed = 100, acc = 100, cp = 0)
    
            self.arm.relmovj([0, 0, -15, 0], speed = 100, acc = 100, cp = 0)
    
        # Lift straight up away from machine
        self.arm.relmovj([0, 0, place_height, 0], speed = 10, acc = 20, cp = 0)
    
        # If we need to wait for the operator, move the robot arm to the safe positon
        if wait_for_operator:
            self.move_to_safety()
            self.packager.wait_for_button(1)
    
        # send command to make a bag, but do not wait for it
        self.packager.make_bags(1, False)

        return None

    def reject_board(self, p, place_height):
    
        # Move to the reject bin
        self.arm.movj(p, speed = 100, acc = 100, cp = 100)
    
        # Move straight down towards reject bin
        self.arm.relmovj([0, 0, -place_height, 0], speed = 50, acc = 20, cp = 0)
    
        # drop board
        self.gripper.open_gripper("limit", 100)
    
        # Lift straight up away from reject bin
        self.arm.relmovj([0, 0, place_height, 0], speed = 100, acc = 20, cp = 0)
        self.arm.movj(tester1 + [0, 0, 200, 0], speed = 100, acc = 20, cp = 100)
    
    def move_to_safety(self):
        self.arm.relmovj([0, 0, 25, 0], speed = 20, acc = 20, cp = 0)
        self.arm.movj(save_point, speed = 50, acc = 50)
        self.arm.wait_for_movement_completion()
        self.gripper.open_gripper("limit", 100)
        time.sleep(1)