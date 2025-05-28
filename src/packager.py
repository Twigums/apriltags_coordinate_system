import time

from light_tower import Tower


# This module defines functions for controlling and
# communicating with the packaging/labelling system and button.

class Packager:
    def __init__(self, arm):
        self.arm = arm
        self.tower = Tower(arm)

        # Dobot outputs
        self.DO_PKG_STOP = 6
        self.DO_PACKAGE_CONTROLLER_N_RESET = 7
        self.DO_MAKE_BAG = 8
        
        # Dobot inputs
        self.DI_BUTTON = 6
        self.DI_BUSY = 7
        self.DI_PACKAGE_CONTROLLER_ERR = 8

    def check_for_errors(self):

        # This triggers the fails state (with the buzzer beeping for the 
        # first 100 cycles) which there currently is no way to recover from.
        # The operator will need to restart the program.
        if self.arm.di(self.DI_PACKAGE_CONTROLLER_ERR) == [1]:
            tower.fail_state(self.arm, 2)
    
        return None

    # Packaging machine controller functions

    def wait_for_packaging_machine(self):
        while self.arm.di(self.DI_BUSY) == [1]:
            time.sleep(0.001)
            # check_for_errors()

        return None

    def make_bags(self, count = 1, wait_until_done = False):
    
        # send number of pulses indicating how many bags to make
        for i in range(count):
            self.arm.do(self.DO_MAKE_BAG, 1)
            time.sleep(0.15)
            self.arm.do(self.DO_MAKE_BAG, 0)
            time.sleep(0.15)
    
        if wait_until_done:
            self.wait_for_packaging_machine()
    
        return None

    def setup_package_controller(self, labeler_off = False):
        if labeler_off == False:
            print("Turning labeler on.")
    
            # Turn the labeler on by setting DO_MAKE_BAG 
            # high before A-Star reset
            self.arm.do([self.DO_MAKE_BAG, self.DO_PACKAGE_CONTROLLER_N_RESET], [1, 0])
            time.sleep(0.15)
    
            self.arm.do(self.DO_PACKAGE_CONTROLLER_N_RESET, 1)
            time.sleep(2)
    
            # Wait for DI_BUSY to go high indicating A-Star is ready
            while self.arm.di(self.DI_BUSY) == [0]:
                self.check_for_errors()
     
            # Acknowledge the signal by setting DO_MAKE_BAG low
            self.arm.do(self.DO_MAKE_BAG, 0)
    
            # Wait for DI_BUSY to go low
            self.wait_for_packaging_machine()
            print("Labeler ready")
    
        else:
            print("Turning labeler off.")
    
            # Turn the labeler off setting DO_MAKE_BAG 
            # low before A-Star reset
            self.arm.do([self.DO_MAKE_BAG, self.DO_PACKAGE_CONTROLLER_N_RESET], [0, 0])
            time.sleep(0.15)
    
            self.arm.do(self.DO_PACKAGE_CONTROLLER_N_RESET, 1)
            time.sleep(2)
    
            # Wait for DI_BUSY to go high indicating A-Star is ready
            while self.arm.di(self.DI_BUSY) == [0]:
                self.check_for_errors()
    
            # Acknowledge the signal by setting DO_MAKE_BAG high
            self.arm.do(self.DO_MAKE_BAG, 1)
    
            # Wait for DI_BUSY to go low as acknowledgement before setting DO_MAKE_BAG low
            self.wait_for_packaging_machine()
    
            self.arm.do(self.DO_MAKE_BAG, 0)
            print("Packager ready")
    
        return None

    # Button functions
    def wait_for_button(self, buzzer_cycles = 0):
        if buzzer_cycles > 0:
            self.tower.buzzer_status("on")
    
        self.tower.light_color("yellow")
        start_time = time.time()
        current_time = 0
        cycles = 0
    
        while True:
            current_time = time.time() - start_time
    
            if current_time > 1:
                self.tower.light_color("yellow")
                start_time = time.time()
                cycles += 1
    
            elif current_time > 0.5:
                self.tower.light_color()
    
            if cycles > buzzer_cycles:
                self.tower.buzzer_status("off")

            if self.arm.di(self.DI_BUTTON) == [0]:
                break
    
        self.tower.light_color("green")
        self.tower.buzzer_status("off")

        return None

    # checks if the button is held down for given time, default 2 sec
    def button_hold(self, hold_time = 2):
        start_time = time.time()
        current_time = 0

        while self.arm.di(self.DI_BUTTON) == [0]:
            current_time = time.time() - start_time
    
            if current_time > hold_time:
                return True
    
        return False