import time

from light_tower import Tower


# This module defines functions for interfacing with a test fixture via
# two of the MG400 robot arm's digital inputs (to read a "ready" and a "result" signal from the tester) 
# and one of its digital outputs ("to send the tester an acknowledgment signal).

class Tester:
    def __init__(self, arm):
        self.arm = arm
        self.tower = Tower(arm)

        # Pins
        self.DI_READY = 4
        self.DI_RESULT = 5
        self.DO_TRIGGER = 4
        self.DO_N_RESET = 5

        self.TEST_READY_STATE = 0

    # Basic tester functions
    
    # Send a High pulse to the tester
    def pulse_tester(self):
        self.arm.do(self.DO_TRIGGER, 1)
        time.sleep(0.1)
        self.arm.do(self.DO_TRIGGER, 0)
        time.sleep(0.1)

    # Check tester's ready signal to see if it is in a specific state (high or low)
    def check_test_ready_state(self, state):
        if self.arm.di(self.DI_READY) == state:
            self.TEST_READY_STATE = state
    
            return True
    
        return False

    # Checks DI_RESULT pin to determine pass or fail
    def get_test_result(self):
        if self.arm.di(self.DI_RESULT) == [1]:
            print("Test result: PASS")
    
            return True
    
        else:
            print("Test result: FAIL")
    
            return False

    # Handshaking functions for checking status of tester
    
    # Wait for tester's ready signal to be in a specific state (high or low)
    def wait_for_test_ready_state(self, state, time_out = 1.5):
        start_time = time.time()
        current_time = 0
    
        while self.check_test_ready_state(state) == False:
    
            # Do nothing when until DI_READY pin goes high or timeout is reached
            current_time = time.time() - start_time
            time.sleep(0.001)
    
            if current_time > time_out:
                # print("Timed Out!") # for debugging
    
                return False
    
        print("Current test ready state: ", self.TEST_READY_STATE) # for debugging
    
        return True

    # Wait for tester's ready signal to change from previous known state
    def wait_for_test_ready_change(self, time_out = 1.5):
        return not self.wait_for_test_ready_state(self.TEST_READY_STATE, time_out)

    # Tester resetting and setup functions
    
    # Sends signal to reset the tester and waits for the tester to signal that it is ready.
    # Will try reseting once by default, but will repeatedly reset if reset_interval is defined.
    def reset_tester(self, reset_interval = None):
        def reset():
            self.arm.do(self.DO_N_RESET, 1)
            time.sleep(0.15)
            self.arm.do(self.DO_N_RESET, 0)
            time.sleep(0.15)

        print("Resetting tester.")
        reset()
    
        state = 1
    
        # Wait for tester to signal it is ready
        while self.wait_for_test_ready_state(state, reset_interval) == False:
    
            # Indicate to operator that there might be a tester communication error
            self.tower.light_color("red")
            print("Waiting for tester to reset and indicate it is ready.")
    
            # Try resetting again if reset_interval is defined.
            if reset_interval:
                reset()
    
        # indicates to operators system is working normally
        self.tower.light_color("green")

        return None

    # Resets tester and selects mode
    def setup_tester(self, mode, reset_interval):
        self.reset_tester(reset_interval)
        print("Tester running setup.")
        
        if mode > 0:
    
            # Tells tester to start mode selection process
            self.pulse_tester()
            while self.wait_for_test_ready_change(time_out = 3) == False:
                
                # Indicate to operator that there might be a tester communication error
                self.tower.light_color("red")
                print("Waiting for tester acknowledge signal before starting mode selection.")
                self.pulse_tester()
    
            self.tower.light_color("green")
            print("Tester ready to select mode.")
            self.select_mode(mode)

            while self.wait_for_test_ready_change() == False:

                # Indicate to operator that there might be a tester communication error
                self.tower.light_color("red")
                print("Waiting for tester indicate mode selection is done.")

    
            self.tower.light_color("green")

        print("Tester ready.")

        return None

    # Select mode
    def select_mode(self, mode_num):
        for i in range(mode_num):
            self.pulse_tester()

        return None