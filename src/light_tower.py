import time

from dobot_arm import DobotArm

# This module defines functions for controlling the light tower.
# https://www.amazon.com/dp/B0871ZQ3KG

class Tower:
    def __init__(self, arm):
        self.arm = arm

        # Pins
        self.DO_RED_LED = 9
        self.DO_YELLOW_LED = 10
        self.DO_GREEN_LED = 11
        self.DO_LEDS = [self.DO_RED_LED, self.DO_YELLOW_LED, self.DO_GREEN_LED]

        self.DO_BUZZER = 12

    # FUNCTIONS FOR turning lights on/off one at a time
    
    def light_color(self, color: str = None):
        setting = [0, 0, 0]
    
        match color:
            case "red":
                setting = [1, 0, 0]
    
            case "yellow":
                setting = [0, 1, 0]
    
            case "green":
                setting = [0, 0, 1]
    
        return self.arm.do(self.DO_LEDS, setting)

    # FUNCTIONS for turning DO_BUZZER/alarm on and off
    # (these functions do not affect the LEDs)
    
    def buzzer_status(self, status: int | str):
        if status in [0, "off"]:
            setting = 0
    
        if status in [1, "on"]:
            setting = 1
    
        return self.arm.do(self.DO_BUZZER, setting)
    
    def fail_state(self, buzzer_cycles = 0):

        # Blink the Red LED until operator resets program
        self.light_color("red")

        def cycle(color: str):
            time.sleep(0.25)
            self.light_color()
            time.sleep(0.25)
            self.light_color("red")
    
        for i in range(buzzer_cycles):
            cycle("red")
    
        # turn DO_BUZZER off after specified number of cycles
        self.buzzer_status("off")

        while True:
            cycle("red")

        return None