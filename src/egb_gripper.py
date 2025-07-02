import serial
import struct
import time


def calculate_crc(data):
    crc = 0xFFFF

    for b in data:
        crc ^= b

        for _ in range(8):
            lsb = crc & 0x0001
            crc >>= 1

            if lsb:
                crc ^= 0xA001

    return struct.pack("<H", crc)

# sample control for https://www.pololu.com/file/0J2112/RobustMotion%20RM-EGB%20gripper%20user%20manual.pdf
class EGB_Gripper():
    def __init__(self, port = "/dev/ttyUSB0", baudrate = 115200):
        self.packet_size = 8

        # refer to manual
        self.ser = serial.Serial(
            port = port,
            baudrate = baudrate, # baud rates for gripper -> [9600, 19200, 38400, 57600, 115200]
            bytesize = self.packet_size,
            parity = "N",
            stopbits = 1,
            timeout = 2,
        )

        self.status_dict = {
            "04": "moving",
            "05": "undetected",
            "07": "detected",
        }

    def send_command(self, command, expected_res = ""):
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()

        self.ser.write(command)

        time.sleep(0.1)

        bytes_waiting = self.ser.in_waiting
        response = self.ser.read(bytes_waiting)

        # print(f"RES_BYTES: {bytes_waiting}")
        # print(f"RES: {response.hex().upper() if response else 'EMPTY'}")

        if bytes_waiting > 0:
            if response == expected_res and expected_res != "":
                return True

            return response

        return False
    
    def read_status(self):
        hex_cmd = "01 03 10 08 00 02 41 09"
        byte_cmd = bytes.fromhex(hex_cmd)

        res = self.send_command(byte_cmd)
        status_hex = res[4:5].hex()

        return self.status_dict[status_hex]

    def init_gripper(self):
        hex_init_bit1 = "01 05 00 11 00 00 9D CF"
        hex_init_cmd = "01 05 00 11 FF 00 DC 3F"

        byte_init_bit1 = bytes.fromhex(hex_init_bit1)
        byte_init_cmd = bytes.fromhex(hex_init_cmd)
        
        self.send_command(byte_init_bit1, b"\x01\x05\x00\x11\x00\x00\x9d\xcf")

        time.sleep(0.1)

        self.send_command(byte_init_cmd, b"\x01\x05\x00\x11\xff\x00\xdc?")

        while self.read_status() == False:
            print("Waiting for initialization to complete...")

        print("Initialization complete. Ready for movement commands.")

    def get_position(self):
        hex_cmd = "01 03 10 10 00 02 C1 0E"
        bytes_cmd = bytes.fromhex(hex_cmd)

        res = self.send_command(bytes_cmd)
        pos_mm = int.from_bytes(res[3:5], "big") / 100

        return pos_mm

    def set_force(self, force):
        hex_force = format(force, "02X")
        hex_cmd = f"01 10 10 06 00 02 04 00 {hex_force} 00 00"
        crc = calculate_crc(bytes.fromhex(hex_cmd))
    
        hex_cmd += f" {crc[0]:02X} {crc[1]:02X}"
        bytes_cmd = bytes.fromhex(hex_cmd)

        return self.send_command(bytes_cmd, b"\x01\x10\x10\x06\x00\x02\xa5\t")

    def set_distance(self, distance):
        hex_distance = format(distance, "02X")
        hex_cmd = f"01 10 10 04 00 02 04 00 {hex_distance} 00 00"
        crc = calculate_crc(bytes.fromhex(hex_cmd))
    
        hex_cmd += f" {crc[0]:02X} {crc[1]:02X}"
        bytes_cmd = bytes.fromhex(hex_cmd)

        return self.send_command(bytes_cmd, b"\x01\x10\x10\x04\x00\x02\x04\xc9")