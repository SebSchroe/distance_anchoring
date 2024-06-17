import serial
import numpy as np
import time

port = "COM3" # depending on USB-Port
slider = serial.Serial(port, baudrate=9600, timeout=0, rtscts=False)

# main function to get slider value
def get_slider_value(serial_port=slider, in_metres=True):
    serial_port.flushInput()
    buffer_string = ''
    while True:
        while serial_port.inWaiting() == 0: # added waiting loop until new values are in buffer
            time.sleep(0.05)
        buffer_string += serial_port.read(serial_port.inWaiting()).decode("ascii")
        if '\n' in buffer_string:
            lines = buffer_string.split('\n')  # Guaranteed to have at least 2 entries
            last_received = lines[-2].rstrip()
            if last_received:
                last_received = int(last_received)
                if in_metres:
                    last_received = np.interp(last_received, xp=[0, 1023], fp=[0, 15]) - 1.5
                return last_received

# checking loop that returns values, when button in pressed
def check_slider_values():
    print('Press ctl + c to stop checking')
    try:
        while True:
            value = get_slider_value()
            print(f"Received value: {value}")
            time.sleep(1)
    except KeyboardInterrupt:
        print("Checking stopped.")

check_slider_values()
