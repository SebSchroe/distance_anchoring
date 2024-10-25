import time
import pygame
import serial
import logging
import serial.tools.list_ports
import threading

RUN = True
MIN_LED = 90
MAX_LED = 450
LEDS_PER_METER = 30
MIN_LED_DISTANCE = 1  # in meter
CURR_COMMAND = None
CURR_LED = MIN_LED


def init_controllers(multiple_controllers=False):
    pygame.init()
    pygame.joystick.init()
    if multiple_controllers:
        joysticks = [pygame.joystick.Joystick(x) for x in range(pygame.joystick.get_count())]
        for idx, controller in enumerate(joysticks):
            print(f'{controller.get_name()} connected // Controller ID: {idx}')
    else:
        joysticks = pygame.joystick.Joystick(0)
        print(f'{joysticks.get_name()} connected // Controller ID: {joysticks.get_id()}')
    return joysticks


def init_arduino():
    port = 'COM5'
    try:
        arduino = serial.Serial(port, baudrate=115200, timeout=0, rtscts=False)
        return arduino
    except serial.SerialException as e:
        logging.error(f"Failed to connect to the serial port {port}: {e}")


def get_controller_inputs(controller):
    inputs = {'y_speed': round(controller.get_axis(1), ndigits=2),
              'button_a': controller.get_button(0),
              'd_pad': controller.get_hat(0)}
    return inputs


def get_current_led(inputs, curr_led):
    if inputs['y_speed'] > 0.05 and curr_led > MIN_LED:
        curr_led -= 7 * (inputs['y_speed'] ** 3)
    elif inputs['y_speed'] < -0.05 and curr_led < MAX_LED:
        # pos = -1 *
        curr_led -= 7 * (inputs['y_speed'] ** 3)
    elif inputs['d_pad'][1] == 1 and curr_led < MAX_LED:
        curr_led += 1
        time.sleep(0.15)
    elif inputs['d_pad'][1] == -1 and curr_led > MIN_LED:
        curr_led -= 1
        time.sleep(0.15)
    return curr_led


def get_command(inputs):
    if inputs['button_a']:
        command = 'green'
        return command


def build_msg(string, int_number, float_number):
    msg = f'<{string}, {int_number}, {float_number}>'
    return msg


def send_to_arduino(msg, arduino):
    arduino.write(msg.encode)


def led_control_func():
    global RUN
    global CURR_LED
    global CURR_COMMAND
    RUN = True
    # curr_led = MIN_LED
    arduino = init_arduino()
    controller = init_controllers()
    while RUN:
        time.sleep(0.02)
        pygame.event.pump()
        inputs = get_controller_inputs(controller)
        CURR_LED = get_current_led(inputs=inputs, curr_led=CURR_LED)
        CURR_COMMAND = get_command(inputs=inputs)
        msg = build_msg(string=CURR_COMMAND, int_number=CURR_LED, float_number=0.0)
        arduino.write(msg.encode())
        print(msg, end='\r')
    pygame.quit()


def start_led_control():
    t = threading.Thread(target=led_control_func)
    t.start()
    return t


def stop_led_control():
    global RUN
    RUN = False


def get_led(on_command='green'):
    while CURR_COMMAND != on_command:
        continue
    led = CURR_LED
    return led


def led_to_meter(curr_led):
    adjusted_led = curr_led - MIN_LED
    curr_led_distance = adjusted_led / LEDS_PER_METER + MIN_LED_DISTANCE
    return curr_led_distance
