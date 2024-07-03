import network
import socket
from machine import Pin, PWM, I2C
import time
import ssd1306

# Configuration for your WiFi network
SSID = 'KDD Lab'
PASSWORD = 'kdd$ec123'

# Setup the WiFi connection
wlan = network.WLAN(network.STA_IF)
wlan.active(True)
wlan.connect(SSID, PASSWORD)

# Wait for connection
while not wlan.isconnected():
    pass

print('Connected to WiFi. IP:', wlan.ifconfig()[0])

# Setup the L298 motor driver pins
IN1 = Pin(25, Pin.OUT)
IN2 = Pin(26, Pin.OUT)
EN_A = PWM(Pin(27), freq=1000)

IN3 = Pin(33, Pin.OUT)
IN4 = Pin(32, Pin.OUT)
EN_B = PWM(Pin(14), freq=1000)

# Setup the servos
servo1 = PWM(Pin(21), freq=50)
servo2 = PWM(Pin(22), freq=50)

# Setup I2C for the OLED display
i2c = I2C(-1, scl=Pin(22), sda=Pin(21))
oled_width = 128
oled_height = 64
oled = ssd1306.SSD1306_I2C(oled_width, oled_height, i2c)
oled.text(wlan.ifconfig()[0], 10, 25)
oled.show()

# Function to set motor speed and direction
def set_motor(speed, direction):
    speed = int(speed * 1023 / 100)  # Convert 0-100% speed to duty cycle (0-1023)
    if direction == 'forward':
        EN_A.duty(speed)
        EN_B.duty(speed)
        IN1.value(1)
        IN2.value(0)
        IN3.value(1)
        IN4.value(0)
    elif direction == 'backward':
        EN_A.duty(speed)
        EN_B.duty(speed)
        IN1.value(0)
        IN2.value(1)
        IN3.value(0)
        IN4.value(1)
    elif direction == 'left':
        EN_A.duty(speed)
        EN_B.duty(speed)
        IN1.value(0)
        IN2.value(1)
        IN3.value(1)
        IN4.value(0)
    elif direction == 'right':
        EN_A.duty(speed)
        EN_B.duty(speed)
        IN1.value(1)
        IN2.value(0)
        IN3.value(0)
        IN4.value(1)
    elif direction == 'stop':
        EN_A.duty(0)
        EN_B.duty(0)
        IN1.value(0)
        IN2.value(0)
        IN3.value(0)
        IN4.value(0)

# Function to set servo angle
def set_servo(servo, angle):
    min_duty = 40  # Minimum duty cycle for 0 degrees
    max_duty = 115  # Maximum duty cycle for 180 degrees
    duty = int(min_duty + (angle / 180) * (max_duty - min_duty))
    servo.duty(duty)

# Function to display a bot face
def display_face(face_type):
    oled.fill(0)  # Clear the display
    if face_type == 'happy':
        oled.text('0 u 0', 50, 25)
    elif face_type == 'sad':
        oled.text('O w O', 50, 25)
    elif face_type == 'neutral':
        oled.text('O - O', 50, 25)
    oled.show()

# Setup the web server
addr = socket.getaddrinfo('0.0.0.0', 80)[0][-1]
s = socket.socket()
s.bind(addr)
s.listen(1)

print('Listening on', addr)

# HTML for the web page
html = """<!DOCTYPE html>
<html>
<head>
    <title>ESP32 Car Control</title>
</head>
<body>
    <h1>ESP32 Car Control</h1>
    <button onclick="sendCommand('forward')">Forward</button>
    <button onclick="sendCommand('backward')">Backward</button>
    <button onclick="sendCommand('left')">Left</button>
    <button onclick="sendCommand('right')">Right</button>
    <button onclick="sendCommand('stop')">Stop</button>
    <br>
    <label for="servo1">Servo 1 Angle:</label>
    <input type="range" id="servo1" name="servo1" min="0" max="180" onchange="sendServoCommand('servo1', this.value)">
    <br>
    <label for="servo2">Servo 2 Angle:</label>
    <input type="range" id="servo2" name="servo2" min="0" max="180" onchange="sendServoCommand('servo2', this.value)">
    <br>
    <button onclick="sendFaceCommand('happy')">Show Happy Face</button>
    <button onclick="sendFaceCommand('sad')">Show Sad Face</button>
    <button onclick="sendFaceCommand('neutral')">Show Neutral Face</button>
    <script>
        function sendCommand(cmd) {
            var xhttp = new XMLHttpRequest();
            xhttp.open("GET", "/" + cmd, true);
            xhttp.send();
        }
        function sendServoCommand(servo, angle) {
            var xhttp = new XMLHttpRequest();
            xhttp.open("GET", "/" + servo + "?angle=" + angle, true);
            xhttp.send();
        }
        function sendFaceCommand(face) {
            var xhttp = new XMLHttpRequest();
            xhttp.open("GET", "/face?type=" + face, true);
            xhttp.send();
        }
    </script>
</body>
</html>
"""

# Main loop to handle web requests
while True:
    try:
       
        # Handle API and web app requests based on the path
        cl, addr = s.accept()
        print('Client connected from', addr)
        request = cl.recv(1024).decode()
        print('Request:', request)
        if 'POST /command' in request:
            command = request.split('\r\n\r\n')[1]
            if command == 'forward':
                set_motor(100, 'forward')
            elif command == 'backward':
                set_motor(100, 'backward')
            elif command == 'left':
                set_motor(100, 'left')
            elif command == 'right':
                set_motor(100, 'right')
            elif command == 'stop':
                set_motor(0, 'stop')
            elif 'servo1:' in command:
                angle = int(command.split(':')[1])
                set_servo(servo1, angle)
            elif 'servo2:' in command:
                angle = int(command.split(':')[1])
                set_servo(servo2, angle)
        elif "GET" in request:
            if 'GET /forward' in request:
                set_motor(100, 'forward')
            elif 'GET /backward' in request:
                set_motor(100, 'backward')
            elif 'GET /left' in request:
                set_motor(100, 'left')
            elif 'GET /right' in request:
                set_motor(100, 'right')
            elif 'GET /stop' in request:
                set_motor(0, 'stop')
            elif 'GET /servo1' in request:
                angle = int(request.split('angle=')[1].split(' ')[0])
                set_servo(servo1, angle)
            elif 'GET /servo2' in request:
                angle = int(request.split('angle=')[1].split(' ')[0])
                set_servo(servo2, angle)
            elif 'GET /face?type=' in request:
                face_type = request.split('type=')[1].split(' ')[0]
                display_face(face_type)
        response = html
        cl.send('HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n')
        cl.send(response)
        cl.close()
    except Exception as e:
        print('Error:', e)
    finally:
        if 'cl' in locals():
            cl.close()

