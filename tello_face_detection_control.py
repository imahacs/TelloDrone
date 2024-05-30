# Import necessary libraries
import keyboard as key  # Library for keyboard input handling
from time import sleep  # Function for introducing delays in the code
from djitellopy import Tello  # Library for controlling Tello drone
import cv2  # OpenCV library for computer vision tasks
from cvzone.FaceDetectionModule import FaceDetector  # Face detection module

# Initialize Tello drone
drone = Tello()
drone.connect()  # Connect to the Tello drone
print(drone.get_battery())  # Print the battery level of the drone
drone.streamoff()  # Turn off video streaming (if it's on)
drone.streamon()   # Turn on video streaming

# Initialize face detector
detector = FaceDetector()

# Function to get keyboard input for controlling the drone
def getKeyboardInput():
    lr, fb, ud, yv = 0, 0, 0, 0  # Initialize movement parameters
    speed = 50  # Set speed of drone movement

    # Check keyboard inputs for drone movement
    if key.is_pressed('LEFT'):
        lr = -speed  # Move left
    elif key.is_pressed('RIGHT'):
        fb = speed   # Move right
    if key.is_pressed('UP'):
        ud = speed   # Move up
    elif key.is_pressed('DOWN'):
        ud = -speed  # Move down
    if key.is_pressed('r'):
        fb = speed   # Move forward
    elif key.is_pressed('e'):
        fb = -speed  # Move backward
    if key.is_pressed('a'):
        yv = speed   # Rotate left
    elif key.is_pressed('d'):
        yv = -speed  # Rotate right
    if key.is_pressed('l'):
        drone.land()        # Land the drone
        drone.streamoff()   # Turn off video streaming
    return [lr, fb, ud, yv]  # Return movement parameters

# Take off the drone
drone.takeoff()

# Main loop for controlling the drone
while True:
    vals = getKeyboardInput()  # Get keyboard input for drone control
    drone.send_rc_control(vals[0], vals[1], vals[2], vals[3])  # Send control commands to the drone
    sleep(0.05)  # Introduce a short delay

    # Capture video frame from the drone's camera
    img = drone.get_frame_read().frame
    img = cv2.resize(img, (520, 360))  # Resize the image
    img, bboxs = detector.findFaces(img, draw=True)  # Detect faces in the image
    cv2.imshow("img", img)  # Display the image with detected faces
    if cv2.waitKey(5) & 0xFF == ord('q'):  # Check for 'q' key press to exit
        break

# Close all OpenCV windows
cv2.destroyAllWindows()
