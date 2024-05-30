# Import the Tello class from the djitellopy library
from djitellopy import tello 
import cv2

# Create an instance of the Tello class
me = tello.Tello()

# Connect to the Tello drone
me.connect()

# Print the battery level of the drone
print(me.get_battery())

# Turn off the video stream (in case it's already on)
me.streamoff()

# Turn on the video stream
me.streamon()

# Command the drone to take off
me.takeoff()

# Rotate the drone clockwise by 360 degrees
me.rotate_clockwise(360)

# Move the drone up by 80 centimeters
me.move_up(80)

# Rotate the drone counter-clockwise by 90 degrees
me.rotate_counter_clockwise(90)

# Move the drone forward by 20 centimeters
me.move_forward(20)

# Move the drone back by 20 centimeters
me.move_back(20)

# Move the drone forward again by 20 centimeters
me.move_forward(20)

# Loop to continuously capture and display the drone's video feed
while True:
    # Get a frame from the drone's video stream
    img = me.get_frame_read().frame

    # Convert the frame from BGR to RGB color space
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize the frame to 640x480 pixels
    img = cv2.resize(img, (640, 480))

    # Display the frame in a window named "Team1"
    cv2.imshow("Team1", img)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(5) & 0xFF == ord('q'):
        me.streamoff()
        break

# Close all OpenCV windows
cv2.destroyAllWindows()
