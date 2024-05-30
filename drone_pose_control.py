# Import the necessary libraries
from djitellopy import tello
import cv2
import cvzone
from cvzone.PoseModule import PoseDetector

# Initialize PoseDetector
detector = PoseDetector()

# Define dimensions
hi, wi = 480, 640

# Initialize PID controllers for X, Y, and Z axes
xPID = cvzone.PID([0.22, 0, 0.1], wi // 2)
yPID = cvzone.PID([0.27, 0, 0.1], hi // 2, axis=1)
zPID = cvzone.PID([0.00016, 0, 0.000011], 150000, limit=[-20, 15])

# Initialize live plots for PID controllers
myPlotX = cvzone.LivePlot(yLimit=[-100, 100], char='X')
myPlotY = cvzone.LivePlot(yLimit=[-100, 100], char='Y')
myPlotZ = cvzone.LivePlot(yLimit=[-100, 100], char='Z')

# Initialize Tello drone
me = tello.Tello()
me.connect()
print(me.get_battery())
me.streamoff()
me.streamon()
me.takeoff()
me.move_up(80)

# Main loop for controlling the drone based on pose detection
while True:
    # Get a frame from the drone's video stream
    img = me.get_frame_read().frame
    img = cv2.resize(img, (640, 480))

    # Detect poses and keypoints in the frame
    img = detector.findPose(img, draw=True)
    lmlist, bboxInfo = detector.findPosition(img, draw=True)

    # Initialize values for controlling the drone
    xVal = 0
    yVal = 0
    zVal = 0

    # If a bounding box around a person is detected
    if bboxInfo:
        cx, cy = bboxInfo['center']
        x, y, w, h = bboxInfo['bbox']
        area = w * h

        # Update PID controllers and obtain control values
        xVal = int(xPID.update(cx))
        yVal = int(yPID.update(cy))
        zVal = int(zPID.update(area))

        # Update live plots with control values
        imgPlotX = myPlotX.update(xVal)
        imgPlotY = myPlotY.update(yVal)
        imgPlotZ = myPlotZ.update(zVal)

        # Draw PID controllers on the frame
        img = xPID.draw(img, [cx, cy])
        img = yPID.draw(img, [cx, cy])

        # Stack images with live plots for display
        imgStacked = cvzone.stackImages([img, imgPlotX, imgPlotY, imgPlotZ], 2, 0.75)

        # Display the area of the bounding box
        cv2.putText(imgStacked, str(area), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    else:
        # Stack only the original frame if no bounding box is detected
        imgStacked = cvzone.stackImages([img], 1, 0.75)

    # Send control commands to the drone
    me.send_rc_control(0, -zVal, -yVal, xVal)

    # Display the stacked image with live plots and control information
    cv2.imshow("Shehanah", imgStacked)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(5) & 0xFF == ord('q'):
        me.land()
        break

# Close all OpenCV windows
cv2.destroyAllWindows()
