# Import the necessary libraries
import cv2
from cvzone.PoseModule import PoseDetector

# Create a PoseDetector object
detector = PoseDetector()

# Open the default camera (0 for front camera, 1 for back camera)
cap = cv2.VideoCapture(0)

# Main loop for capturing and processing video frames
while True:
    # Read a frame from the camera
    _, img = cap.read()
    
    # Find and draw the pose on the frame
    img = detector.findPose(img, draw=True)
    
    # Find the position of landmarks on the detected pose
    detector.findPosition(img, draw=True)

    # Display the annotated frame
    cv2.imshow("Shehanh", img)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
