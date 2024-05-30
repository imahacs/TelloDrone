# Import the necessary libraries
import cv2
from cvzone.FaceDetectionModule import FaceDetector

# Open the default camera (0 for front camera, 1 for back camera)
cap = cv2.VideoCapture(0)

# Create a FaceDetector object
detector = FaceDetector()

# Main loop for capturing and processing video frames
while True:
    # Read a frame from the camera
    _, img = cap.read()
    
    # Find faces in the frame and draw bounding boxes around them
    img, boxs = detector.findFaces(img, draw=True)

    # Display the annotated frame
    cv2.imshow("img", img)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Close all OpenCV windows
cv2.destroyAllWindows()
