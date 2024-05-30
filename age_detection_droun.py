import cv2
import numpy as np
from djitellopy import Tello 
import keyboard as key
from time import sleep 

# Model files
FACE_MODEL = "weights/res10_300x300_ssd_iter_140000_fp16.caffemodel"
FACE_PROTO = "weights/deploy.prototxt.txt"
AGE_PROTO = 'weights/age_net.caffemodel'
AGE_MODEL = 'weights/deploy_age.prototxt'

# Load models
face_net = cv2.dnn.readNet(FACE_MODEL, FACE_PROTO)
age_net = cv2.dnn.readNet(AGE_MODEL, AGE_PROTO)

# Define mean values and age list
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

# Function to detect faces in a frame
def get_faces(frame):
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
    face_net.setInput(blob)
    detections = face_net.forward()
    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            start_x, start_y, end_x, end_y = box.astype(int)
            faces.append((start_x, start_y, end_x, end_y))
    return faces

# Function to predict age of faces
def predict_age(frame):
    faces = get_faces(frame)
    for (start_x, start_y, end_x, end_y) in faces:
        face = frame[start_y:end_y, start_x:end_x]
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        age_net.setInput(blob)
        age_preds = age_net.forward()
        age = age_list[age_preds[0].argmax()]
        label = f"{age}"
        cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
        cv2.putText(frame, label, (start_x, start_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return frame

# Initialize drone
drone = Tello()
drone.connect()
print(drone.get_battery())
drone.streamoff()
drone.streamon()

# Function to get keyboard input for drone control
def getKeyboardInput():
    lr, fb, ud, yv = 0,0,0,0
    speed = 50

    if key.is_pressed('LEFT'): lr = -speed
    elif key.is_pressed('RIGHT'): fb = speed

    if key.is_pressed('UP'): ud = speed
    elif key.is_pressed('DOWN'): ud = -speed

    if key.is_pressed('w'): fb = speed
    elif key.is_pressed('s'): fb = -speed

    if key.is_pressed('a'): yv = speed
    elif key.is_pressed('d'): yv = -speed

    if key.is_pressed('l'):
        drone.land()
        drone.streamoff()
    return [lr, fb, ud, yv]

# Drone takeoff
drone.takeoff()

while True:
    vals = getKeyboardInput()
    drone.send_rc_control(vals[0], vals[1], vals[2], vals[3])
    sleep(0.05)
    img = drone.get_frame_read().frame
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 480))

    # Predict age of faces in the image
    img = predict_age(img)

    # Display the image with age predictions
    cv2.imshow('Age Detection ', img)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(10) & 0xFF == ord('q'):
        drone.streamoff()
        break

# Release the drone connection and close windows
drone.release()
cv2.destroyAllWindows()
