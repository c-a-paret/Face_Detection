import cv2
import time

video = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


while True:
    check, frame = video.read()


    grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(grey_frame, scaleFactor=1.5, minNeighbors=5)
    # print(faces)

    for x,y,w,h in faces:
        # print("Adding rectangle")
        frame = cv2.rectangle(frame, (x,y), (x+w, y+h), (50,255,0), 2)

    cv2.imshow("Capturing", frame)
    key = cv2.waitKey(1)

    if key == ord('q'):
        break;


video.release()
cv2.destroyAllWindows()
