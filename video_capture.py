import cv2

video = cv2.VideoCapture(0)
face_cascade1 = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
face_cascade2 = cv2.CascadeClassifier("haarcascade_eye.xml")

frames_shown = 1

while True:
    check, frame = video.read()

    grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade1.detectMultiScale(grey_frame, scaleFactor=1.5, minNeighbors=5)
    eyes = face_cascade2.detectMultiScale(grey_frame, scaleFactor=1.5, minNeighbors=5)

    for x,y,w,h in eyes:
        frame = cv2.rectangle(frame, (x,y), (x+w, y+h), (50,50,255), 1)

    for x,y,w,h in faces:
        frame = cv2.rectangle(frame, (x,y), (x+w, y+h), (50,255,0), 1)

    cv2.imshow("Capturing", frame)
    key = cv2.waitKey(1)

    frames_shown += 1

    if key == ord('q'):
        break;


print("{} frames shown".format(frames_shown))
video.release()
cv2.destroyAllWindows()
