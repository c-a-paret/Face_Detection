import cv2

video = cv2.VideoCapture(0)
face_cascade1 = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
face_cascade2 = cv2.CascadeClassifier("haarcascade_eye.xml")

frames_shown = 1

while True:
    check, frame = video.read()

    grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade1.detectMultiScale(grey_frame, scaleFactor=1.2, minNeighbors=5)

    for x,y,w,h in faces:
        sub_face = frame[y:y+h, x:x+w]
        sub_face = cv2.GaussianBlur(sub_face,(23, 23), 30)
        frame[y:y+sub_face.shape[0], x:x+sub_face.shape[1]] = sub_face
        frame = cv2.putText(frame,'Face Found!',(x+w+10,y+(h/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)

    cv2.imshow("Capturing", frame)
    key = cv2.waitKey(1)

    frames_shown += 1

    if key == ord('q'):
        break;

print("{} frames shown".format(frames_shown))
video.release()
cv2.destroyAllWindows()
