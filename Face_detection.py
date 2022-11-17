import cv2

window_name = "Detected Objects in webcam"
video = cv2.VideoCapture(0)

while video.isOpened():
    ret, frame = video.read()

    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cascade_classifier = cv2.CascadeClassifier(
        "haarcascade_frontalface.xml")
    detected_objects = cascade_classifier.detectMultiScale(
        image, minSize=(40, 40))

    if len(detected_objects) != 0:
        for (x, y, height, width) in detected_objects:
            cv2.rectangle(
                frame, (x, y), ((x + height), (y + width)), (0, 255, 0), 5)
    cv2.imshow(window_name, frame)

    if cv2.waitKey(1) == 27:
        break

video.release()
cv2.destroyAllWindows()