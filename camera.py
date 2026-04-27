from ultralytics import YOLO
import cv2

model = YOLO("bomba.pt")

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Camera started. Press Q to quit.")

while True:
    success, frame = camera.read()

    if not success:
        print("Error: Could not read frame from camera.")
        break

    results = model(frame, conf=0.5)

    annotated_frame = results[0].plot()

    cv2.imshow("YOLO Camera Test", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()