import cv2
from ultralytics import YOLO

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("Error: Could not access laptop webcam.")
        return

    print("Loading YOLO model...")
    model_path = "models/runs/detect/train-2/weights/best.engine"
    model = YOLO(model_path)

    print("Webcam feed started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        results = model.predict(frame, verbose=False, conf=0.5)[0]

        annotated_frame = results.plot()

        cv2.imshow("Webcam Cone Detection Test", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
