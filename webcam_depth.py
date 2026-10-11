import cv2
from ultralytics import YOLO

REAL_CONE_WIDTH_M = 0.228  

FOCAL_LENGTH_PX = 1100.0

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

    
    c_x = 1280 / 2
    c_y = 720 / 2

    print("Webcam feed started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        results = model.predict(frame, verbose=False, conf=0.5)[0]
        annotated_frame = results.plot()

        if results.boxes:
            for box in results.boxes:
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                
                box_width_px = x2 - x1
                
                if box_width_px > 0:
                    
                    z_dist = (REAL_CONE_WIDTH_M * FOCAL_LENGTH_PX) / box_width_px
                    
                    
                    u = (x1 + x2) / 2
                    v = (y1 + y2) / 2
                    
                    
                    x_pos = ((u - c_x) * z_dist) / FOCAL_LENGTH_PX
                    y_pos = ((v - c_y) * z_dist) / FOCAL_LENGTH_PX
                    
                    
                    text = f"Z: {z_dist:.2f}m | X: {x_pos:.2f}m | Y: {y_pos:.2f}m"
                    cv2.putText(annotated_frame, text, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow("Monocular 3D Cone Estimation", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
