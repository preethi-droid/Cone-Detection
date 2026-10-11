import pyzed.sl as sl
import cv2
import math
from ultralytics import YOLO

def main():
    
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.camera_fps = 60
    
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    
    init_params.coordinate_units = sl.UNIT.METER 

    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("Failed to open ZED Camera. Check USB 3.0 connection.")
        return

    
    print("Loading YOLO model...")
    
    model_path = "models/runs/detect/train-2/weights/best.engine" 
    model = YOLO(model_path, task="detect")

    image_zed = sl.Mat()
    point_cloud = sl.Mat()
    runtime_parameters = sl.RuntimeParameters()

    print("Live feed started. Press 'q' to quit.")

    while True:
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            
            zed.retrieve_image(image_zed, sl.VIEW.LEFT)
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)

            frame_bgra = image_zed.get_data()
            frame_bgr = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)

            
            results = model.predict(frame_bgr, verbose=False, conf=0.5)[0]
            annotated_frame = results.plot()

            
            if results.boxes:
                for box in results.boxes:
                    
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)

                    
                    err, point3D = point_cloud.get_value(center_x, center_y)
                    
                    if err == sl.ERROR_CODE.SUCCESS:
                        x, y, z, color = point3D
                        
                        
                        if math.isfinite(z):
                            
                            distance = math.sqrt(x*x + y*y + z*z)
                            
                            
                            text = f"Dist: {distance:.2f}m | X:{x:.2f} Y:{y:.2f} Z:{z:.2f}"
                            cv2.putText(annotated_frame, text, (x1, y1 - 10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            
            cv2.imshow("ZED 2i Cone Detection (3D Position)", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()
    zed.close()

if __name__ == "__main__":
    main()
