This project implements a computer vision pipeline using YOLO11n and the ZED SDK to detect cones in real time and estimate their spatial distance from the camera.

## Repository Structure
- `config/`: Configuration files and settings.
- `scripts/`: Helper utilities and modular code.
- `external.py`: Detection using external webcam
- `webcam_depth.py`: Depth processing script for laptop webcams.
- `webcam_yolo_live.py`: detection using laptop webcam.
- `zed_yolo_live.py`: Main execution script combining the ZED camera feed with YOLO detection.
- `requirements.txt`: List of Python packages required for the project.

---

## Model Training and Configuration

The object detection model is based on **YOLO11n (Nano)**, paired with a ZED2i camera.

### Dataset and Split
* **Source:** Custom cone dataset annotated and exported via Roboflow.
* **Data Distribution:** 
  * Training set: 80% of the total images.
  * Validation set: 15% of images, evaluated periodically during training to check for overfitting.
  * Test set: 5% of images used for final evaluation.

### Batch Formation and Augmentation
* **Batch Size:** 8 images per batch
* **Augmentations:** During training, images underwent mosaic augmentation (combining four images into a single training grid to improve small-object detection), random scaling, HSV color adjustments, and horizontal flips. These steps helped the model generalize across different outdoor lighting conditions, shadows, and distances.

### Training Parameters
* **Base Model:** YOLO11n pretrained weights.
* **Epochs:** 50 total epochs.
* **Input Resolution:** 640x640 pixels.
* **Optimizer:** Default AdamW/SGD configuration with cosine learning rate decay.

### Results
Performance metrics evaluated on the validation set after 50 epochs[cite: 2]:
- **Precision:** 97.8%
- **Recall:** 99.1%
- **mAP@0.5:** 98.9%
- **mAP@0.5:0.95:** 84.0%

---

## Setup and Installation

1. Clone the repository:
   ```Bash
   git clone [https://github.com/preethi-droid/Cone-Detection.git](https://github.com/preethi-droid/Cone-Detection.git)
   cd Cone-Detection
   ```
   
2. Create and activate a Python virtual environment:
    ```Bash
    python3 -m venv venv
    source venv/bin/activate
    ```
3.  Install the required dependencies:
    ```Bash
    pip install -r requirements.txt
    ```

4.  Usage

    To run the main pipeline using the ZED camera:
    ```Bash
    python zed_yolo_live.py
    ```
