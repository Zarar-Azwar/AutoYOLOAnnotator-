# Video Object Detection Pipeline using Florence-2

This project provides a comprehensive pipeline for extracting frames from videos and performing object detection using Microsoft's Florence-2-large model. The pipeline converts the detections into YOLO format, making it easy to use the annotations for training custom object detection models.

## Features

- Video frame extraction with configurable intervals
- Object detection using Florence-2-large model
- Automatic conversion to YOLO annotation format
- Visualization tools for bounding box display
- Progress tracking with status bars
- Support for both JPG and PNG output formats

## Prerequisites

```bash
pip install torch transformers pillow opencv-python matplotlib tqdm
```

You'll also need to have access to the Microsoft Florence-2-large model:
```python
model_id = 'microsoft/Florence-2-large'
```

## Usage Examples

### 1. Processing a Video File

```python
# Upload and process a video file
from google.colab import files
uploaded = files.upload()
video_path = list(uploaded.keys())[0]

# Extract frames
Clips_To_Frame.clip_to_frame_main(f"/content/{video_path}")
```

Expected output:
```
Video Properties:
Total Frames: 300
FPS: 30
Duration: 10.00 seconds
Extracting every 30 frame(s)
[==========>] 100% Extracting frames
Extraction complete!
Saved 10 frames to: extracted_frames
```

### 2. Running Object Detection

The pipeline will automatically:
1. Load each extracted frame
2. Perform object detection
3. Convert annotations to YOLO format
4. Visualize results

Example output structure:
```
Dataset creation complete!

Class mapping:
person: 0
car: 1
dog: 2

Directory structure created:
├── yolo_dataset/
│   ├── images/
│   ├── labels/
│   ├── classes.txt
│   └── dataset.yaml
```

## Configuration Options

### Frame Extraction

```python
output_dir = "extracted_frames"  # Custom output directory
frame_interval = 30  # Extract one frame every second (assuming 30fps video)
output_format = 'jpg'  # or 'png'
```

### YOLO Conversion

The converter creates a complete YOLO dataset structure with:
- Images directory
- Labels directory
- classes.txt file
- dataset.yaml configuration

## Visualization

The `plot_bbox` function provides visualization of detection results:
- Red bounding boxes around detected objects
- Labels with class names
- Automatic figure sizing and display

## Output Format

### YOLO Annotation Format
Each annotation file contains one line per object:
```
<class_id> <x_center> <y_center> <width> <height>
```
Where all values are normalized between 0 and 1.

### Dataset YAML Structure
```yaml
path: /path/to/yolo_dataset
train: images
val: images
nc: 3  # number of classes
names: ['person', 'car', 'dog']  # class names
```

## Error Handling

The pipeline includes comprehensive error handling for:
- Missing video files
- Invalid output formats
- Video loading errors
- Directory creation issues

## Notes

- The frame extraction interval can be adjusted based on your needs
- All coordinates are automatically normalized for YOLO format
- The visualization tool supports multiple objects per frame
- Progress bars provide real-time processing status

## Contributing

Feel free to submit issues and enhancement requests!

