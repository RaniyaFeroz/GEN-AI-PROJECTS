# Intelligent Vision Agent

**An AI-powered image analysis application for object detection and visual insights using YOLO and Streamlit.**

## Overview

Intelligent Vision Agent is a computer vision application that uses a pretrained YOLO (You Only Look Once) model to detect and analyze objects in images.

The application provides an interactive interface for uploading images, visualizing detected objects, reviewing prediction confidence scores, and exploring image-level detection statistics.

Built with Python, Streamlit, and Ultralytics YOLO, the project brings object detection capabilities into an accessible web-based application.

## Key Features

- **Image Upload:** Upload images for automated object detection.
- **YOLO-Based Detection:** Identify multiple objects using a pretrained YOLO model.
- **Detection Visualization:** View detected objects with bounding boxes.
- **Confidence Analysis:** Display confidence scores for individual detections.
- **Object Statistics:** Summarize detected object counts and categories.
- **Visual Analytics:** Explore detection results through interactive charts and summaries.
- **Performance Analysis:** Display image analysis time.
- **Multi-Image Benchmarking:** Process multiple images and compare detection results.
- **Detection History:** Review previously processed image results.
- **Downloadable Results:** Download detection outputs for further reference.
- **Service Status:** View the application's model and processing status.

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Computer Vision | OpenCV |
| Object Detection | Ultralytics YOLO |
| Deep Learning | PyTorch, Torchvision |
| Web Application | Streamlit |
| Data Processing | NumPy, Pandas |
| Image Processing | Pillow |
| Visualization | Matplotlib, Plotly |

## Project Architecture

The application follows a modular image-processing workflow:

```text
           Input Image
                |
                v
        Image Preprocessing
                |
                v
       Pretrained YOLO Model
                |
                v
        Object Detection
                |
        +-------+--------+
        |                |
        v                v
  Bounding Boxes    Detection Data
        |                |
        +-------+--------+
                |
                v
       Result Processing
                |
                v
       Analytics & Summary
                |
                v
       Streamlit Dashboard
                |
                v
       Downloadable Results
```

## Project Structure

```text
intelligent_vision_agent/
│
├── app.py
│
├── requirements.txt
│
├── README.md
│
└── .gitignore
```

| File | Description |
|---|---|
| `app.py` | Main Streamlit application containing the image detection and dashboard functionality. |
| `requirements.txt` | Lists the Python dependencies required to run the application. |
| `README.md` | Project documentation and setup instructions. |
| `.gitignore` | Specifies files and directories excluded from version control. |

## Installation and Setup

Follow these steps to run the application locally.

### 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

### 2. Navigate to the Project Directory

```bash
cd intelligent_vision_agent
```

### 3. Create a Virtual Environment

```bash
python -m venv .venv
```

### 4. Activate the Virtual Environment

**Windows:**

```bash
.venv\Scripts\activate
```

**Linux / macOS:**

```bash
source .venv/bin/activate
```

### 5. Install Dependencies

```bash
python -m pip install --upgrade pip
```

```bash
pip install -r requirements.txt
```

### 6. Run the Application

```bash
streamlit run app.py
```

The application will open in your default web browser.

**Note:** The pretrained YOLO model may be downloaded automatically during the first execution. An active internet connection may be required for the initial model download.

## Application Workflow

1. Launch the Streamlit application.
2. Upload an image through the application interface.
3. The image is processed using the pretrained YOLO model.
4. The model identifies objects and generates bounding-box predictions.
5. Review the detected objects and their confidence scores.
6. Explore object counts, visual summaries, and processing information.
7. Use the multi-image benchmarking and history features where required.
8. Download the available detection results.

## Model and Detection

The project uses a pretrained YOLO model provided by the Ultralytics framework.

YOLO performs object detection by identifying objects within an image and predicting their corresponding bounding boxes, class labels, and confidence scores.

The application uses these predictions to generate visual detection results and analytical summaries.

## Current Scope and Limitations

The current implementation focuses on image-based object detection.

The following capabilities are outside the scope of this version:

- Real-time video object detection.
- Webcam-based object detection.
- Custom model training and fine-tuning.
- Ground-truth-based model evaluation.

## Future Enhancements

Potential improvements include:

- Real-time object detection using webcam input.
- Video-based object detection.
- Custom dataset training and model fine-tuning.
- Advanced model performance evaluation.
- Detailed analytical report generation.
- Cloud deployment and API integration.

## Use Cases

The application demonstrates practical applications of computer vision, including:

- Automated image analysis.
- Object identification and localization.
- Visual data inspection.
- Detection result analysis.
- Multi-image object detection experiments.

## Author

**Raniya N**  
Data Science | Artificial Intelligence | Machine Learning | Computer Vision

## Acknowledgements

- [Ultralytics](https://www.ultralytics.com/) — YOLO object detection framework.
- [Streamlit](https://streamlit.io/) — Interactive web application framework.
- [PyTorch](https://pytorch.org/) — Deep learning framework.
- [OpenCV](https://opencv.org/) — Computer vision and image processing library.

---

*This project was developed as a practical implementation of deep learning and computer vision concepts using a pretrained object detection model.*
