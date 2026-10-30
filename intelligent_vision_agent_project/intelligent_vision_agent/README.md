# Intelligent Vision Agent

A Streamlit computer-vision project for image object detection, visual summaries, analysis history, and multi-image inference-time benchmarking.

## Features
- Upload JPG, JPEG, PNG, or WEBP images
- Detect common objects with a pretrained YOLO model
- View annotated images with bounding boxes
- Review object counts, class counts, confidence scores, and inference time
- Download detection results as CSV
- Keep a session-level detection history
- Run a runtime benchmark across multiple images
- Check demo service status

## Requirements
- Python 3.10 or 3.11 recommended
- Internet access for the first pretrained model download
- 4 GB RAM minimum recommended

## Setup on Windows
Open Command Prompt or the VS Code terminal in this project folder:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Your browser should open the app. If it does not, open the local URL printed in the terminal (usually `http://localhost:8501`).

## How to use
1. Choose **Image Analysis**.
2. Upload an image.
3. Select the model and confidence threshold in the sidebar.
4. Click **Analyze Image**.
5. Review the original and annotated images, metrics, summary, and detection table.
6. Use **AI Performance** to run a runtime benchmark on multiple images.

## Project structure
```text
intelligent_vision_agent/
├── app.py
├── requirements.txt
└── README.md
```

## Notes and limitations
- The first run downloads the selected YOLO weights (`yolov8n.pt` or `yolov8s.pt`).
- The benchmark measures inference runtime and detection volume; it does not calculate model accuracy without ground-truth labels.
- Detection quality depends on lighting, image quality, object size, and the pretrained model's known classes.
- The pipeline stages are modular components of a vision workflow, not separate autonomous LLM agents.
- Use only images you have permission to process.

## Ideas for future improvements
- Add video upload and frame-by-frame detection
- Add object tracking
- Add a labeled test dataset and mAP/precision/recall evaluation
- Store history in SQLite
- Add authentication and deployment configuration
