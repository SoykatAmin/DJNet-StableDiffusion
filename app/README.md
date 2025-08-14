# DJ Transition Generator Web App

A Flask web application that allows users to upload two audio files and generate smooth transitions between them using our trained U-Net model.

## Supported Formats

- WAV (recommended)
- MP3
- FLAC
- M4A
- AAC

**File Size Limit**: 50MB per file

## Quick Start

### 1. Install Dependencies

```bash
cd app
pip install -r requirements.txt
```

### 2. Ensure Model is Available

Make sure you have a trained model checkpoint in this location:
- `../checkpoints/5k/best_model_kaggle.pt`

### 3. Run the Application

```bash
python app.py
```

The web app will be available at `http://localhost:5000`

## File Structure

```
app/
├── app.py                 # Main Flask application
├── templates/
│   └── index.html        # Web interface template
├── static/               # Static files (CSS, JS, images)
├── uploads/              # Temporary uploaded files
├── outputs/              # Generated transitions
├── requirements.txt      # Python dependencies
└── README.md            # This file
```