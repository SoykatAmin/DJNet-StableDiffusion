# DJ Transition Generator Web App

A Flask web application that allows users to upload two audio files and generate smooth transitions between them using our trained U-Net model.

## Features

- 🎵 **Easy Upload**: Drag and drop interface for audio files
- 🎛️ **Segment Selection**: Choose any 12-second segment from your tracks with intuitive sliders
- 🎧 **Instant Playback**: Listen to transitions directly in your browser without downloading
- 📊 **Audio Information**: View track details (duration, format, sample rate, channels)
- 🎛️ **AI-Powered**: Uses trained U-Net model for transition generation
- 📥 **Instant Download**: Get your transition as a WAV file
- 🧹 **Session Management**: Automatic cleanup of uploaded files
- 📱 **Responsive Design**: Works on desktop and mobile devices

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

Make sure you have a trained model checkpoint in one of these locations:
- `../checkpoints/5k/best_model_kaggle.pt`
- `../checkpoints/production_model_epoch_50.pt`
- `../checkpoints/production_model_epoch_45.pt`
- `../checkpoints_long_segments/best_model.pt`

### 3. Run the Application

```bash
python app.py
```

The web app will be available at `http://localhost:5000`

## How It Works

1. **Upload**: Users upload two audio files through the web interface
2. **Analyze**: The server analyzes audio files and displays duration/format information
3. **Select Segments**: Users choose starting points for each track using interactive sliders
4. **Processing**: The server converts selected audio segments to mel-spectrograms
5. **Generation**: The U-Net model generates a transition spectrogram
6. **Reconstruction**: The transition is converted back to audio using Griffin-Lim
7. **Playback**: Users can listen to the transition instantly in the browser
8. **Download**: Users can download the generated transition file

## API Endpoints

- `GET /` - Main web interface
- `POST /upload` - Upload files and get audio information
- `POST /generate` - Generate transition with selected segments
- `GET /play/<session_id>` - Stream generated transition for playback
- `GET /download/<session_id>` - Download generated transition
- `GET /status` - Check model status
- `POST /cleanup/<session_id>` - Clean up session files

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

## Configuration

Key parameters can be modified in `app.py`:

- `MAX_FILE_SIZE`: Maximum upload size (default: 50MB)
- `ALLOWED_EXTENSIONS`: Supported file formats
- `UPLOAD_FOLDER`: Directory for uploaded files
- `OUTPUT_FOLDER`: Directory for generated transitions

## Security Notes

- Change the `secret_key` in production
- Consider adding user authentication for production use
- Implement rate limiting for heavy usage
- Add HTTPS in production deployment

## Production Deployment

For production deployment, consider:

1. **WSGI Server**: Use Gunicorn or uWSGI instead of Flask dev server
2. **Reverse Proxy**: Place behind Nginx or Apache
3. **Environment Variables**: Store sensitive configuration in env vars
4. **File Cleanup**: Implement automatic cleanup of old sessions
5. **Monitoring**: Add logging and monitoring

Example with Gunicorn:
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## Troubleshooting

### Model Not Loading
- Check that checkpoint files exist in the expected locations
- Ensure the model architecture matches the checkpoint
- Check CUDA availability if using GPU

### Upload Errors
- Verify file format is supported
- Check file size limits
- Ensure sufficient disk space

### Audio Quality Issues
- Use WAV format for best quality
- Check input audio quality
- Ensure proper sample rates

## Example Usage

1. Open `http://localhost:5000` in your browser
2. Upload two audio files (Source Track A and Source Track B)
3. **NEW**: View audio information (duration, format, etc.)
4. **NEW**: Use sliders to choose starting points for each track
5. **NEW**: Preview segments and adjust as needed
6. Click "Generate Transition"
7. Wait for processing (usually 30-60 seconds)
8. **NEW**: Listen to the transition directly in your browser
9. Download the generated transition file
10. Use the transition in your DJ software!

## Performance

- **Generation Time**: ~30-60 seconds depending on hardware
- **Memory Usage**: ~2-4GB RAM with model loaded
- **GPU Acceleration**: Automatically used if available
- **Concurrent Users**: Limited by server resources

## License

This web application is part of the DJ Transition Generator project for educational and research purposes.
