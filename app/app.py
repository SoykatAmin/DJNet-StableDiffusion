#!/usr/bin/env python3
"""
Flask Web App for DJ Transition Generation
Upload two songs and generate a smooth transition between them
"""
import os
import sys
import uuid
import shutil
from datetime import datetime
from pathlib import Path
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, jsonify, send_file, flash, redirect, url_for
import torch
import torchaudio
import librosa
import soundfile as sf
import numpy as np

# Add parent directory to path to import our modules
sys.path.append('../')
sys.path.append('../src')
sys.path.append('../configs')

try:
    from long_segment_config import *
    print("Configuration loaded successfully")
except ImportError:
    print("Using default configuration")
    SAMPLE_RATE = 22050
    N_FFT = 2048
    HOP_LENGTH = 512
    N_MELS = 128
    SPECTROGRAM_HEIGHT = 128
    SPECTROGRAM_WIDTH = 512
    SEGMENT_DURATION = 12.0
    IN_CHANNELS = 3
    OUT_CHANNELS = 1
    MODEL_DIM = 512

from src.models.production_unet import ProductionUNet
from src.utils.audio_processing import AudioProcessor
from src.utils.evaluation import TransitionEvaluator

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Change this in production!

# Configuration
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'm4a', 'aac'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Global model instance (loaded once at startup)
generator = None

class ExtendedAudioProcessor(AudioProcessor):
    """Extended AudioProcessor with segment selection"""
    
    def audio_to_spectrogram_with_offset(self, audio_path, start_time=0):
        """Convert audio file to normalized mel spectrogram with time offset"""
        print(f"   Loading: {audio_path} (starting at {start_time}s)")
        
        # Load audio file
        waveform, orig_sample_rate = torchaudio.load(audio_path)
        
        # Resample if necessary
        if orig_sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=orig_sample_rate,
                new_freq=self.sample_rate
            )
            waveform = resampler(waveform)
        
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Calculate start sample
        start_sample = int(start_time * self.sample_rate)
        total_samples = waveform.shape[1]
        
        # Extract segment
        target_samples = int(self.segment_duration * self.sample_rate)
        end_sample = start_sample + target_samples
        
        if start_sample >= total_samples:
            # Start time is beyond audio length, create silence
            waveform = torch.zeros(1, target_samples)
        elif end_sample > total_samples:
            # Extract what we can and pad with silence
            available_audio = waveform[:, start_sample:]
            padding_needed = target_samples - available_audio.shape[1]
            padding = torch.zeros(1, padding_needed)
            waveform = torch.cat([available_audio, padding], dim=1)
        else:
            # Extract the requested segment
            waveform = waveform[:, start_sample:end_sample]
        
        # Convert to mel spectrogram
        mel_spec = self.mel_transform(waveform)
        
        # Convert to log scale
        log_mel_spec = torch.log(mel_spec + 1e-8)
        
        # Remove channel dimension
        log_mel_spec = log_mel_spec.squeeze(0)
        
        # Crop to target size (prefer cropping over interpolation to preserve audio quality)
        target_height, target_width = self.spectrogram_size
        current_height, current_width = log_mel_spec.shape
        
        # Crop frequency dimension (mel bins) if needed
        if current_height > target_height:
            log_mel_spec = log_mel_spec[:target_height, :]
        elif current_height < target_height:
            padding = torch.full((target_height - current_height, current_width), -10.0)
            log_mel_spec = torch.cat([log_mel_spec, padding], dim=0)
        
        # Crop time dimension if needed
        if current_width > target_width:
            log_mel_spec = log_mel_spec[:, :target_width]
        elif current_width < target_width:
            padding = torch.full((target_height, target_width - current_width), -10.0)
            log_mel_spec = torch.cat([log_mel_spec, padding], dim=1)
        
        # Normalize to [-1, 1]
        spec_min = log_mel_spec.min()
        spec_max = log_mel_spec.max()
        
        if spec_max > spec_min:
            normalized = (log_mel_spec - spec_min) / (spec_max - spec_min)
        else:
            normalized = torch.zeros_like(log_mel_spec)
        
        normalized = normalized * 2.0 - 1.0
        
        return normalized
    
    def get_audio_info(self, audio_path):
        """Get audio file information"""
        try:
            info = sf.info(audio_path)
            return {
                'duration': info.duration,
                'sample_rate': info.samplerate,
                'channels': info.channels,
                'format': info.format
            }
        except Exception as e:
            print(f"Error getting audio info: {e}")
            return None

class CrossfadeGenerator:
    """Simple crossfade transition generator"""
    
    def __init__(self):
        self.sample_rate = SAMPLE_RATE
        self.segment_duration = SEGMENT_DURATION
    
    def generate_crossfade_transition(self, source_a_path, source_b_path, session_id, 
                                    start_a=0, start_b=0, crossfade_duration=4.0):
        """Generate a simple crossfade transition between two audio sources"""
        try:
            print(f"Generating crossfade transition for session {session_id}")
            print(f"   Source A segment: {start_a}s")
            print(f"   Source B segment: {start_b}s")
            print(f"   Crossfade duration: {crossfade_duration}s")
            
            # Create session output directory
            session_output_dir = Path(OUTPUT_FOLDER) / session_id
            session_output_dir.mkdir(exist_ok=True)
            
            # Load audio segments
            print("Loading audio segments...")
            audio_a = self._load_audio_segment(source_a_path, start_a)
            audio_b = self._load_audio_segment(source_b_path, start_b)
            
            # Generate crossfade
            print("Creating crossfade...")
            transition_audio = self._create_crossfade(audio_a, audio_b, crossfade_duration)
            
            # Save transition
            transition_path = session_output_dir / "transition.wav"
            sf.write(str(transition_path), transition_audio, self.sample_rate)
            
            print(f"Crossfade transition generated: {transition_path}")
            return str(transition_path), None
            
        except Exception as e:
            error_msg = f"Error generating crossfade: {str(e)}"
            print(f"Error: {error_msg}")
            return None, error_msg
    
    def _load_audio_segment(self, audio_path, start_time):
        """Load a segment of audio from file"""
        # Load audio
        audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        
        # Calculate segment boundaries
        start_sample = int(start_time * self.sample_rate)
        segment_samples = int(self.segment_duration * self.sample_rate)
        end_sample = start_sample + segment_samples
        
        # Extract segment
        if start_sample >= len(audio):
            # Start beyond audio length, return silence
            return np.zeros(segment_samples)
        elif end_sample > len(audio):
            # Extract available audio and pad with silence
            available_audio = audio[start_sample:]
            padding_needed = segment_samples - len(available_audio)
            return np.concatenate([available_audio, np.zeros(padding_needed)])
        else:
            # Extract the requested segment
            return audio[start_sample:end_sample]
    
    def _create_crossfade(self, audio_a, audio_b, crossfade_duration):
        """Create crossfade transition between two audio segments"""
        crossfade_samples = int(crossfade_duration * self.sample_rate)
        
        # Total transition duration is 12 seconds
        total_samples = int(self.segment_duration * self.sample_rate)
        
        # Calculate section lengths
        fade_in_end = crossfade_samples // 2
        fade_out_start = total_samples - (crossfade_samples // 2)
        
        # Create transition audio
        transition = np.zeros(total_samples)
        
        # First part: audio A fading out
        if fade_in_end > 0:
            # Take from beginning of audio A
            a_segment = audio_a[:fade_in_end]
            transition[:len(a_segment)] = a_segment
        
        # Middle part: crossfade
        if fade_out_start > fade_in_end:
            crossfade_length = fade_out_start - fade_in_end
            
            # Get segments for crossfade
            a_fade_start = fade_in_end
            a_fade_end = min(a_fade_start + crossfade_length, len(audio_a))
            a_crossfade = audio_a[a_fade_start:a_fade_end]
            
            b_crossfade = audio_b[:crossfade_length]
            
            # Ensure both segments are same length
            min_length = min(len(a_crossfade), len(b_crossfade), crossfade_length)
            a_crossfade = a_crossfade[:min_length]
            b_crossfade = b_crossfade[:min_length]
            
            # Create fade curves
            fade_out_curve = np.linspace(1.0, 0.0, min_length)
            fade_in_curve = np.linspace(0.0, 1.0, min_length)
            
            # Apply crossfade
            crossfaded = a_crossfade * fade_out_curve + b_crossfade * fade_in_curve
            transition[fade_in_end:fade_in_end + len(crossfaded)] = crossfaded
        
        # Last part: audio B fading in
        if fade_out_start < total_samples:
            remaining_samples = total_samples - fade_out_start
            b_segment_start = fade_out_start - fade_in_end + crossfade_samples // 2
            b_segment_end = b_segment_start + remaining_samples
            
            if b_segment_start < len(audio_b):
                b_segment = audio_b[b_segment_start:min(b_segment_end, len(audio_b))]
                transition[fade_out_start:fade_out_start + len(b_segment)] = b_segment
        
        return transition
class DJWebGenerator:
    """DJ Transition Generator for web interface"""
    
    def __init__(self, checkpoint_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.audio_processor = ExtendedAudioProcessor(
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            spectrogram_size=(SPECTROGRAM_HEIGHT, SPECTROGRAM_WIDTH),
            segment_duration=SEGMENT_DURATION
        )
        self.evaluator = TransitionEvaluator(sample_rate=SAMPLE_RATE)
        
        # Load model if checkpoint provided
        if checkpoint_path and os.path.exists(checkpoint_path):
            self.model, self.checkpoint = self.load_model(checkpoint_path)
            self.model_loaded = True
        else:
            self.model = None
            self.checkpoint = None
            self.model_loaded = False
            print(f"Warning: Model checkpoint not found: {checkpoint_path}")
    
    def load_model(self, checkpoint_path):
        """Load the trained model from checkpoint"""
        print(f"Loading model from {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Initialize model
        model = ProductionUNet(
            in_channels=IN_CHANNELS,
            out_channels=OUT_CHANNELS,
            model_dim=MODEL_DIM
        ).to(self.device)
        
        # Load model weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"Model loaded successfully on {self.device}")
        return model, checkpoint
    
    def generate_transition(self, source_a_path, source_b_path, session_id, start_a=0, start_b=0, method="model"):
        """Generate transition between two audio sources with method selection"""
        try:
            print(f"Generating transition for session {session_id} using {method}")
            print(f"   Source A segment: {start_a}s")
            print(f"   Source B segment: {start_b}s")
            
            # Create session output directory
            session_output_dir = Path(OUTPUT_FOLDER) / session_id
            session_output_dir.mkdir(exist_ok=True)
            
            if method == "crossfade":
                # Use simple crossfade
                crossfade_gen = CrossfadeGenerator()
                return crossfade_gen.generate_crossfade_transition(
                    source_a_path, source_b_path, session_id, start_a, start_b
                )
            elif method == "model":
                # Use AI model
                if not self.model_loaded:
                    return None, "Model not loaded. Please check checkpoint path or use crossfade method."
                
                # Convert audio to spectrograms with segment selection
                print("Processing audio files...")
                source_a_spec = self.audio_processor.audio_to_spectrogram_with_offset(source_a_path, start_a)
                source_b_spec = self.audio_processor.audio_to_spectrogram_with_offset(source_b_path, start_b)
                
                # Generate transition
                print("Generating AI transition...")
                transition_spec = self._generate_transition_spectrogram(source_a_spec, source_b_spec)
                
                # Convert back to audio
                print("Converting to audio...")
                transition_audio = self.audio_processor.spectrogram_to_audio(transition_spec)
                
                # Save transition
                transition_path = session_output_dir / "transition.wav"
                if hasattr(transition_audio, 'cpu'):
                    transition_audio = transition_audio.cpu()
                if hasattr(transition_audio, 'numpy'):
                    transition_audio = transition_audio.numpy()
                
                import soundfile as sf
                sf.write(str(transition_path), transition_audio, SAMPLE_RATE)
                
                print(f"AI transition generated: {transition_path}")
                return str(transition_path), None
            else:
                return None, f"Unknown method: {method}"
            
        except Exception as e:
            error_msg = f"Error generating transition: {str(e)}"
            print(f"Error: {error_msg}")
            return None, error_msg
    
    def _generate_transition_spectrogram(self, source_a_spec, source_b_spec):
        """Generate transition spectrogram using the model"""
        # Ensure tensors are on correct device
        source_a_spec = source_a_spec.to(self.device)
        source_b_spec = source_b_spec.to(self.device)
        
        # Create noise for the transition channel
        noise_spec = torch.randn_like(source_a_spec) * 0.1
        noise_spec = noise_spec.to(self.device)
        
        # Stack inputs: [source_a, source_b, noise]
        input_tensor = torch.stack([source_a_spec, source_b_spec, noise_spec], dim=0)
        input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension
        
        # Generate transition
        with torch.no_grad():
            transition = self.model(input_tensor)
        
        # Remove batch and channel dimensions
        transition = transition.squeeze(0).squeeze(0)
        
        return transition

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def init_model():
    """Initialize the model at startup"""
    global generator
    
    # Look for best checkpoint
    checkpoint_candidates = [
        "../checkpoints/5k/best_model_kaggle.pt",
        "../checkpoints/5k/epoch30.pt",
        "../checkpoints/production_model_epoch_45.pt",
        "../checkpoints_long_segments/best_model.pt"
    ]
    
    checkpoint_path = None
    for candidate in checkpoint_candidates:
        if os.path.exists(candidate):
            checkpoint_path = candidate
            break
    
    if checkpoint_path:
        print(f"🎯 Found checkpoint: {checkpoint_path}")
        generator = DJWebGenerator(checkpoint_path)
    else:
        print("⚠️ No model checkpoint found!")
        generator = DJWebGenerator()  # Initialize without model

@app.route('/')
def index():
    """Main page"""
    model_status = "loaded" if generator and generator.model_loaded else "not_loaded"
    return render_template('index.html', model_status=model_status)

@app.route('/upload', methods=['POST'])
def upload_files():
    """Handle file upload and get audio info"""
    try:
        # Check if files are present
        if 'source_a' not in request.files or 'source_b' not in request.files:
            return jsonify({'error': 'Both audio files are required'}), 400
        
        source_a_file = request.files['source_a']
        source_b_file = request.files['source_b']
        
        # Check if files are selected
        if source_a_file.filename == '' or source_b_file.filename == '':
            return jsonify({'error': 'Both audio files must be selected'}), 400
        
        # Check file extensions
        if not (allowed_file(source_a_file.filename) and allowed_file(source_b_file.filename)):
            return jsonify({'error': f'Allowed file types: {", ".join(ALLOWED_EXTENSIONS)}'}), 400
        
        # Generate unique session ID
        session_id = str(uuid.uuid4())
        
        # Create session upload directory
        session_upload_dir = Path(UPLOAD_FOLDER) / session_id
        session_upload_dir.mkdir(exist_ok=True)
        
        # Save uploaded files
        source_a_filename = secure_filename(source_a_file.filename)
        source_b_filename = secure_filename(source_b_file.filename)
        
        source_a_path = session_upload_dir / f"source_a_{source_a_filename}"
        source_b_path = session_upload_dir / f"source_b_{source_b_filename}"
        
        source_a_file.save(str(source_a_path))
        source_b_file.save(str(source_b_path))
        
        print(f"Files uploaded to session {session_id}")
        
        # Get audio information
        if generator and generator.audio_processor:
            source_a_info = generator.audio_processor.get_audio_info(str(source_a_path))
            source_b_info = generator.audio_processor.get_audio_info(str(source_b_path))
        else:
            source_a_info = source_b_info = None
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'source_a_info': source_a_info,
            'source_b_info': source_b_info,
            'message': 'Files uploaded successfully! Choose segments to generate transition.'
        })
        
    except Exception as e:
        print(f"Upload error: {str(e)}")
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/generate', methods=['POST'])
def generate_transition():
    """Generate transition with selected segments and method"""
    try:
        data = request.json
        session_id = data.get('session_id')
        start_a = float(data.get('start_a', 0))
        start_b = float(data.get('start_b', 0))
        method = data.get('method', 'model')  # Default to model
        
        if not session_id:
            return jsonify({'error': 'Session ID required'}), 400
        
        # Validate method
        if method not in ['model', 'crossfade']:
            return jsonify({'error': 'Method must be either "model" or "crossfade"'}), 400
        
        # Check if uploaded files exist
        session_upload_dir = Path(UPLOAD_FOLDER) / session_id
        source_a_files = list(session_upload_dir.glob("source_a_*"))
        source_b_files = list(session_upload_dir.glob("source_b_*"))
        
        if not source_a_files or not source_b_files:
            return jsonify({'error': 'Audio files not found. Please upload again.'}), 404
        
        source_a_path = str(source_a_files[0])
        source_b_path = str(source_b_files[0])
        
        # Generate transition with selected method
        if method == 'model' and (not generator or not generator.model_loaded):
            return jsonify({'error': 'AI model not loaded. Please use crossfade method or check server configuration.'}), 500
        
        transition_path, error = generator.generate_transition(
            source_a_path, 
            source_b_path, 
            session_id,
            start_a,
            start_b,
            method
        )
        
        if error:
            return jsonify({'error': error}), 500
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'method': method,
            'message': f'Transition generated successfully using {method}!'
        })
        
    except Exception as e:
        print(f"Generation error: {str(e)}")
        return jsonify({'error': f'Generation failed: {str(e)}'}), 500

@app.route('/play/<session_id>')
def play_transition(session_id):
    """Stream generated transition for playback"""
    try:
        transition_path = Path(OUTPUT_FOLDER) / session_id / "transition.wav"
        
        if not transition_path.exists():
            return jsonify({'error': 'Transition file not found'}), 404
        
        return send_file(
            str(transition_path),
            mimetype='audio/wav',
            as_attachment=False
        )
        
    except Exception as e:
        print(f"Play error: {str(e)}")
        return jsonify({'error': f'Playback failed: {str(e)}'}), 500

@app.route('/download/<session_id>')
def download_transition(session_id):
    """Download generated transition"""
    try:
        transition_path = Path(OUTPUT_FOLDER) / session_id / "transition.wav"
        
        if not transition_path.exists():
            return jsonify({'error': 'Transition file not found'}), 404
        
        return send_file(
            str(transition_path),
            as_attachment=True,
            download_name=f"dj_transition_{session_id[:8]}.wav",
            mimetype='audio/wav'
        )
        
    except Exception as e:
        print(f"Download error: {str(e)}")
        return jsonify({'error': f'Download failed: {str(e)}'}), 500

@app.route('/status')
def status():
    """API endpoint to check model status"""
    if generator and generator.model_loaded:
        return jsonify({
            'model_loaded': True,
            'device': str(generator.device),
            'checkpoint_info': {
                'epoch': generator.checkpoint.get('epoch', 'Unknown') if generator.checkpoint else 'Unknown',
                'val_loss': generator.checkpoint.get('best_val_loss', 'Unknown') if generator.checkpoint else 'Unknown'
            }
        })
    else:
        return jsonify({
            'model_loaded': False,
            'error': 'Model checkpoint not found'
        })

@app.route('/cleanup/<session_id>', methods=['POST'])
def cleanup_session(session_id):
    """Clean up session files"""
    try:
        # Clean upload directory
        upload_dir = Path(UPLOAD_FOLDER) / session_id
        if upload_dir.exists():
            shutil.rmtree(upload_dir)
        
        # Clean output directory
        output_dir = Path(OUTPUT_FOLDER) / session_id
        if output_dir.exists():
            shutil.rmtree(output_dir)
        
        return jsonify({'success': True, 'message': 'Session cleaned up'})
        
    except Exception as e:
        return jsonify({'error': f'Cleanup failed: {str(e)}'}), 500

if __name__ == '__main__':
    # Initialize model at startup
    print("🚀 Starting DJ Transition Generator Web App")
    init_model()
    
    # Create required directories
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    print("🌐 Starting Flask server...")
    app.run(debug=True, host='0.0.0.0', port=5000)
