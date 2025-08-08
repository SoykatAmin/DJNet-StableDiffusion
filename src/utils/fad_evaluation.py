"""
Fréchet Audio Distance (FAD) evaluation for DJ transitions
"""
import torch
import numpy as np
from scipy.linalg import sqrtm
import librosa
from pathlib import Path
import soundfile as sf
import warnings
warnings.filterwarnings('ignore')

class FADEvaluator:
    """
    Fréchet Audio Distance evaluation for DJ transitions
    Uses VGGish embeddings to measure perceptual audio quality
    """
    
    def __init__(self, sample_rate=16000, model_name='vggish'):
        self.sample_rate = sample_rate  # VGGish expects 16kHz
        self.model_name = model_name
        self.feature_extractor = self._load_feature_extractor()
        print(f"Initialized FAD evaluator with {model_name}")
    
    def _load_feature_extractor(self):
        """Load pre-trained VGGish model for feature extraction"""
        try:
            # Try TensorFlow Hub VGGish first
            import tensorflow as tf
            import tensorflow_hub as hub
            tf.config.set_visible_devices([], 'GPU')  # Use CPU for TF
            print("Loading VGGish from TensorFlow Hub...")
            return hub.load('https://tfhub.dev/google/vggish/1')
        except Exception as e:
            print(f"TensorFlow Hub failed: {e}")
            try:
                # Fallback to torchvggish
                import torchvggish
                print("Loading VGGish from torchvggish...")
                vggish = torchvggish.vggish()
                vggish.eval()
                return vggish
            except Exception as e2:
                print(f"torchvggish failed: {e2}")
                # Fallback to simple MFCC features
                print("Using MFCC features as fallback...")
                return "mfcc"
    
    def extract_features_mfcc(self, audio_files):
        """Fallback MFCC feature extraction"""
        features = []
        
        for audio_file in audio_files:
            try:
                # Load audio
                audio, sr = librosa.load(audio_file, sr=self.sample_rate)
                
                # Extract MFCC features
                mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
                mfcc_features = np.mean(mfcc.T, axis=0)  # Average over time
                features.append(mfcc_features)
                
            except Exception as e:
                print(f"Error processing {audio_file}: {e}")
                continue
        
        return np.array(features)
    
    def extract_features_vggish(self, audio_files):
        """Extract VGGish features from audio files"""
        features = []
        
        for audio_file in audio_files:
            try:
                # Load audio
                audio, sr = librosa.load(audio_file, sr=self.sample_rate)
                
                # Ensure minimum length (VGGish expects at least 0.975 seconds)
                min_length = int(0.975 * self.sample_rate)
                if len(audio) < min_length:
                    audio = np.pad(audio, (0, min_length - len(audio)), 'constant')
                
                # Extract features
                if hasattr(self.feature_extractor, 'signatures'):
                    # TensorFlow Hub VGGish
                    import tensorflow as tf
                    audio_tensor = tf.constant(audio, dtype=tf.float32)
                    feature = self.feature_extractor(audio_tensor)
                    features.append(feature.numpy())
                else:
                    # torchvggish
                    import torch
                    audio_tensor = torch.tensor(audio).unsqueeze(0)
                    with torch.no_grad():
                        feature = self.feature_extractor.forward(audio_tensor)
                    features.append(feature.numpy())
                
            except Exception as e:
                print(f"Error processing {audio_file}: {e}")
                continue
        
        if len(features) == 0:
            raise ValueError("No features could be extracted")
            
        return np.vstack(features)
    
    def extract_features(self, audio_files):
        """Extract features from audio files"""
        if self.feature_extractor == "mfcc":
            return self.extract_features_mfcc(audio_files)
        else:
            return self.extract_features_vggish(audio_files)
    
    def calculate_fad(self, real_features, generated_features):
        """
        Calculate Fréchet Audio Distance between real and generated features
        
        Args:
            real_features: Features from real audio samples
            generated_features: Features from generated audio samples
            
        Returns:
            fad_score: Fréchet Audio Distance
        """
        # Ensure we have enough samples
        if len(real_features) < 2 or len(generated_features) < 2:
            raise ValueError("Need at least 2 samples in each set for FAD calculation")
        
        # Calculate means
        mu_real = np.mean(real_features, axis=0)
        mu_gen = np.mean(generated_features, axis=0)
        
        # Calculate covariances
        sigma_real = np.cov(real_features, rowvar=False)
        sigma_gen = np.cov(generated_features, rowvar=False)
        
        # Add small regularization to avoid numerical issues
        eps = 1e-6
        sigma_real += eps * np.eye(sigma_real.shape[0])
        sigma_gen += eps * np.eye(sigma_gen.shape[0])
        
        # Calculate FAD
        diff = mu_real - mu_gen
        
        try:
            covmean = sqrtm(sigma_real @ sigma_gen)
            
            # Handle complex results from sqrtm
            if np.iscomplexobj(covmean):
                if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                    print("Warning: Imaginary component in covariance matrix sqrt")
                covmean = covmean.real
            
            fad = diff @ diff + np.trace(sigma_real + sigma_gen - 2 * covmean)
            
        except Exception as e:
            print(f"Error in FAD calculation: {e}")
            # Fallback to simpler metric
            fad = np.linalg.norm(diff) + np.trace(sigma_real) + np.trace(sigma_gen)
        
        return float(fad)
    
    def evaluate_transitions(self, real_transition_dir, generated_transition_dir):
        """
        Evaluate generated transitions against real ones using FAD
        
        Args:
            real_transition_dir: Directory containing real transition audio files
            generated_transition_dir: Directory containing generated transition audio files
            
        Returns:
            dict: Evaluation results including FAD score and interpretation
        """
        # Get file lists
        real_files = list(Path(real_transition_dir).glob("*.wav"))
        generated_files = list(Path(generated_transition_dir).glob("*.wav"))
        
        if len(real_files) == 0:
            raise ValueError(f"No real audio files found in {real_transition_dir}")
        if len(generated_files) == 0:
            raise ValueError(f"No generated audio files found in {generated_transition_dir}")
        
        print(f"Evaluating {len(generated_files)} generated vs {len(real_files)} real transitions")
        
        # Extract features
        print("Extracting features from real transitions...")
        real_features = self.extract_features(real_files[:min(100, len(real_files))])  # Limit for memory
        
        print("Extracting features from generated transitions...")
        generated_features = self.extract_features(generated_files[:min(100, len(generated_files))])
        
        # Calculate FAD
        print("Calculating FAD score...")
        fad_score = self.calculate_fad(real_features, generated_features)
        
        return {
            'fad_score': fad_score,
            'real_transitions': len(real_files),
            'generated_transitions': len(generated_files),
            'real_features_shape': real_features.shape,
            'generated_features_shape': generated_features.shape,
            'interpretation': self._interpret_fad_score(fad_score)
        }
    
    def _interpret_fad_score(self, fad_score):
        """Interpret FAD score"""
        if fad_score < 1.0:
            return "Excellent - Very similar to real transitions"
        elif fad_score < 5.0:
            return "Good - Close to real transition quality"
        elif fad_score < 15.0:
            return "Fair - Noticeable differences from real transitions"
        elif fad_score < 50.0:
            return "Poor - Significant differences from real transitions"
        else:
            return "Very Poor - Large differences from real transitions"
    
    def compare_models(self, real_dir, *generated_dirs):
        """
        Compare multiple models using FAD
        
        Args:
            real_dir: Directory with real transitions
            *generated_dirs: Variable number of generated transition directories
            
        Returns:
            dict: Comparison results for all models
        """
        results = {}
        
        for i, gen_dir in enumerate(generated_dirs):
            model_name = f"Model_{i+1}" if len(generated_dirs) > 1 else "Generated"
            try:
                result = self.evaluate_transitions(real_dir, gen_dir)
                results[model_name] = result
                print(f"{model_name}: FAD = {result['fad_score']:.3f} ({result['interpretation']})")
            except Exception as e:
                print(f"Error evaluating {model_name}: {e}")
                results[model_name] = {'error': str(e)}
        
        return results
