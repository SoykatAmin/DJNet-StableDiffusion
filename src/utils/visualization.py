import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Optional, Tuple, Union
import seaborn as sns


def plot_spectrograms(
    spectrograms: List[torch.Tensor],
    titles: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 5),
    cmap: str = 'viridis',
    vmin: float = -1.0,
    vmax: float = 1.0
) -> None:
    """
    Plot multiple spectrograms side by side.
    
    Args:
        spectrograms: List of spectrogram tensors
        titles: Optional titles for each spectrogram
        save_path: Optional path to save the plot
        figsize: Figure size
        cmap: Colormap for spectrograms
        vmin: Minimum value for colormap
        vmax: Maximum value for colormap
    """
    n_specs = len(spectrograms)
    
    fig, axes = plt.subplots(1, n_specs, figsize=figsize)
    if n_specs == 1:
        axes = [axes]
    
    for i, spec in enumerate(spectrograms):
        # Convert to numpy and handle dimensions
        spec_np = spec.numpy() if isinstance(spec, torch.Tensor) else spec
        
        # Plot spectrogram
        im = axes[i].imshow(
            spec_np,
            aspect='auto',
            origin='lower',
            cmap=cmap,
            vmin=vmin,
            vmax=vmax
        )
        
        # Add title
        if titles and i < len(titles):
            axes[i].set_title(titles[i], fontsize=12, fontweight='bold')
        
        # Labels
        axes[i].set_xlabel('Time Frames')
        if i == 0:
            axes[i].set_ylabel('Mel Bins')
        
        # Add colorbar
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved spectrogram plot to {save_path}")
    
    plt.show()


def plot_training_history(
    train_losses: List[float],
    val_losses: Optional[List[float]] = None,
    steps: Optional[List[int]] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6)
) -> None:
    """
    Plot training and validation loss curves.
    
    Args:
        train_losses: List of training losses
        val_losses: Optional list of validation losses
        steps: Optional list of step numbers
        save_path: Optional path to save the plot
        figsize: Figure size
    """
    if steps is None:
        steps = list(range(len(train_losses)))
    
    plt.figure(figsize=figsize)
    
    # Plot training loss
    plt.plot(steps, train_losses, label='Training Loss', color='blue', alpha=0.7)
    
    # Plot validation loss if provided
    if val_losses:
        val_steps = steps[:len(val_losses)]
        plt.plot(val_steps, val_losses, label='Validation Loss', color='red', alpha=0.7)
    
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add log scale option for better visualization
    plt.yscale('log')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved training history to {save_path}")
    
    plt.show()


def plot_transition_progression(
    spectrograms: List[torch.Tensor],
    timesteps: List[int],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (20, 4)
) -> None:
    """
    Plot the progression of a transition through diffusion timesteps.
    
    Args:
        spectrograms: List of spectrograms at different timesteps
        timesteps: List of corresponding timesteps
        save_path: Optional path to save the plot
        figsize: Figure size
    """
    n_steps = len(spectrograms)
    
    fig, axes = plt.subplots(1, n_steps, figsize=figsize)
    if n_steps == 1:
        axes = [axes]
    
    for i, (spec, timestep) in enumerate(zip(spectrograms, timesteps)):
        spec_np = spec.numpy() if isinstance(spec, torch.Tensor) else spec
        
        im = axes[i].imshow(
            spec_np,
            aspect='auto',
            origin='lower',
            cmap='viridis',
            vmin=-1,
            vmax=1
        )
        
        axes[i].set_title(f't={timestep}', fontsize=10)
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        
        if i == 0:
            axes[i].set_ylabel('Mel Bins')
        if i == n_steps // 2:
            axes[i].set_xlabel('Diffusion Denoising Process')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved transition progression to {save_path}")
    
    plt.show()


def plot_attention_maps(
    attention_maps: List[torch.Tensor],
    layer_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 10)
) -> None:
    """
    Plot attention maps from different layers of the model.
    
    Args:
        attention_maps: List of attention map tensors
        layer_names: Optional names for each layer
        save_path: Optional path to save the plot
        figsize: Figure size
    """
    n_layers = len(attention_maps)
    n_cols = min(4, n_layers)
    n_rows = (n_layers + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_layers > 1 else [axes]
    
    for i, attn_map in enumerate(attention_maps):
        if i >= len(axes):
            break
        
        # Average over heads if multi-head attention
        if attn_map.dim() > 2:
            attn_map = attn_map.mean(dim=0)
        
        attn_np = attn_map.numpy() if isinstance(attn_map, torch.Tensor) else attn_map
        
        im = axes[i].imshow(attn_np, cmap='Blues', aspect='auto')
        
        if layer_names and i < len(layer_names):
            axes[i].set_title(layer_names[i])
        else:
            axes[i].set_title(f'Layer {i+1}')
        
        axes[i].set_xlabel('Key Position')
        axes[i].set_ylabel('Query Position')
        
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
    
    # Hide unused subplots
    for i in range(len(attention_maps), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved attention maps to {save_path}")
    
    plt.show()


def plot_noise_schedule(
    scheduler,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> None:
    """
    Plot the noise schedule used in diffusion training.
    
    Args:
        scheduler: Diffusion noise scheduler
        save_path: Optional path to save the plot
        figsize: Figure size
    """
    timesteps = np.arange(scheduler.num_train_timesteps)
    
    # Get schedule parameters
    alphas = scheduler.alphas.numpy() if hasattr(scheduler, 'alphas') else None
    betas = scheduler.betas.numpy() if hasattr(scheduler, 'betas') else None
    alphas_cumprod = scheduler.alphas_cumprod.numpy() if hasattr(scheduler, 'alphas_cumprod') else None
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Plot betas
    if betas is not None:
        axes[0, 0].plot(timesteps, betas)
        axes[0, 0].set_title('Beta Schedule')
        axes[0, 0].set_xlabel('Timestep')
        axes[0, 0].set_ylabel('Beta')
        axes[0, 0].grid(True, alpha=0.3)
    
    # Plot alphas
    if alphas is not None:
        axes[0, 1].plot(timesteps, alphas)
        axes[0, 1].set_title('Alpha Schedule')
        axes[0, 1].set_xlabel('Timestep')
        axes[0, 1].set_ylabel('Alpha')
        axes[0, 1].grid(True, alpha=0.3)
    
    # Plot cumulative alphas
    if alphas_cumprod is not None:
        axes[1, 0].plot(timesteps, alphas_cumprod)
        axes[1, 0].set_title('Cumulative Alpha Schedule')
        axes[1, 0].set_xlabel('Timestep')
        axes[1, 0].set_ylabel('Alpha Cumprod')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot signal-to-noise ratio
    if alphas_cumprod is not None:
        snr = alphas_cumprod / (1 - alphas_cumprod)
        axes[1, 1].plot(timesteps, 10 * np.log10(snr))
        axes[1, 1].set_title('Signal-to-Noise Ratio')
        axes[1, 1].set_xlabel('Timestep')
        axes[1, 1].set_ylabel('SNR (dB)')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved noise schedule plot to {save_path}")
    
    plt.show()


def create_transition_animation(
    spectrograms: List[torch.Tensor],
    output_path: str,
    fps: int = 10,
    cmap: str = 'viridis'
) -> None:
    """
    Create an animation showing the transition generation process.
    
    Args:
        spectrograms: List of spectrograms at different stages
        output_path: Path to save the animation (should end with .gif or .mp4)
        fps: Frames per second
        cmap: Colormap for spectrograms
    """
    try:
        from matplotlib.animation import FuncAnimation, PillowWriter
    except ImportError:
        print("Animation requires matplotlib with pillow. Install with: pip install pillow")
        return
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Initialize with first spectrogram
    spec_np = spectrograms[0].numpy() if isinstance(spectrograms[0], torch.Tensor) else spectrograms[0]
    im = ax.imshow(spec_np, aspect='auto', origin='lower', cmap=cmap, vmin=-1, vmax=1)
    
    ax.set_xlabel('Time Frames')
    ax.set_ylabel('Mel Bins')
    title = ax.set_title('Transition Generation: Step 0')
    
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    def animate(frame):
        if frame < len(spectrograms):
            spec_np = spectrograms[frame].numpy() if isinstance(spectrograms[frame], torch.Tensor) else spectrograms[frame]
            im.set_array(spec_np)
            title.set_text(f'Transition Generation: Step {frame}')
        return [im, title]
    
    anim = FuncAnimation(fig, animate, frames=len(spectrograms), interval=1000//fps, blit=False)
    
    if output_path.endswith('.gif'):
        anim.save(output_path, writer=PillowWriter(fps=fps))
    else:
        # For other formats, try default writer
        anim.save(output_path, fps=fps)
    
    print(f"Saved animation to {output_path}")
    plt.close()


def plot_model_architecture(
    model,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> None:
    """
    Plot a simplified visualization of the model architecture.
    
    Args:
        model: The model to visualize
        save_path: Optional path to save the plot
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # This is a simplified visualization
    # For detailed architecture plots, consider using tools like torchviz
    
    layers = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            layers.append(name)
    
    # Create a simple block diagram
    y_positions = np.linspace(0.1, 0.9, len(layers))
    
    for i, (layer_name, y_pos) in enumerate(zip(layers[:20], y_positions[:20])):  # Limit to first 20 layers
        # Draw rectangle for layer
        rect = patches.Rectangle((0.1, y_pos-0.02), 0.8, 0.04, 
                               linewidth=1, edgecolor='black', facecolor='lightblue')
        ax.add_patch(rect)
        
        # Add layer name
        ax.text(0.5, y_pos, layer_name.split('.')[-1], ha='center', va='center', fontsize=8)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('Model Architecture (Simplified)')
    ax.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved model architecture to {save_path}")
    
    plt.show()


if __name__ == "__main__":
    # Test visualization utilities
    print("Testing visualization utilities...")
    
    # Create dummy spectrograms
    spec1 = torch.randn(128, 128)
    spec2 = torch.randn(128, 128)
    spec3 = torch.randn(128, 128)
    
    # Test spectrogram plotting
    plot_spectrograms(
        [spec1, spec2, spec3],
        titles=['Preceding', 'Transition', 'Following']
    )
    
    # Test training history
    train_losses = [1.0 - 0.01*i + 0.1*np.random.randn() for i in range(100)]
    val_losses = [0.8 - 0.005*i + 0.05*np.random.randn() for i in range(0, 100, 5)]
    
    plot_training_history(train_losses, val_losses)
    
    print("Visualization utilities test completed!")
