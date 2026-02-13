"""
Utility functions for camera application and metadata analysis.
"""
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Optional
import matplotlib.pyplot as plt
def analyze_metadata(csv_file: str) -> pd.DataFrame:
    """
    Load and analyze metadata CSV file.
    Args:
        csv_file: Path to metadata CSV file
    Returns:
        DataFrame with additional analysis columns
    """
    df = pd.read_csv(csv_file)
    # Calculate time differences between frames
    if 'timestamp_ticks' in df.columns:
        df['timestamp_diff'] = df['timestamp_ticks'].diff()
        df['timestamp_ms'] = (df['timestamp_ticks'] - df['timestamp_ticks'].iloc[0]) / 1000.0
    # Calculate exposure statistics
    if 'exposure_time_us' in df.columns:
        print(f"Exposure time statistics:")
        print(f"  Mean: {df['exposure_time_us'].mean():.2f} µs")
        print(f"  Min: {df['exposure_time_us'].min():.2f} µs")
        print(f"  Max: {df['exposure_time_us'].max():.2f} µs")
        print(f"  Std: {df['exposure_time_us'].std():.2f} µs")
    # Analyze trigger events (Line1 transitions)
    if 'line1_status' in df.columns:
        # Find rising edges (trigger events)
        df['line1_rising_edge'] = (df['line1_status'].diff() > 0).astype(int)
        num_triggers = df['line1_rising_edge'].sum()
        print(f"\nTrigger events (Line1 rising edges): {num_triggers}")
    return df
def plot_metadata(csv_file: str, output_file: str = None):
    """
    Create visualization plots for metadata.
    Args:
        csv_file: Path to metadata CSV file
        output_file: Optional path to save plot image
    """
    df = pd.read_csv(csv_file)
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle(f'Metadata Analysis: {Path(csv_file).stem}', fontsize=14)
    # Plot 1: Frame timing
    if 'timestamp_ticks' in df.columns:
        timestamp_ms = (df['timestamp_ticks'] - df['timestamp_ticks'].iloc[0]) / 1000.0
        axes[0].plot(df['frame_id'], timestamp_ms, linewidth=0.5)
        axes[0].set_xlabel('Frame ID')
        axes[0].set_ylabel('Time (ms)')
        axes[0].set_title('Frame Timing')
        axes[0].grid(True, alpha=0.3)
    # Plot 2: Exposure time
    if 'exposure_time_us' in df.columns:
        axes[1].plot(df['frame_id'], df['exposure_time_us'], linewidth=0.5)
        axes[1].set_xlabel('Frame ID')
        axes[1].set_ylabel('Exposure Time (µs)')
        axes[1].set_title('Exposure Time per Frame')
        axes[1].grid(True, alpha=0.3)
    # Plot 3: GPIO line status
    if 'line1_status' in df.columns:
        axes[2].plot(df['frame_id'], df['line1_status'], linewidth=0.5, label='Line1')
    if 'line2_status' in df.columns:
        axes[2].plot(df['frame_id'], df['line2_status'], linewidth=0.5, label='Line2')
    if 'line3_status' in df.columns:
        axes[2].plot(df['frame_id'], df['line3_status'], linewidth=0.5, label='Line3')
    axes[2].set_xlabel('Frame ID')
    axes[2].set_ylabel('Line Status (0=Low, 1=High)')
    axes[2].set_title('GPIO Line Status')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim(-0.1, 1.1)
    plt.tight_layout()
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_file}")
    else:
        plt.show()
def extract_triggered_frames(csv_file: str, video_file: str, output_dir: str = "triggered_frames"):
    """
    Extract frames where Line1 was high (triggered) from video.
    Args:
        csv_file: Path to metadata CSV file
        video_file: Path to video file
        output_dir: Directory to save extracted frames
    """
    try:
        import cv2
    except ImportError:
        print("OpenCV (cv2) required for this function. Install with: pip install opencv-python")
        return
    # Load metadata
    df = pd.read_csv(csv_file)
    # Find triggered frames
    if 'line1_status' not in df.columns:
        print("No line1_status column found in metadata")
        return
    triggered_frames = df[df['line1_status'] == 1]['frame_id'].tolist()
    print(f"Found {len(triggered_frames)} triggered frames")
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    # Open video
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print(f"Failed to open video: {video_file}")
        return
    # Extract frames
    frame_count = 0
    saved_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count in triggered_frames:
            output_file = output_path / f"frame_{frame_count:06d}.png"
            cv2.imwrite(str(output_file), frame)
            saved_count += 1
        frame_count += 1
    cap.release()
    print(f"Extracted {saved_count} frames to {output_dir}/")
def calculate_frame_rate(csv_file: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Calculate average frame rate from metadata.
    Args:
        csv_file: Path to metadata CSV file
    Returns:
        Tuple of (mean_fps, std_fps) or (None, None) if no timestamp data
    """
    df = pd.read_csv(csv_file)
    if 'timestamp_ticks' not in df.columns:
        print("No timestamp data available")
        return None, None
    # Calculate time differences in seconds (assuming ticks are in microseconds)
    time_diffs = df['timestamp_ticks'].diff().dropna()
    time_diffs_sec = time_diffs / 1_000_000.0  # Convert to seconds
    # Calculate instantaneous FPS
    fps = 1.0 / time_diffs_sec
    mean_fps = fps.mean()
    std_fps = fps.std()
    print(f"Frame rate statistics:")
    print(f"  Mean: {mean_fps:.2f} FPS")
    print(f"  Std: {std_fps:.2f} FPS")
    print(f"  Min: {fps.min():.2f} FPS")
    print(f"  Max: {fps.max():.2f} FPS")
    return mean_fps, std_fps
def find_sync_events(csv_file: str, line: int = 1) -> List[int]:
    """
    Find frame IDs where a GPIO line had a rising edge (sync/trigger event).
    Args:
        csv_file: Path to metadata CSV file
        line: GPIO line number (1, 2, or 3)
    Returns:
        List of frame IDs with rising edge events
    """
    df = pd.read_csv(csv_file)
    line_col = f'line{line}_status'
    if line_col not in df.columns:
        print(f"Column {line_col} not found in metadata")
        return []
    # Find rising edges
    rising_edges = df[line_col].diff() > 0
    event_frames = df[rising_edges]['frame_id'].tolist()
    print(f"Found {len(event_frames)} rising edge events on Line{line}")
    return event_frames
def export_metadata_summary(csv_file: str, output_file: str = None):
    """
    Export a summary of metadata statistics to text file.
    Args:
        csv_file: Path to metadata CSV file
        output_file: Path to output text file (default: csv_file with .txt extension)
    """
    df = pd.read_csv(csv_file)
    if output_file is None:
        output_file = Path(csv_file).with_suffix('.txt')
    with open(output_file, 'w') as f:
        f.write(f"Metadata Summary: {Path(csv_file).name}\n")
        f.write("=" * 60 + "\n\n")
        # Basic info
        f.write(f"Total Frames: {len(df)}\n")
        f.write(f"Recording Duration: {df['timestamp_software'].max() - df['timestamp_software'].min():.2f} seconds\n\n")
        # Exposure statistics
        if 'exposure_time_us' in df.columns:
            f.write("Exposure Time Statistics:\n")
            f.write(f"  Mean: {df['exposure_time_us'].mean():.2f} µs\n")
            f.write(f"  Min: {df['exposure_time_us'].min():.2f} µs\n")
            f.write(f"  Max: {df['exposure_time_us'].max():.2f} µs\n")
            f.write(f"  Std: {df['exposure_time_us'].std():.2f} µs\n\n")
        # Frame rate
        if 'timestamp_ticks' in df.columns:
            time_diffs = df['timestamp_ticks'].diff().dropna()
            time_diffs_sec = time_diffs / 1_000_000.0
            fps = 1.0 / time_diffs_sec
            f.write("Frame Rate Statistics:\n")
            f.write(f"  Mean: {fps.mean():.2f} FPS\n")
            f.write(f"  Min: {fps.min():.2f} FPS\n")
            f.write(f"  Max: {fps.max():.2f} FPS\n")
            f.write(f"  Std: {fps.std():.2f} FPS\n\n")
        # Trigger events
        for line_num in [1, 2, 3]:
            line_col = f'line{line_num}_status'
            if line_col in df.columns:
                rising_edges = (df[line_col].diff() > 0).sum()
                f.write(f"Line{line_num} Rising Edges: {rising_edges}\n")
    print(f"Summary saved to: {output_file}")
# Example usage
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python utils.py <metadata.csv>")
        print("\nExample:")
        print("  python utils.py recording_20260108_120000_metadata.csv")
        sys.exit(1)
    csv_file = sys.argv[1]
    if not Path(csv_file).exists():
        print(f"File not found: {csv_file}")
        sys.exit(1)
    print(f"Analyzing: {csv_file}\n")
    # Run analysis
    df = analyze_metadata(csv_file)
    print(f"\nDataFrame shape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")
    # Calculate frame rate
    print("\n" + "=" * 60)
    calculate_frame_rate(csv_file)
    # Find sync events
    print("\n" + "=" * 60)
    find_sync_events(csv_file, line=1)
    # Export summary
    print("\n" + "=" * 60)
    export_metadata_summary(csv_file)
    # Ask to create plot
    print("\n" + "=" * 60)
    response = input("Create visualization plot? (y/n): ")
    if response.lower() == 'y':
        plot_file = Path(csv_file).with_suffix('.png')
        plot_metadata(csv_file, str(plot_file))
