"""
Data loaders for real GPS datasets (FGI-SpoofRepo, TEXBAT).
"""
import os
import numpy as np
from typing import Tuple, List, Dict, Optional
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from preprocessing.signal_processing import read_iq_data


def load_fgi_dataset(data_dir: str, scenario: Optional[str] = None) -> Tuple[List[np.ndarray], List[int], List[Dict]]:
    """
    Load FGI-SpoofRepo dataset.
    
    This function provides a template for loading FGI-SpoofRepo data.
    Users must download the dataset manually from:
    https://github.com/Finnish-Geospatial-Institute/FGI-SpoofRepo
    
    Args:
        data_dir: Path to FGI-SpoofRepo data directory
        scenario: Optional specific scenario to load
    
    Returns:
        Tuple of (signals, labels, metadata)
    
    Notes:
        FGI-SpoofRepo structure (example):
        - data_dir/
          - scenario_1/
            - authentic/
              - file1.bin
            - spoofed/
              - file2.bin
          - scenario_2/
            ...
    
    Instructions for users:
    1. Download FGI-SpoofRepo from GitHub
    2. Extract to data/raw/fgi-spoof-repo/
    3. Point this function to that directory
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(
            f"FGI-SpoofRepo directory not found: {data_dir}\n\n"
            "Please download FGI-SpoofRepo from:\n"
            "https://github.com/Finnish-Geospatial-Institute/FGI-SpoofRepo\n\n"
            "Extract to: data/raw/fgi-spoof-repo/\n"
        )
    
    print(f"Loading FGI-SpoofRepo from: {data_dir}")
    
    # This is a template - actual implementation depends on FGI dataset structure
    signals = []
    labels = []
    metadata = []
    
    # Example structure parsing (adjust based on actual FGI format)
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(('.bin', '.dat')):
                file_path = os.path.join(root, file)
                
                # Determine label from directory structure
                if 'authentic' in root.lower() or 'genuine' in root.lower():
                    label = 0
                elif 'spoof' in root.lower():
                    label = 1
                else:
                    continue  # Skip unknown
                
                # Read signal (adjust parameters as needed)
                try:
                    # Example: read first 0.5s
                    fs = 5e6  # Adjust based on actual sampling rate
                    num_samples = int(0.5 * fs)
                    signal = read_iq_data(file_path, 0, num_samples)
                    
                    if signal is not None:
                        signals.append(signal)
                        labels.append(label)
                        metadata.append({
                            'filename': file,
                            'path': file_path,
                            'label': label,
                            'scenario': scenario or 'unknown'
                        })
                except Exception as e:
                    print(f"Warning: Could not load {file}: {e}")
    
    print(f"Loaded {len(signals)} signals from FGI-SpoofRepo")
    return signals, labels, metadata


def load_texbat_dataset(data_dir: str, fs: float = 5e6, 
                       segment_duration: float = 0.5,
                       spoof_start_time: float = 17.0,
                       max_segments: Optional[int] = None) -> Tuple[List[np.ndarray], List[int], List[Dict]]:
    """
    Load TEXBAT dataset with time-based labeling.
    
    Args:
        data_dir: Path to TEXBAT data directory
        fs: Sampling frequency (Hz)
        segment_duration: Duration of each segment (seconds)
        spoof_start_time: Time when spoofing starts (seconds)
        max_segments: Maximum segments to load (None = all)
    
    Returns:
        Tuple of (signals, labels, metadata)
    
    Notes:
        TEXBAT format: Binary files with int16 interleaved I/Q
        Label logic: Before spoof_start_time = Authentic (0), after = Spoofed (1)
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"TEXBAT directory not found: {data_dir}")
    
    print(f"Loading TEXBAT dataset from: {data_dir}")
    
    signals = []
    labels = []
    metadata = []
    
    # Find binary files
    files = [f for f in os.listdir(data_dir) if f.lower().endswith(('.bin', '.dat'))]
    
    if not files:
        raise ValueError(f"No binary files found in {data_dir}")
    
    segment_samples = int(segment_duration * fs)
    hop_samples = segment_samples // 2  # 50% overlap
    
    for file in files:
        file_path = os.path.join(data_dir, file)
        file_size = os.path.getsize(file_path)
        total_samples = file_size // 4  # 4 bytes per IQ sample
        
        print(f"Processing {file}...")
        
        segment_idx = 0
        for start_sample in range(0, total_samples - segment_samples, hop_samples):
            if max_segments and len(signals) >= max_segments:
                break
            
            # Read segment
            signal = read_iq_data(file_path, start_sample, segment_samples)
            if signal is None:
                continue
            
            # Determine label based on time
            segment_time = start_sample / fs
            label = 0 if segment_time < spoof_start_time else 1
            
            signals.append(signal)
            labels.append(label)
            metadata.append({
                'filename': file,
                'segment_index': segment_idx,
                'segment_time_s': segment_time,
                'label': label
            })
            
            segment_idx += 1
        
        if max_segments and len(signals) >= max_segments:
            break
    
    print(f"Loaded {len(signals)} segments from TEXBAT")
    print(f"  Authentic: {sum(1 for l in labels if l == 0)}")
    print(f"  Spoofed: {sum(1 for l in labels if l == 1)}")
    
    return signals, labels, metadata
