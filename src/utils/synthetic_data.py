"""
Synthetic GPS signal generator for testing and offline execution.
"""
import numpy as np
from typing import Tuple, List
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from preprocessing.signal_processing import generate_ca_code


def generate_synthetic_gps_signal(prn: int, fs: float, duration: float,
                                  cn0: float = 45.0, doppler: float = 0.0,
                                  spoofed: bool = False, 
                                  spoof_params: dict = None) -> np.ndarray:
    """
    Generate synthetic GPS signal with optional spoofing effects.
    
    Args:
        prn: PRN number (1-32)
        fs: Sampling frequency (Hz)
        duration: Signal duration (seconds)
        cn0: Carrier-to-Noise density ratio (dB-Hz)
        doppler: Doppler frequency shift (Hz)
        spoofed: Whether to apply spoofing effects
        spoof_params: Spoofing parameters dict with keys:
            - 'power_increase': Additional power in dB (default: 10)
            - 'secondary_peak': Add secondary peak (default: False)
            - 'secondary_delay': Delay of secondary peak in chips (default: 50)
            - 'secondary_power': Power of secondary relative to main (default: 0.5)
    
    Returns:
        Complex IQ signal
    """
    if spoof_params is None:
        spoof_params = {}
    
    num_samples = int(fs * duration)
    ca_chip_rate = 1.023e6
    
    # Generate C/A code
    ca_code = generate_ca_code(prn)
    
    # Oversample C/A code
    samples_per_chip = int(fs / ca_chip_rate)
    ca_oversampled = np.repeat(ca_code, samples_per_chip)
    
    # Tile to match signal length
    repeats = int(np.ceil(num_samples / len(ca_oversampled)))
    ca_signal = np.tile(ca_oversampled, repeats)[:num_samples]
    
    # Generate carrier with Doppler
    t = np.arange(num_samples) / fs
    carrier = np.exp(1j * 2 * np.pi * doppler * t)
    
    # Modulate
    signal = ca_signal * carrier
    
    # Apply spoofing effects if requested
    if spoofed:
        power_increase = spoof_params.get('power_increase', 10.0)
        signal = signal * np.sqrt(10 ** (power_increase / 10))
        
        # Add secondary peak for multipath/spoofing signature
        if spoof_params.get('secondary_peak', True):
            secondary_delay_chips = spoof_params.get('secondary_delay', 50)
            secondary_power = spoof_params.get('secondary_power', 0.5)
            
            delay_samples = int(secondary_delay_chips * samples_per_chip)
            secondary_signal = np.zeros(num_samples, dtype=complex)
            if delay_samples < num_samples:
                secondary_signal[delay_samples:] = signal[:-delay_samples] * secondary_power
                signal = signal + secondary_signal
    
    # Calculate noise power from C/N0
    # C/N0 = 10*log10(C / (N0 * BW))
    # Signal power
    signal_power = np.mean(np.abs(signal)**2)
    
    # Noise power per Hz
    n0_linear = signal_power / (10 ** (cn0 / 10))
    
    # Total noise power over bandwidth
    noise_power = n0_linear * fs
    
    # Add AWGN noise
    noise_i = np.random.randn(num_samples) * np.sqrt(noise_power / 2)
    noise_q = np.random.randn(num_samples) * np.sqrt(noise_power / 2)
    noise = noise_i + 1j * noise_q
    
    signal_noisy = signal + noise
    
    return signal_noisy


def generate_synthetic_dataset(num_authentic: int = 100, num_spoofed: int = 100,
                              fs: float = 5e6, duration: float = 0.5,
                              prn_range: Tuple[int, int] = (1, 5),
                              random_state: int = 42) -> Tuple[List[np.ndarray], List[int], List[dict]]:
    """
    Generate synthetic dataset with authentic and spoofed signals.
    
    Args:
        num_authentic: Number of authentic signals
        num_spoofed: Number of spoofed signals
        fs: Sampling frequency (Hz)
        duration: Duration per signal (seconds)
        prn_range: Range of PRN numbers to use (min, max)
        random_state: Random state for reproducibility
    
    Returns:
        Tuple of (signals, labels, metadata)
    """
    np.random.seed(random_state)
    
    signals = []
    labels = []
    metadata = []
    
    # Generate authentic signals
    print(f"Generating {num_authentic} authentic signals...")
    for i in range(num_authentic):
        prn = np.random.randint(prn_range[0], prn_range[1] + 1)
        cn0 = np.random.uniform(40, 50)  # Typical C/N0 range
        doppler = np.random.uniform(-5000, 5000)  # Doppler range in Hz
        
        signal = generate_synthetic_gps_signal(
            prn=prn, fs=fs, duration=duration,
            cn0=cn0, doppler=doppler, spoofed=False
        )
        
        signals.append(signal)
        labels.append(0)  # Authentic
        metadata.append({
            'prn': prn,
            'cn0': cn0,
            'doppler': doppler,
            'spoofed': False,
            'segment_index': i
        })
    
    # Generate spoofed signals
    print(f"Generating {num_spoofed} spoofed signals...")
    for i in range(num_spoofed):
        prn = np.random.randint(prn_range[0], prn_range[1] + 1)
        cn0 = np.random.uniform(45, 55)  # Higher C/N0 for spoofing
        doppler = np.random.uniform(-5000, 5000)
        
        # Random spoofing parameters
        spoof_params = {
            'power_increase': np.random.uniform(5, 15),
            'secondary_peak': np.random.choice([True, False], p=[0.7, 0.3]),
            'secondary_delay': np.random.uniform(30, 100),
            'secondary_power': np.random.uniform(0.3, 0.7),
        }
        
        signal = generate_synthetic_gps_signal(
            prn=prn, fs=fs, duration=duration,
            cn0=cn0, doppler=doppler, spoofed=True,
            spoof_params=spoof_params
        )
        
        signals.append(signal)
        labels.append(1)  # Spoofed
        metadata.append({
            'prn': prn,
            'cn0': cn0,
            'doppler': doppler,
            'spoofed': True,
            'spoof_params': spoof_params,
            'segment_index': num_authentic + i
        })
    
    print(f"Dataset generation complete: {len(signals)} total signals")
    return signals, labels, metadata
