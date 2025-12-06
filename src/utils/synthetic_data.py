"""
<<<<<<< HEAD
Synthetic GPS signal generation for testing and development.

Allows pipeline execution without real GPS data.
"""
import numpy as np
from typing import Tuple, Optional
import pandas as pd


def generate_synthetic_gps_signal(
    duration_s: float = 1.0,
    fs: float = 5e6,
    prn: int = 1,
    cn0_db: float = 45.0,
    add_spoofing: bool = False,
    spoofing_delay_chips: float = 0.5,
    spoofing_power_ratio: float = 1.5,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate synthetic GPS L1 C/A signal.
    
    Creates a simplified GPS signal with C/A code, carrier, and noise.
    Optionally adds a spoofing signal with specified delay and power.
    
    Args:
        duration_s: Signal duration in seconds
        fs: Sampling frequency in Hz
        prn: PRN number (1-32)
        cn0_db: Carrier-to-Noise density ratio in dB-Hz
        add_spoofing: Whether to add spoofing signal
        spoofing_delay_chips: Delay of spoofing signal in chips
        spoofing_power_ratio: Power ratio of spoofing to authentic signal
        seed: Random seed for reproducibility
    
    Returns:
        Complex I/Q signal
        
    Example:
        >>> # Generate authentic signal
        >>> signal_auth = generate_synthetic_gps_signal(duration_s=0.5, cn0_db=45)
        >>> 
        >>> # Generate spoofed signal
        >>> signal_spoof = generate_synthetic_gps_signal(
        ...     duration_s=0.5, cn0_db=45, add_spoofing=True,
        ...     spoofing_delay_chips=0.3, spoofing_power_ratio=2.0
        ... )
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate C/A code
    from ..preprocessing.signal_io import generate_ca_code
    ca_code = generate_ca_code(prn)
    
    # Parameters
    ca_chip_rate = 1.023e6  # Hz
    carrier_freq = 0  # Baseband (already mixed down)
    num_samples = int(duration_s * fs)
    
    # Oversample C/A code
    samples_per_chip = int(fs / ca_chip_rate)
    code_oversampled = np.repeat(ca_code, samples_per_chip)
    
    # Repeat code to fill duration
    ca_period_samples = len(code_oversampled)
    repeats = int(np.ceil(num_samples / ca_period_samples))
    code_full = np.tile(code_oversampled, repeats)[:num_samples]
    
    # Generate carrier (complex exponential)
    t = np.arange(num_samples) / fs
    carrier = np.exp(1j * 2 * np.pi * carrier_freq * t)
    
    # Modulate code onto carrier
    signal_carrier = code_full * carrier
    
    # Calculate signal and noise power from C/N0
    # C/N0 = C / (N0 * BW) where BW = fs for baseband
    cn0_linear = 10 ** (cn0_db / 10)
    noise_power_density = 1.0  # Reference
    signal_power = cn0_linear * noise_power_density * fs
    
    # Scale signal
    signal_carrier = signal_carrier * np.sqrt(signal_power / 2)  # /2 for I and Q
    
    # Add noise
    noise_power = noise_power_density * fs
    noise_i = np.random.normal(0, np.sqrt(noise_power / 2), num_samples)
    noise_q = np.random.normal(0, np.sqrt(noise_power / 2), num_samples)
    noise = noise_i + 1j * noise_q
    
    signal = signal_carrier + noise
    
    # Add spoofing signal if requested
    if add_spoofing:
        # Delay spoofing code
        delay_samples = int(spoofing_delay_chips * samples_per_chip)
        code_spoofed = np.roll(code_full, delay_samples)
        
        # Modulate spoofed code
        signal_spoof = code_spoofed * carrier
        
        # Scale by power ratio
        spoof_power = signal_power * spoofing_power_ratio
        signal_spoof = signal_spoof * np.sqrt(spoof_power / 2)
        
        # Add to signal
        signal = signal + signal_spoof
    
    return signal


def generate_synthetic_dataset(
    n_authentic: int = 100,
    n_spoofed: int = 100,
    duration_s: float = 0.5,
    fs: float = 5e6,
    prn: int = 1,
    cn0_authentic: Tuple[float, float] = (40, 50),
    cn0_spoofed: Tuple[float, float] = (45, 60),
    spoofing_delay_range: Tuple[float, float] = (0.2, 1.0),
    spoofing_power_range: Tuple[float, float] = (1.2, 3.0),
    seed: Optional[int] = None
) -> Tuple[list, np.ndarray]:
=======
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
>>>>>>> main
    """
    Generate synthetic dataset with authentic and spoofed signals.
    
    Args:
<<<<<<< HEAD
        n_authentic: Number of authentic signal segments
        n_spoofed: Number of spoofed signal segments
        duration_s: Duration of each segment
        fs: Sampling frequency
        prn: PRN number
        cn0_authentic: (min, max) C/N0 range for authentic signals
        cn0_spoofed: (min, max) C/N0 range for spoofed signals
        spoofing_delay_range: (min, max) delay in chips for spoofed signals
        spoofing_power_range: (min, max) power ratio for spoofed signals
        seed: Random seed
    
    Returns:
        Tuple of (signal_list, label_array)
        where label_array: 0=authentic, 1=spoofed
        
    Example:
        >>> signals, labels = generate_synthetic_dataset(
        ...     n_authentic=50, n_spoofed=50, duration_s=0.5
        ... )
        >>> print(f"Generated {len(signals)} signals")
        >>> print(f"Authentic: {np.sum(labels == 0)}, Spoofed: {np.sum(labels == 1)}")
    """
    if seed is not None:
        np.random.seed(seed)
    
    signals = []
    labels = []
    
    # Generate authentic signals
    for i in range(n_authentic):
        cn0 = np.random.uniform(*cn0_authentic)
        signal = generate_synthetic_gps_signal(
            duration_s=duration_s,
            fs=fs,
            prn=prn,
            cn0_db=cn0,
            add_spoofing=False,
            seed=None  # Use global random state
        )
        signals.append(signal)
        labels.append(0)
    
    # Generate spoofed signals
    for i in range(n_spoofed):
        cn0 = np.random.uniform(*cn0_spoofed)
        delay = np.random.uniform(*spoofing_delay_range)
        power_ratio = np.random.uniform(*spoofing_power_range)
        
        signal = generate_synthetic_gps_signal(
            duration_s=duration_s,
            fs=fs,
            prn=prn,
            cn0_db=cn0,
            add_spoofing=True,
            spoofing_delay_chips=delay,
            spoofing_power_ratio=power_ratio,
            seed=None
        )
        signals.append(signal)
        labels.append(1)
    
    labels = np.array(labels)
    
    return signals, labels


def create_synthetic_features_dataframe(
    n_authentic: int = 100,
    n_spoofed: int = 100,
    fs: float = 5e6,
    prn: int = 1,
    seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Generate synthetic dataset and extract features in one step.
    
    Convenience function that generates signals and immediately
    extracts features into a DataFrame ready for training.
    
    Args:
        n_authentic: Number of authentic samples
        n_spoofed: Number of spoofed samples
        fs: Sampling frequency
        prn: PRN number
        seed: Random seed
    
    Returns:
        DataFrame with features and labels
        
    Example:
        >>> df = create_synthetic_features_dataframe(n_authentic=100, n_spoofed=100)
        >>> X = df.drop('label', axis=1)
        >>> y = df['label']
        >>> # Ready for training!
    """
    from ..features.feature_pipeline import build_feature_vector
    
    # Generate signals
    signals, labels = generate_synthetic_dataset(
        n_authentic=n_authentic,
        n_spoofed=n_spoofed,
        fs=fs,
        prn=prn,
        seed=seed
    )
    
    # Extract features
    features_df = build_feature_vector(signals, fs=fs, prn=prn)
    
    # Add labels
    features_df['label'] = labels
    
    return features_df
=======
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
    
    # GPS signal constants
    GPS_DOPPLER_MIN = -5000  # Hz
    GPS_DOPPLER_MAX = 5000   # Hz
    
    # Generate authentic signals
    print(f"Generating {num_authentic} authentic signals...")
    for i in range(num_authentic):
        prn = np.random.randint(prn_range[0], prn_range[1] + 1)
        cn0 = np.random.uniform(40, 50)  # Typical C/N0 range
        doppler = np.random.uniform(GPS_DOPPLER_MIN, GPS_DOPPLER_MAX)
        
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
        doppler = np.random.uniform(GPS_DOPPLER_MIN, GPS_DOPPLER_MAX)
        
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
>>>>>>> main
