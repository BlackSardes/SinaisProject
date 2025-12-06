"""
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
    """
    Generate synthetic dataset with authentic and spoofed signals.
    
    Args:
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
