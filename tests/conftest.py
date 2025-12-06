"""
Pytest configuration and fixtures.
"""
import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_signal():
    """Generate a simple synthetic GPS-like signal."""
    from src.utils.synthetic_data import generate_synthetic_gps_signal
    return generate_synthetic_gps_signal(
        duration_s=0.1,
        fs=5e6,
        prn=1,
        cn0_db=45,
        seed=42
    )


@pytest.fixture
def sample_authentic_signal():
    """Generate authentic GPS signal."""
    from src.utils.synthetic_data import generate_synthetic_gps_signal
    return generate_synthetic_gps_signal(
        duration_s=0.1,
        fs=5e6,
        prn=1,
        cn0_db=45,
        add_spoofing=False,
        seed=42
    )


@pytest.fixture
def sample_spoofed_signal():
    """Generate spoofed GPS signal."""
    from src.utils.synthetic_data import generate_synthetic_gps_signal
    return generate_synthetic_gps_signal(
        duration_s=0.1,
        fs=5e6,
        prn=1,
        cn0_db=50,
        add_spoofing=True,
        spoofing_delay_chips=0.5,
        spoofing_power_ratio=2.0,
        seed=42
    )


@pytest.fixture
def sample_dataset():
    """Generate small synthetic dataset."""
    from src.utils.synthetic_data import generate_synthetic_dataset
    return generate_synthetic_dataset(
        n_authentic=10,
        n_spoofed=10,
        duration_s=0.1,
        fs=5e6,
        seed=42
    )
