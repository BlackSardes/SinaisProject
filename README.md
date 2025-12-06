# SinaisProject - GPS Spoofing Detection Pipeline

A comprehensive pipeline for detecting GPS spoofing attacks using signal processing and machine learning techniques.

## Overview

This project implements a complete pipeline for GPS spoofing detection, including:
- Signal preprocessing (filtering, normalization, segmentation)
- Feature extraction (correlation analysis, statistical features)
- Machine learning classification (Random Forest, SVM, MLP)
- Visualization and evaluation tools

## Table of Contents

- [Installation](#installation)
- [Project Structure](#project-structure)
- [Data Setup](#data-setup)
- [Usage](#usage)
- [Pipeline Overview](#pipeline-overview)
- [Contributing](#contributing)
- [License](#license)

## Installation

### Prerequisites

- Python 3.8 or higher
- pip or conda package manager

### Using pip

```bash
# Clone the repository
git clone https://github.com/BlackSardes/SinaisProject.git
cd SinaisProject

# Install dependencies
pip install -r requirements.txt
```

### Using conda

```bash
# Clone the repository
git clone https://github.com/BlackSardes/SinaisProject.git
cd SinaisProject

# Create conda environment
conda env create -f environment.yml
conda activate sinais-gps-spoofing
```

## Project Structure

```
SinaisProject/
├── data/                      # Data directory (not tracked in git)
│   └── README.md             # Instructions for data download
├── notebooks/                 # Jupyter notebooks
│   ├── EDA.ipynb             # Exploratory Data Analysis
│   ├── feature_demo.ipynb    # Feature extraction demo
│   └── training_eval.ipynb   # Model training and evaluation
├── src/                       # Source code
│   ├── preprocessing/        # Signal preprocessing
│   │   ├── signal_io.py      # Signal loading (.mat, .csv)
│   │   ├── normalization.py  # Signal normalization
│   │   ├── filtering.py      # Bandpass and DC removal
│   │   ├── resampling.py     # Signal resampling
│   │   ├── segmentation.py   # Window segmentation
│   │   ├── noise.py          # Outlier removal and smoothing
│   │   └── cn0_estimation.py # C/N0 estimation
│   ├── features/             # Feature extraction
│   │   ├── correlation.py    # Cross-correlation computation
│   │   ├── metrics.py        # FWHM and other metrics
│   │   └── feature_extraction.py # Feature vector building
│   ├── models/               # Machine learning models
│   │   ├── train.py          # Model training
│   │   ├── evaluate.py       # Model evaluation
│   │   └── persistence.py    # Model save/load
│   └── utils/                # Utilities
│       └── plots.py          # Visualization functions
├── docs/                     # Documentation
│   └── DECISIONS.md          # Design decisions and rationale
├── tests/                    # Unit tests
│   ├── test_metrics.py       # Tests for feature metrics
│   └── test_preprocessing.py # Tests for preprocessing
├── scripts/                  # Scripts
│   └── script_run_pipeline.py # End-to-end pipeline script
├── requirements.txt          # Python dependencies
├── environment.yml           # Conda environment file
└── README.md                # This file
```

## Data Setup

### FGI-SpoofRepo Dataset

This pipeline is designed to work with the FGI-SpoofRepo dataset (Finnish Geospatial Research Institute GPS Spoofing Repository).

#### Download Instructions

1. Visit the FGI-SpoofRepo website or repository
2. Download the GPS signal datasets
3. Create a `data/` directory in the project root (if not already exists)
4. Extract the downloaded files into the `data/` directory

Expected data structure:
```
data/
├── scenario1/
│   ├── authentic.bin
│   └── spoofed.bin
├── scenario2/
│   └── ...
└── README.md
```

#### Supported Formats

- **Binary IQ data** (`.bin`, `.dat`): Raw I/Q samples as int16
- **MATLAB files** (`.mat`): Processed signals with metadata
- **CSV files** (`.csv`): Tabulated signal data or extracted features

### Synthetic Data

For testing without real data, the pipeline can generate synthetic GPS-like signals. See `scripts/script_run_pipeline.py` with the `--synthetic` flag.

## Usage

### Quick Start - End-to-End Pipeline

Run the complete pipeline with synthetic data:

```bash
python scripts/script_run_pipeline.py --synthetic
```

Run with real FGI-SpoofRepo data:

```bash
python scripts/script_run_pipeline.py --data-path data/scenario1/
```

### Step-by-Step Usage

#### 1. Preprocessing

```python
from src.preprocessing import load_signal, normalize_signal, bandpass_filter

# Load signal
signal, metadata = load_signal('data/signal.mat')

# Normalize
signal_norm = normalize_signal(signal, method='power')

# Bandpass filter
signal_filt = bandpass_filter(signal_norm, lowcut=1e6, highcut=2e6, fs=5e6)
```

#### 2. Feature Extraction

```python
from src.preprocessing import window_segment
from src.features import build_feature_vector

# Segment signal
windows = window_segment(signal_filt, window_size=5000, overlap=2500)

# Extract features
features_df = build_feature_vector(windows, prn=1, fs=5e6, reference_code=ref_code)
```

#### 3. Model Training

```python
from src.models import train_model, evaluate_model

# Train model
pipeline, metrics = train_model(X_train, y_train, model_name='random_forest', use_smote=False)

# Evaluate
eval_metrics = evaluate_model(pipeline, X_test, y_test)
print(f"Accuracy: {eval_metrics['accuracy']:.3f}")
```

#### 4. Save and Load Model

```python
from src.models import save_model, load_model

# Save
save_model(pipeline, 'models/rf_detector.joblib')

# Load
pipeline = load_model('models/rf_detector.joblib')
```

### Jupyter Notebooks

Explore the pipeline interactively:

```bash
jupyter notebook notebooks/
```

- **EDA.ipynb**: Exploratory data analysis
- **feature_demo.ipynb**: Feature extraction demonstration
- **training_eval.ipynb**: Model training and evaluation

## Pipeline Overview

### 1. Signal Preprocessing

- **Load signals**: Support for .mat, .csv, and binary formats
- **DC removal**: Remove constant offset
- **Normalization**: Power-based normalization
- **Filtering**: Bandpass filters for noise reduction
- **Segmentation**: Window-based processing
- **Smoothing**: Savitzky-Golay and median filters

### 2. Feature Extraction

Key features extracted from GPS signals:

- **Correlation features**:
  - Peak height
  - FWHM (Full Width at Half Maximum)
  - Peak-to-secondary ratio
  - Peak offset
- **Statistical features**:
  - Mean, variance
  - Skewness, kurtosis
  - Signal energy
  - SNR estimates

### 3. Classification Models

- **Random Forest** (primary): Robust, interpretable, handles class imbalance
- **SVM**: Support Vector Machine for non-linear separation
- **MLP**: Multi-Layer Perceptron for deep learning approach

All models use scikit-learn Pipelines with:
- Missing value imputation
- Feature scaling
- Optional SMOTE for class imbalance

### 4. Evaluation

Comprehensive evaluation metrics:
- Accuracy, Precision, Recall, F1-score
- Confusion matrix
- ROC AUC score
- Cross-validation scores
- Feature importance (for tree-based models)

## Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src

# Run specific test file
pytest tests/test_metrics.py -v
```

## Model Performance

Typical performance on FGI-SpoofRepo dataset:

| Model          | Accuracy | F1-Score | Training Time |
|----------------|----------|----------|---------------|
| Random Forest  | ~95%     | ~0.94    | ~5s           |
| SVM            | ~93%     | ~0.92    | ~10s          |
| MLP            | ~92%     | ~0.91    | ~30s          |

*Note: Performance varies by dataset and spoofing scenario.*

## Configuration

Key parameters can be adjusted in the pipeline:

- **Sampling frequency**: Default 5 MHz (GPS L1)
- **Window size**: Default 0.001s (1ms)
- **C/A chip rate**: 1.023 MHz
- **Model parameters**: See `src/models/train.py`

## Documentation

- **[DECISIONS.md](docs/DECISIONS.md)**: Design decisions linked to Sinais e Sistemas concepts
- **Docstrings**: All functions have detailed docstrings
- **Type hints**: Most functions include type annotations

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add docstrings to all functions
- Include type hints where possible
- Write unit tests for new features
- Update documentation as needed

## License

[Add license information]

## Acknowledgments

- FGI (Finnish Geospatial Research Institute) for the SpoofRepo dataset
- Signal processing concepts based on "Sinais e Sistemas" (Oppenheim & Willsky)
- GPS signal processing references from the GPS ICD

## Contact

For questions or issues, please open an issue on GitHub or contact [your-email].

## References

1. Psiaki, M. L., & Humphreys, T. E. (2016). GNSS Spoofing and Detection. Proceedings of the IEEE.
2. FGI-SpoofRepo: GPS Spoofing Dataset
3. Oppenheim, A. V., & Willsky, A. S. Signals and Systems.
