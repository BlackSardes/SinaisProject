# SinaisProject - GPS Spoofing Detection

A robust machine learning pipeline for detecting GPS spoofing attacks using signal processing and classification techniques. This project was developed for the Signals and Systems (ES413) course.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

GPS spoofing is a security threat where false GPS signals are transmitted to deceive receivers. This project implements a complete pipeline for detecting such attacks by analyzing:

- **Correlation peak distortions**: Spoofing creates multiple peaks or asymmetric patterns
- **Power anomalies**: Spoofing signals often have higher power (C/N0) than authentic signals
- **Statistical features**: Distribution changes in signal properties

The pipeline uses machine learning models (Random Forest, SVM, Neural Networks) trained on features extracted from GPS signal correlation functions.

## ✨ Features

### Signal Processing
- Multi-format signal loading (binary, MATLAB, CSV, synthetic)
- GPS C/A code generation for all PRN satellites (1-32)
- Comprehensive preprocessing pipeline:
  - DC offset removal
  - Frequency correction (Doppler/IF)
  - Interference mitigation (pulse blanking, frequency domain)
  - Bandpass and notch filtering
  - Signal normalization

### Feature Extraction
- **Correlation-based features**:
  - Peak-to-secondary ratio
  - Full Width at Half Maximum (FWHM)
  - Peak asymmetry
  - Skewness and kurtosis
  - Energy distribution
  
- **Power metrics**:
  - C/N0 estimation
  - SNR calculation
  - Noise floor analysis
  
- **Statistical features**:
  - Distribution moments
  - Spectral characteristics
  - Temporal patterns

### Machine Learning
- Multiple model support (Random Forest, SVM, MLP)
- Class imbalance handling (SMOTE, class weights)
- Stratified cross-validation
- Comprehensive evaluation metrics
- Model persistence (save/load)

### Visualization
- Correlation profile plots
- Feature distribution analysis
- Confusion matrices
- ROC curves
- Feature importance rankings
- C/N0 temporal evolution

## 🛠️ Installation

### Prerequisites
- Python 3.9 or higher
- pip or conda package manager

### Using pip

```bash
# Clone the repository
git clone https://github.com/BlackSardes/SinaisProject.git
cd SinaisProject

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

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
conda activate sinaisproject
```

## 📁 Project Structure

```
SinaisProject/
├── data/                          # Data directory (gitignored)
│   ├── raw/                       # Raw signal files
│   └── processed/                 # Processed features
├── notebooks/                     # Jupyter notebooks
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_feature_extraction_demo.ipynb
│   └── 03_model_training_evaluation.ipynb
├── src/                           # Source code
│   ├── preprocessing/             # Signal preprocessing
│   │   ├── signal_io.py          # I/O operations
│   │   ├── signal_processing.py  # Processing functions
│   │   ├── prn_codes.py          # PRN code generation
│   │   └── pipeline.py           # Complete pipeline
│   ├── features/                  # Feature extraction
│   │   ├── correlation.py        # Correlation features
│   │   ├── statistical.py        # Statistical features
│   │   └── pipeline.py           # Feature pipeline
│   ├── models/                    # Machine learning models
│   │   ├── training.py           # Training functions
│   │   └── evaluation.py         # Evaluation metrics
│   └── utils/                     # Utilities
│       └── plots.py              # Visualization functions
├── tests/                         # Unit tests
├── scripts/                       # Executable scripts
│   └── run_pipeline.py           # End-to-end pipeline
├── docs/                          # Documentation
│   └── DECISIONS.md              # Technical decisions
├── requirements.txt               # Python dependencies
├── environment.yml                # Conda environment
└── README.md                      # This file
```

## 🚀 Quick Start

### 1. Synthetic Signal Example

```python
from src.preprocessing.signal_io import generate_synthetic_signal
from src.preprocessing.pipeline import preprocess_signal
from src.features.pipeline import extract_features_from_segment

# Generate synthetic GPS signal
signal = generate_synthetic_signal(
    num_samples=50000,
    fs=5e6,
    snr_db=10.0,
    prn=1,
    add_spoofing=True
)

# Preprocess
signal_processed = preprocess_signal(signal, fs=5e6)

# Extract features
features = extract_features_from_segment(
    signal_processed,
    fs=5e6,
    prn=1
)

print(f"Extracted {len(features)} features")
```

### 2. Process Real Data File

```python
from src.features.pipeline import extract_features_from_file

# Define labeling function (TEXBAT dataset example)
def label_func(time_s):
    return 1 if time_s >= 17.0 else 0  # Spoofing starts at 17s

# Extract features from entire file
features_df = extract_features_from_file(
    file_path='data/raw/signal.bin',
    fs=5e6,
    prn=1,
    segment_duration=0.5,
    label_func=label_func,
    verbose=True
)

print(features_df.head())
```

### 3. Train and Evaluate Model

```python
from src.models.training import train_model, create_train_test_split
from src.models.evaluation import evaluate_model
from src.utils.plots import plot_confusion_matrix, plot_roc_curve

# Prepare data
X = features_df.drop(columns=['label', 'segment_start_time']).values
y = features_df['label'].values

# Split data
X_train, X_test, y_train, y_test = create_train_test_split(X, y)

# Train model
model, cv_results = train_model(
    X_train, y_train,
    model_name='random_forest',
    balance_method='class_weight'
)

# Evaluate
results = evaluate_model(model, X_test, y_test)

# Visualize
plot_confusion_matrix(results['confusion_matrix'])
```

## 📖 Usage

### Complete Pipeline Script

Run the end-to-end pipeline:

```bash
python scripts/run_pipeline.py --data-dir data/raw --output-dir results
```

### Jupyter Notebooks

Explore the interactive notebooks:

```bash
jupyter notebook notebooks/
```

1. **01_exploratory_analysis.ipynb**: Dataset exploration and visualization
2. **02_feature_extraction_demo.ipynb**: Feature extraction walkthrough
3. **03_model_training_evaluation.ipynb**: Model training and comparison

### Adding New Datasets

1. Place signal files in `data/raw/`
2. Supported formats:
   - Binary I/Q files (`.bin`, `.dat`): TEXBAT/FGI format
   - MATLAB files (`.mat`): Must contain 'signal' or 'I'/'Q' variables
   - CSV files (`.csv`): Two columns (I, Q)

3. Define labeling function based on your dataset:

```python
def custom_label_func(time_s):
    # Return 1 for spoofed segments, 0 for authentic
    if time_s >= SPOOF_START_TIME:
        return 1
    return 0
```

## 📚 Documentation

Detailed documentation is available in the `docs/` directory:

- **[DECISIONS.md](docs/DECISIONS.md)**: Technical decisions and justifications
  - Why specific features were chosen
  - Signal processing techniques explained
  - Algorithm selection rationale
  - Connection to Signal and Systems theory

### Key Concepts

#### Signal Processing Pipeline
1. **DC Removal**: Removes hardware-induced offset
2. **Frequency Correction**: Compensates for Doppler and IF
3. **Interference Mitigation**: Suppresses RFI and impulsive noise
4. **Normalization**: Standardizes power levels

#### Feature Extraction
- **Peak-to-Secondary Ratio**: Measures correlation peak purity (decreases with spoofing)
- **Asymmetry**: Quantifies peak imbalance (increases with overlapping signals)
- **C/N0**: Carrier-to-noise density ratio (increases in power attacks)

#### Models
- **Random Forest**: Best overall performance, handles non-linear relationships
- **SVM**: Good for high-dimensional feature spaces
- **MLP**: Neural network for complex pattern learning

## 🧪 Testing

Run unit tests:

```bash
python -m pytest tests/
```

Run specific test module:

```bash
python -m pytest tests/test_preprocessing.py -v
```

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is developed for academic purposes as part of the ES413 course.

## 🙏 Acknowledgments

- **TEXBAT Dataset**: GPS spoofing test dataset
- **IS-GPS-200**: GPS Interface Specification for C/A code generation
- **Course**: Sinais e Sistemas (ES413)

## 📧 Contact

For questions or issues, please open an issue on GitHub.

---

**Note**: This project requires GPS signal data to function. Synthetic signals can be generated for testing, but real spoofing detection requires authentic datasets like TEXBAT or FGI.
