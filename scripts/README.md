# Scripts

This directory contains executable scripts for running the GPS Spoofing Detection pipeline.

## Main Pipeline Script

### `run_pipeline.py`

Complete end-to-end pipeline that:
1. Loads or generates GPS signal data
2. Preprocesses signals
3. Extracts features
4. Trains multiple ML models
5. Evaluates performance
6. Generates visualizations and reports

#### Usage with Synthetic Data (Testing)

```bash
python scripts/run_pipeline.py --synthetic --output-dir results
```

This will:
- Generate 100 synthetic signal segments
- Extract features from each segment
- Train Random Forest, SVM, and MLP models
- Save results, models, and visualizations to `results/`

#### Usage with Real Data

```bash
python scripts/run_pipeline.py \
    --data-dir data/raw \
    --output-dir results \
    --fs 5e6 \
    --prn 1 \
    --segment-duration 0.5 \
    --spoof-start-time 17.0
```

#### Arguments

- `--data-dir`: Directory containing signal files (default: `data/raw`)
- `--output-dir`: Output directory for results (default: `results`)
- `--synthetic`: Use synthetic signals instead of real data
- `--fs`: Sampling frequency in Hz (default: 5e6)
- `--prn`: PRN satellite number (default: 1)
- `--segment-duration`: Segment duration in seconds (default: 0.5)
- `--spoof-start-time`: Time when spoofing starts (default: 17.0)
- `--models`: Models to train (default: all)
- `--cv-folds`: Cross-validation folds (default: 5)
- `--random-state`: Random seed (default: 42)
- `--verbose`: Print detailed progress

#### Example: Train Only Random Forest

```bash
python scripts/run_pipeline.py \
    --synthetic \
    --models random_forest \
    --cv-folds 10 \
    --verbose
```

#### Output Structure

After running, the output directory contains:

```
results/
├── features.csv              # Extracted features
├── models/                   # Trained models
│   ├── random_forest_class_weight.pkl
│   ├── svm_class_weight.pkl
│   └── mlp_none.pkl
├── reports/                  # Evaluation reports
│   ├── model_comparison.csv
│   ├── random_forest_class_weight_report.txt
│   └── ...
└── figures/                  # Visualizations
    ├── feature_distributions.png
    ├── cn0_over_time.png
    ├── model_comparison.png
    ├── random_forest_class_weight_confusion_matrix.png
    ├── random_forest_class_weight_roc_curve.png
    └── ...
```

## Creating Custom Scripts

To create your own pipeline scripts:

```python
#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.signal_io import generate_synthetic_signal
from src.preprocessing.pipeline import preprocess_signal
from src.features.pipeline import extract_features_from_segment
from src.models.training import train_model

# Your custom pipeline logic here
```

## Tips

1. **Start with synthetic data** to verify the pipeline works
2. **Use `--verbose`** to see detailed progress
3. **Adjust `--spoof-start-time`** based on your dataset
4. **Save intermediate results** by modifying the script
5. **Run on smaller datasets first** for quick iteration

## Troubleshooting

**Error: No data files found**
- Check that data files are in the specified directory
- Supported formats: `.bin`, `.dat`, `.mat`, `.csv`

**Error: Not enough memory**
- Reduce segment duration
- Process files individually
- Use a machine with more RAM

**Error: Model training fails**
- Check class distribution (need both authentic and spoofed samples)
- Try different balance methods
- Reduce number of features

For more help, see the main README.md or docs/DECISIONS.md
