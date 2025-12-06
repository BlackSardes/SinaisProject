
# Technical Decisions and Justifications

This document explains the technical decisions made in the GPS Spoofing Detection project, with emphasis on the connection to Signal Processing and Machine Learning theory.

## Table of Contents

1. [Preprocessing Module](#preprocessing-module)
2. [Feature Extraction](#feature-extraction)
3. [Machine Learning Models](#machine-learning-models)
4. [Evaluation Strategy](#evaluation-strategy)
5. [References](#references)

---

## Preprocessing Module

### 1. DC Offset Removal

**Decision**: Always remove DC offset before processing.

**Justification**:
- **Signal Theory**: DC offset is a constant bias in the signal that doesn't carry information
- **Hardware Reality**: ADC and RF front-end imperfections introduce DC
- **Impact on Correlation**: DC offset creates a spurious peak at zero lag in autocorrelation
- **Implementation**: Simple subtraction of signal mean: `signal - mean(signal)`

**Mathematical Basis**:
```
x_clean(t) = x(t) - E[x(t)]
```

### 2. Frequency Correction (Doppler/IF Removal)

**Decision**: Apply frequency correction through complex multiplication with local oscillator.

**Justification**:
- **Frequency Shifting Property**: Multiplication by exp(-j2πft) shifts spectrum by -f Hz
- **GPS Reality**: Signals arrive with:
  - Intermediate Frequency (IF): From RF front-end downconversion
  - Doppler shift: Due to satellite motion (~±5 kHz for GPS L1)
- **Correlation Requirement**: For optimal correlation, received and local codes must be frequency-aligned

**Mathematical Basis**:
```
x_baseband(t) = x_IF(t) · exp(-j2π·f_correction·t)

where f_correction = f_IF + f_doppler
```

**Signal & Systems Connection**:
- Modulation property of Fourier Transform
- Preserves signal information while shifting frequency

### 3. Notch Filter for RFI Suppression

**Decision**: Use IIR notch filter (Butterworth-based) for narrowband interference.

**Justification**:
- **Interference Type**: Narrowband RFI from communication systems, jammers
- **Filter Design**:
  - IIR: More efficient than FIR for narrow notches
  - Quality factor Q: Controls notch width (BW ≈ f₀/Q)
  - Default Q=30: Narrow enough to preserve GPS bandwidth (~2 MHz)

**Frequency Domain Analysis**:
```
H(f) has deep notch at f₀ with bandwidth BW = f₀/Q
```

**Trade-offs**:
- **Pros**: Preserves GPS signal, removes interference
- **Cons**: Phase distortion near notch (acceptable for magnitude-based correlation)

### 4. Pulse Blanking

**Decision**: Zero out samples exceeding threshold (default: 4σ).

**Justification**:
- **Interference Type**: Impulsive noise (radar pulses, lightning)
- **Time-Domain Approach**: More effective than frequency-domain for transient events
- **Threshold Selection**: 4σ captures 99.99% of Gaussian noise, flags outliers
- **Non-linear**: Unlike filtering, completely removes corrupted samples

**Statistical Basis**:
```
x_blanked(t) = {
  x(t),  if |x(t)| < μ + 4σ
  0,     otherwise
}
```

### 5. Power Normalization

**Decision**: Normalize to unit power: E[|x|²] = 1

**Justification**:
- **Hardware Variability**: Different receivers have different gains
- **Spoofing Independence**: Power level shouldn't be the only discriminant
- **C/N0 Calculation**: Enables meaningful power comparisons
- **ML Benefit**: Prevents features from being dominated by amplitude differences

**Mathematical Basis**:
```
x_normalized = x / √(E[|x|²])
```

---

## Feature Extraction

### Correlation-Based Features

#### 1. Peak-to-Secondary Ratio (P/S Ratio)

**Decision**: Primary spoofing detection feature.

**Justification**:
- **Autocorrelation Property**: GPS C/A codes have near-ideal autocorrelation
  - Main peak: 1023 (code length)
  - Max sidelobe: ±65
  - P/S ratio: ~15.7 (23.9 dB)
- **Spoofing Effect**: Multiple signal sources create additional peaks
- **Theoretical Basis**: Sum of two correlation functions creates interference pattern

**Mathematical Model**:
```
R_spoofed(τ) = R_authentic(τ) + α·R_spoofer(τ - δ)

where:
- α: power ratio
- δ: delay between signals
```

**Detection Principle**: P/S ratio decreases when secondary peaks emerge.

**Expected Values**:
- Authentic: >15 (often >20)
- Spoofed: <10 (depends on delay δ)

#### 2. Full Width at Half Maximum (FWHM)

**Decision**: Measure correlation peak width.

**Justification**:
- **Theoretical Width**: For GPS C/A code at fs, FWHM ≈ 1-2 chips
- **Broadening Causes**:
  - Multipath: Creates multiple delayed replicas
  - Dual-signal spoofing: Overlapping peaks widen effective width
  - Frequency mismatch: Sinc function widening
- **Robustness**: Less sensitive to absolute power than peak value

**Measurement**:
```
FWHM = max(τ_high) - min(τ_low)
where R(τ_high) = R(τ_low) = 0.5·R_peak
```

#### 3. Peak Asymmetry

**Decision**: Quantify left-right imbalance around peak.

**Justification**:
- **Ideal Case**: GPS C/A autocorrelation is symmetric (triangular main lobe)
- **Asymmetry Source**: Overlapping authentic and spoofer signals with small delay
- **Sensitivity**: Very sensitive to delay differences <1 chip

**Calculation**:
```
Asymmetry = (Area_right - Area_left) / (Area_right + Area_left)

where areas computed ±1 chip from peak
```

**Range**: [-1, +1], ideal = 0

#### 4. Skewness and Kurtosis

**Decision**: Higher-order statistical moments of correlation peak region.

**Justification**:
- **Normal Case**: GPS correlation has specific shape (triangular + noise)
- **Skewness**: Detects asymmetry in distribution (third moment)
- **Kurtosis**: Detects tail weight / peakedness (fourth moment)
- **ML Value**: Captures subtle shape changes missed by simpler metrics

**Statistical Basis**:
```
Skewness = E[(X - μ)³] / σ³
Kurtosis = E[(X - μ)⁴] / σ⁴ - 3
```

### Power-Based Features

#### 5. C/N0 (Carrier-to-Noise Density Ratio)

**Decision**: Estimate C/N0 from correlation results.

**Justification**:
- **GPS Standard Metric**: Industry standard for signal quality
- **Physical Meaning**: Carrier power divided by noise power density
- **Spoofing Signature**: Most attacks use higher power to dominate receiver
- **Typical Values**:
  - Clear sky: 45-50 dB-Hz
  - Urban: 30-40 dB-Hz
  - Spoofing: Often >50 dB-Hz (power attack)

**Estimation Method** (Narrowband-Wideband Power Ratio):
```
C/N0 = 10·log₁₀(P_carrier / N₀)

where:
P_carrier ≈ (Peak_corr)² / N_samples
N₀ ≈ (Total_power - P_carrier) / BW
```

**Why It Works for Detection**:
- Spoofing signal must be stronger than authentic to capture receiver
- Sudden C/N0 increase is suspicious (satellites don't suddenly get closer)

#### 6. Noise Floor Estimation

**Decision**: Estimate noise from correlation function excluding peak region.

**Justification**:
- **Robust Estimator**: Uses median of non-peak regions
- **Complementary to C/N0**: High C/N0 + high noise suggests jamming
- **Baseline for Threshold**: Peak detection threshold = noise_floor + k·σ

**Implementation**:
```
Noise_floor = median(R(τ)) for |τ - τ_peak| > 2 chips
```

### Statistical Features

#### 7. Spectral Features

**Decision**: Include frequency-domain characteristics.

**Justification**:
- **Frequency Anomalies**: Spoofing signal may have different Doppler
- **Spectral Flatness**: Distinguishes tone-like (signal) from noise-like
- **Bandwidth**: Changes if spoofing signal is not perfectly synchronized

**Key Metrics**:
- **Spectral Centroid**: Center of mass in frequency (detects Doppler errors)
- **Spectral Spread**: Frequency dispersion
- **Bandwidth (90%)**: Frequency range containing 90% of energy

#### 8. Temporal Features

**Decision**: Include time-domain statistics.

**Justification**:
- **Complementary**: Different view than correlation features
- **Noise Characteristics**: Distribution changes under interference
- **Phase Information**: Phase discontinuities indicate frequency jumps

**Selected Features**:
- Signal magnitude moments (mean, std, skewness, kurtosis)
- Phase statistics
- Zero-crossing rate
- Autocorrelation at fixed lags

---

## Machine Learning Models

### Model Selection

#### 1. Random Forest (Primary Choice)

**Decision**: Use Random Forest as primary classifier.

**Justification**:
- **Non-linear**: GPS spoofing has non-linear decision boundaries
- **Feature Interactions**: Automatically captures interactions (e.g., high C/N0 + low P/S ratio)
- **Robust**: Less sensitive to outliers than single decision tree
- **Interpretable**: Provides feature importance
- **No Scaling Required**: Tree-based, doesn't require feature normalization

**Hyperparameters**:
```python
{
    'n_estimators': 100,      # 100 trees balance accuracy/speed
    'max_features': 'sqrt',   # Randomization for diversity
    'class_weight': 'balanced' # Handle imbalanced classes
}
```

**Why It Works**:
- Each tree learns different aspect of spoofing signature
- Ensemble reduces variance
- Voting provides confidence measure

#### 2. Support Vector Machine (SVM)

**Decision**: Use RBF kernel SVM as alternative.

**Justification**:
- **High-Dimensional**: Effective in high-dimensional feature spaces
- **Kernel Trick**: RBF kernel handles non-linear separation
- **Margin Maximization**: Finds optimal separating hyperplane
- **Theoretical Foundation**: Strong statistical learning theory basis

**Hyperparameters**:
```python
{
    'C': 1.0,                  # Regularization (default works well)
    'kernel': 'rbf',           # Radial Basis Function for non-linearity
    'gamma': 'scale',          # Adaptive to feature scale
    'class_weight': 'balanced' # Imbalance handling
}
```

**Trade-off**:
- **Pros**: Strong theoretical guarantees, effective in high dimensions
- **Cons**: Slower training than RF, requires feature scaling

#### 3. Multi-Layer Perceptron (Neural Network)

**Decision**: Include MLP for complex pattern learning.

**Justification**:
- **Universal Approximator**: Can learn any continuous function
- **Feature Learning**: Automatically combines low-level features
- **Adaptive**: Architecture can be tuned for specific datasets

**Architecture**:
```python
{
    'hidden_layers': (100, 50),  # Two hidden layers
    'activation': 'relu',         # Non-linearity
    'solver': 'adam',             # Adaptive learning rate
    'early_stopping': True        # Prevent overfitting
}
```

**Why Two Layers**:
- First layer: Learns feature combinations
- Second layer: Learns higher-level patterns
- Not too deep: Prevents overfitting on small datasets

### Class Imbalance Handling

**Problem**: Spoofing events are rare (typically <20% of data).

**Solutions Implemented**:

#### 1. Class Weights (Preferred)

**Decision**: Use `class_weight='balanced'` in models.

**Justification**:
- **Algorithmic**: Adjusts loss function to penalize minority class errors more
- **No Data Duplication**: Doesn't artificially increase dataset size
- **Fast**: No overhead compared to vanilla training

**Mathematical Effect**:
```
Weight_class = N_total / (N_classes · N_class)

Loss_weighted = Σ weight_i · loss_i
```

#### 2. SMOTE (Synthetic Minority Over-sampling Technique)

**Decision**: Offer SMOTE as alternative option.

**Justification**:
- **Data Augmentation**: Creates synthetic minority class samples
- **Mechanism**: Interpolates between existing minority samples
- **Benefit**: Can improve recall (detecting actual spoofing)

**How It Works**:
```
For each minority sample x:
  1. Find K nearest minority neighbors
  2. Randomly select one neighbor x_nn
  3. Create synthetic: x_new = x + λ·(x_nn - x)
     where λ ∈ [0, 1]
```

**Trade-off**:
- **Pros**: Often improves minority class recall
- **Cons**: Can introduce noise, increases training time

### Cross-Validation Strategy

**Decision**: Stratified K-Fold (K=5) cross-validation.

**Justification**:
- **Stratified**: Maintains class distribution in each fold
- **K=5**: Standard choice, balance between variance and computational cost
- **Why Not Leave-One-Out**: Too expensive, high variance
- **Why Not Simple Split**: Doesn't utilize all data for validation

**Process**:
```
For k = 1 to 5:
  1. Hold out fold k as validation
  2. Train on remaining 4 folds
  3. Evaluate on fold k
  4. Record metrics

Final score = mean(fold_scores)
```

---

## Evaluation Strategy

### Metrics Selection

#### Primary Metrics

1. **F1-Score** (Harmonic mean of precision and recall)
   - **Why**: Balances false positives and false negatives
   - **Importance**: Single metric for imbalanced classes

2. **Recall (Sensitivity)**
   - **Definition**: TP / (TP + FN)
   - **Meaning**: Percentage of spoofing attacks detected
   - **Critical**: Missing spoofing is dangerous

3. **Precision**
   - **Definition**: TP / (TP + FP)
   - **Meaning**: Percentage of alerts that are real
   - **Important**: Too many false alarms reduce trust

4. **Specificity (True Negative Rate)**
   - **Definition**: TN / (TN + FP)
   - **Meaning**: Percentage of authentic signals correctly identified
   - **Use**: Ensures normal operations aren't disrupted

#### Secondary Metrics

5. **ROC AUC** (Area Under ROC Curve)
   - **Threshold-Independent**: Evaluates classifier across all thresholds
   - **Range**: [0, 1], random = 0.5, perfect = 1.0
   - **Use**: Model comparison

6. **False Alarm Rate**
   - **Definition**: FP / (FP + TN)
   - **Operational Meaning**: How often genuine signals trigger alerts
   - **Target**: <1% for operational systems

### Confusion Matrix Interpretation

For GPS Spoofing Detection:

```
                 Predicted
                Auth  Spoof
Actual  Auth  |  TN  |  FP  |
        Spoof |  FN  |  TP  |
```

**Implications**:
- **TN (True Negative)**: Authentic signal correctly identified → Good
- **FP (False Positive)**: False alarm → Nuisance, but safe
- **FN (False Negative)**: Missed spoofing → **DANGEROUS**
- **TP (True Positive)**: Spoofing detected → Critical success

**Priority**: Minimize FN (maximize recall), then minimize FP (maximize precision).

### Model Selection Criteria

**Primary**: F1-Score (balance)
**Tie-breaker**: Recall (safety-critical)
**Constraint**: Precision >80% (limit false alarms)

---

## Pipeline Design Philosophy

### Modularity

**Decision**: Separate preprocessing, feature extraction, and modeling.

**Benefits**:
- **Testability**: Each module tested independently
- **Flexibility**: Easy to swap components
- **Reusability**: Preprocessing useful for other GPS applications
- **Debugging**: Isolate problems to specific stage

### Reproducibility

**Decisions**:
1. **Fixed Random Seeds**: `random_state=42` everywhere
2. **Stratified Splits**: Ensure consistent class distribution
3. **Pipeline Configs**: Save preprocessing/feature parameters
4. **Model Serialization**: Save trained models with metadata

**Implementation**:
```python
# Example
config = {
    'random_state': 42,
    'preprocessing': {...},
    'feature_extraction': {...},
    'model': {...}
}
joblib.dump({'model': model, 'config': config}, 'model.pkl')
```

### Scalability

**Considerations**:
1. **Streaming Processing**: Process files in segments (not all in memory)
2. **Parallel Processing**: Use `n_jobs=-1` where possible
3. **Efficient Correlation**: FFT-based (O(N log N)) not direct (O(N²))
4. **Feature Selection**: Reduce dimensionality if needed

---

## References

### GPS Signal Processing
1. IS-GPS-200: GPS Interface Specification
2. Kaplan & Hegarty: "Understanding GPS/GNSS: Principles and Applications"
3. Borre et al.: "A Software-Defined GPS and Galileo Receiver"

### Spoofing Detection
4. Psiaki & Humphreys: "GNSS Spoofing and Detection" (IEEE)
5. TEXBAT Dataset: https://radionavlab.ae.utexas.edu/

### Signal Processing
6. Oppenheim & Schafer: "Discrete-Time Signal Processing"
7. Proakis & Manolakis: "Digital Signal Processing"

### Machine Learning
8. Hastie et al.: "The Elements of Statistical Learning"
9. Scikit-learn Documentation: https://scikit-learn.org/

---

## Future Improvements

### Signal Processing
- Adaptive interference mitigation
- Multi-correlator tracking
- Carrier phase analysis

### Features
- Multi-PRN features (cross-channel correlation)
- Temporal change detection (CUSUM, change-point)
- Doppler rate features

### Models
- Deep learning (CNN for correlation profiles)
- Online learning (adapt to new spoofing techniques)
- Ensemble methods (combine multiple approaches)

### System
- Real-time processing
- Hardware acceleration (GPU)
- Integration with receiver firmware

---

*This document will be updated as the project evolves and new techniques are incorporated.*
