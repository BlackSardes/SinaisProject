# Design Decisions and Technical Rationale

## Overview

This document describes the design decisions made in the GPS spoofing detection pipeline and their connection to signal processing concepts from "Sinais e Sistemas" (Signals and Systems).

## 1. Signal Preprocessing

### 1.1 DC Removal
**Decision**: Remove DC component from signals before processing.

**Rationale**: DC (Direct Current) components represent constant offsets in signals that don't carry useful information for GPS signal analysis. Removing DC helps normalize signals and improves correlation performance.

**Link to Sinais e Sistemas**: DC corresponds to the zero-frequency component in the Fourier transform. Removing it is equivalent to high-pass filtering with a very low cutoff frequency.

### 1.2 Power Normalization
**Decision**: Normalize signals by their RMS (Root Mean Square) power.

**Rationale**: GPS signals can have varying amplitudes due to distance, atmospheric conditions, and receiver characteristics. Normalizing by power ensures consistent feature extraction regardless of signal amplitude.

**Link to Sinais e Sistemas**: Signal power is defined as the mean of the squared signal magnitude: P = E[|x(t)|²]. RMS normalization divides by √P to achieve unit power.

### 1.3 Bandpass Filtering
**Decision**: Implement Butterworth bandpass filters for frequency-selective filtering.

**Rationale**: GPS L1 C/A signals occupy a specific frequency band around 1575.42 MHz. Bandpass filtering removes out-of-band interference and noise.

**Link to Sinais e Sistemas**: Butterworth filters provide maximally flat passband response. The filter transfer function in the frequency domain selectively attenuates frequencies outside the desired band.

## 2. Feature Extraction

### 2.1 Cross-Correlation via FFT
**Decision**: Use FFT-based correlation instead of direct time-domain correlation.

**Rationale**: FFT-based correlation is significantly faster (O(N log N) vs O(N²)) and mathematically equivalent due to the convolution theorem.

**Link to Sinais e Sistemas**: The convolution theorem states that correlation in time domain equals multiplication in frequency domain: corr(x,y) = IFFT(FFT(x) · conj(FFT(y))).

### 2.2 Full Width at Half Maximum (FWHM)
**Decision**: Use FWHM as a key feature for correlation peak analysis.

**Rationale**: In GPS signal processing, correlation peaks from authentic signals have characteristic widths related to the C/A code chip rate. Spoofed signals often exhibit different peak widths due to multipath or imperfect synchronization.

**Link to Sinais e Sistemas**: FWHM measures the bandwidth of a signal or impulse response in time domain. For GPS C/A codes, theoretical FWHM relates to the reciprocal of the chip rate (1/1.023 MHz ≈ 0.98 μs).

### 2.3 Peak-to-Secondary Ratio
**Decision**: Extract ratio of primary correlation peak to secondary peaks.

**Rationale**: Authentic GPS signals produce a strong primary peak at the correct code phase with much smaller secondary lobes. Spoofing can create anomalous secondary peaks.

**Link to Sinais e Sistemas**: This relates to the autocorrelation properties of pseudo-random codes. Gold codes (used in GPS) have controlled autocorrelation side lobes.

### 2.4 Statistical Moments (Skewness, Kurtosis)
**Decision**: Include higher-order statistical moments of correlation profiles.

**Rationale**: These capture the shape of correlation distributions. Spoofed signals may have asymmetric or heavy-tailed distributions.

**Link to Sinais e Sistemas**: Moments characterize probability distributions. Skewness measures asymmetry, kurtosis measures tail heaviness.

## 3. C/N0 Estimation

### 3.1 Correlation-Based C/N0
**Decision**: Estimate Carrier-to-Noise density ratio (C/N0) from correlation peak.

**Rationale**: C/N0 is a standard GPS signal quality metric. Spoofed signals often have abnormal C/N0 values.

**Method**: C/N0 = 10 log₁₀(Psignal / (Pnoise / fs))

**Link to Sinais e Sistemas**: Signal-to-noise ratio (SNR) is the ratio of signal power to noise power. C/N0 normalizes by bandwidth (sampling frequency fs).

## 4. Classification

### 4.1 Random Forest as Primary Model
**Decision**: Use Random Forest with class_weight='balanced' as the primary classifier.

**Rationale**: 
- Handles non-linear relationships well
- Robust to outliers
- Provides feature importance
- class_weight='balanced' handles class imbalance automatically
- Less prone to overfitting than single decision trees

### 4.2 SMOTE as Optional Technique
**Decision**: Provide SMOTE (Synthetic Minority Over-sampling Technique) as an option.

**Rationale**: If dataset has severe class imbalance, SMOTE can generate synthetic samples of the minority class to improve model training.

**Trade-off**: SMOTE can introduce synthetic patterns that don't exist in real data, so we make it optional.

### 4.3 Pipeline Architecture
**Decision**: Use scikit-learn Pipelines with preprocessing steps.

**Rationale**: 
- Ensures consistent preprocessing for training and inference
- Prevents data leakage during cross-validation
- Simplifies model persistence and deployment
- Includes imputation for handling missing values

**Pipeline structure**:
1. SimpleImputer (median strategy) - handles missing values
2. StandardScaler - normalizes features to zero mean, unit variance
3. Optional: SMOTE - oversamples minority class
4. Classifier - Random Forest, SVM, or MLP

## 5. Signal Processing Concepts Applied

### 5.1 Sampling Theorem
GPS signals are typically sampled at 5 MHz or higher to satisfy Nyquist criterion for the 2.046 MHz null-to-null bandwidth of C/A code.

### 5.2 Discrete-Time Signal Processing
All processing uses discrete-time representations. Continuous-time concepts from Sinais e Sistemas are approximated using discrete samples.

### 5.3 Frequency Domain Analysis
FFT provides efficient frequency domain analysis, enabling:
- Spectral visualization
- Fast correlation computation
- Bandpass filtering design

### 5.4 Windowing and Segmentation
Signal windows enable:
- Time-localized feature extraction
- Detection of temporal variations in spoofing
- Computational tractability for long signals

## 6. Integration with FGI-SpoofRepo Dataset

### 6.1 Dataset Structure
The FGI-SpoofRepo provides real GPS signals with known spoofing ground truth, including:
- Pre-spoofing authentic signals
- Various spoofing scenarios (meaconing, signal synthesis)
- Post-spoofing signals

### 6.2 Labeling Strategy
**Decision**: Use time-based labeling with SPOOF_START_TIME_S parameter.

**Rationale**: FGI-SpoofRepo documents when spoofing begins. We label segments before this time as authentic (0) and after as spoofed (1).

### 6.3 Data Format
The pipeline supports:
- Binary IQ data (.bin, .dat) - common in SDR recordings
- MATLAB files (.mat) - used in academic research
- CSV files (.csv) - for processed data

## 7. Performance Considerations

### 7.1 FFT-Based Processing
Using FFT for correlation provides O(N log N) complexity instead of O(N²), enabling real-time processing.

### 7.2 Vectorized Operations
NumPy vectorization provides near-C performance for array operations in Python.

### 7.3 Pipeline Parallelism
Random Forest and cross-validation use `n_jobs=-1` to leverage all CPU cores.

## 8. Future Improvements

Potential enhancements for future work:
1. Deep learning models (CNNs on spectrograms)
2. Real-time streaming processing
3. Multi-satellite correlation analysis
4. Advanced multipath detection
5. Doppler-based features

## References

- Oppenheim, A. V., & Willsky, A. S. "Signals and Systems" (Sinais e Sistemas)
- GPS Interface Control Document (ICD-GPS-200)
- FGI-SpoofRepo: Finnish Geospatial Research Institute GPS Spoofing Repository
- Psiaki, M. L., & Humphreys, T. E. (2016). "GNSS Spoofing and Detection"
