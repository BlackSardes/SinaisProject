<<<<<<< HEAD
# Decisões Técnicas: Detecção de Spoofing em Sinais GPS

Este documento explica as decisões técnicas do projeto com base em conceitos fundamentais de **Sinais e Sistemas** (ES413).

## 1. Visão Geral do Problema

### 1.1 Natureza do Sinal GPS

O sinal GPS L1 C/A é composto por:
- **Código C/A (Course/Acquisition)**: Sequência pseudo-aleatória de 1023 chips a 1.023 MHz
- **Portadora L1**: 1575.42 MHz (geralmente mixada para banda base)
- **Dados de navegação**: 50 bps

**Propriedades importantes**:
- Os códigos C/A são sequências Gold com boas propriedades de autocorrelação
- Ortogonalidade entre diferentes PRNs
- Baixa potência recebida (~-130 dBm)

### 1.2 Ataque de Spoofing

Um ataque de spoofing envolve:
1. **Geração de sinal falso** com código C/A válido
2. **Sincronização** com sinal autêntico (ou ligeiramente deslocado)
3. **Aumento gradual de potência** para dominar o sinal autêntico

**Assinaturas do spoofing**:
- Distorção da função de autocorrelação (ACF)
- Aumento não natural de C/N0
- Múltiplos picos de correlação
- Assimetrias no perfil de correlação

---

## 2. Decisões de Pré-processamento

### 2.1 Normalização de Potência

**Decisão**: Normalizar sinal para potência unitária (E[|x|²] = 1)

**Justificativa (Sinais e Sistemas)**:
- Variações de ganho do AGC (Automatic Gain Control) do receptor não devem afetar a detecção
- Permite comparação consistente de features entre diferentes sessões de gravação
- Essencial para cálculo preciso de C/N0

**Implementação**: `normalize_signal()` em `src/preprocessing/normalization.py`

```python
power = np.mean(np.abs(signal)**2)
signal_normalized = signal / np.sqrt(power)
```

### 2.2 Filtro Passa-Banda

**Decisão**: Implementar filtro Butterworth passa-banda

**Justificativa (Sinais e Sistemas)**:
- GPS C/A tem largura de banda de ~2 MHz (lóbulo principal)
- Filtro rejeita ruído fora da banda de interesse
- **Teorema de Nyquist**: Taxa de amostragem típica (5-25 MHz) é suficiente para capturar o sinal

**Parâmetros típicos**:
- Banda: 0-2 MHz (para sinal em banda base)
- Ordem: 5 (compromisso entre seletividade e fase linear)

**Implementação**: `bandpass_filter()` em `src/preprocessing/filtering.py`

### 2.3 Remoção de DC

**Decisão**: Subtrair média das componentes I e Q

**Justificativa (Sinais e Sistemas)**:
- Offset DC é artefato do hardware (imperfeições do misturador)
- Não contém informação sobre código C/A ou spoofing
- Pode afetar métricas de correlação se não removido

**Implementação**: `remove_dc()` em `src/preprocessing/normalization.py`

### 2.4 Janelamento (Windowing)

**Decisão**: Segmentar sinal em janelas de 0.5-1 segundo com overlap de 50%

**Justificativa (Sinais e Sistemas)**:
- **Análise tempo-frequência**: Permite detectar variações temporais
- 0.5s ≈ 500 períodos do código C/A (suficiente para estatísticas robustas)
- Overlap de 50% aumenta número de amostras sem redundância excessiva

**Implementação**: `window_segment()` em `src/preprocessing/windowing.py`

---

## 3. Decisões de Extração de Features

### 3.1 Correlação Cruzada (Cross-Correlation)

**Decisão**: Usar correlação via FFT (método rápido)

**Justificativa (Sinais e Sistemas)**:
- **Teorema da Convolução**: Correlação no tempo = multiplicação no domínio da frequência
- Complexidade O(N log N) vs O(N²) para correlação direta
- Correlação é fundamental para aquisição GPS

**Equação**:
```
R[k] = IFFT(FFT(signal) * conj(FFT(code)))
```

**Implementação**: `compute_cross_correlation()` em `src/features/correlation.py`

### 3.2 Features do Perfil de Correlação

#### 3.2.1 Peak-to-Secondary Ratio (P/S Ratio)

**Decisão**: Razão entre pico principal e pico secundário

**Justificativa (Sinais e Sistemas)**:
- **Propriedade de autocorrelação**: Código C/A ideal tem pico principal >> picos secundários
- Spoofing cria "ombros" ou picos secundários
- Indicador robusto de múltiplos sinais

**Valor típico**:
- Autêntico: P/S > 10
- Spoofing: P/S < 5

**Implementação**: `compute_peak_ratio()` em `src/features/correlation_features.py`

#### 3.2.2 FWHM (Full Width at Half Maximum)

**Decisão**: Largura do pico de correlação ao meio da altura máxima

**Justificativa (Sinais e Sistemas)**:
- **Relação de incerteza**: Δt · Δf ≥ 1/4π
- Código C/A tem duração de chip de ~1 μs → FWHM esperado ≈ 1-2 chips
- Múltiplos sinais alargam o pico (superposição)

**Implementação**: `compute_fwhm()` em `src/features/correlation_features.py`

#### 3.2.3 Assimetria (Asymmetry)

**Decisão**: Diferença entre área esquerda e direita do pico

**Justificativa (Sinais e Sistemas)**:
- Pico ideal é simétrico (função triângulo para código C/A)
- Spoofing sincronizado cria assimetria devido a atraso relativo
- Métrica sensível a fase relativa entre sinais

**Equação**:
```
Asymmetry = (Area_right - Area_left) / (Area_right + Area_left)
```

**Implementação**: `compute_asymmetry()` em `src/features/correlation_features.py`

### 3.3 Carrier-to-Noise Density Ratio (C/N0)

**Decisão**: Estimar C/N0 a partir do pico de correlação

**Justificativa (Sinais e Sistemas)**:
- **SNR vs C/N0**: SNR depende da largura de banda; C/N0 é normalizado por Hz
- C/N0 = 10 log₁₀(P_carrier / (N₀ · B))
- Spoofing de alta potência aumenta C/N0 abruptamente

**Método de estimação**:
1. Pico de correlação ≈ potência da portadora
2. Região longe do pico ≈ nível de ruído
3. C/N0 = 10 log₁₀(peak² / noise_floor) + 10 log₁₀(fs)

**Implementação**: `estimate_cn0_from_correlation()` em `src/preprocessing/cn0_estimation.py`

### 3.4 Skewness e Kurtosis

**Decisão**: Incluir momentos estatísticos de ordem superior

**Justificativa (Sinais e Sistemas)**:
- **Skewness**: Mede assimetria da distribuição
- **Kurtosis**: Mede "pesadez" das caudas
- Ruído Gaussiano tem skewness=0, kurtosis=3
- Spoofing pode alterar distribuição de amplitude

**Implementação**: Usando `scipy.stats.skew` e `scipy.stats.kurtosis`

---

## 4. Decisões de Classificação

### 4.1 Modelo Principal: Random Forest

**Decisão**: RandomForestClassifier como modelo primário

**Justificativa**:
- **Robustez**: Menos sensível a outliers que SVM
- **Interpretabilidade**: Feature importance identifica métricas mais relevantes
- **Desempenho**: Excelente para classificação binária com features numéricas
- **Não-linearidade**: Captura interações complexas entre features

**Hiperparâmetros**:
- `n_estimators=100`: Suficiente para estabilidade
- `class_weight='balanced'`: Lida com desbalanceamento de classes
- `max_depth=None`: Árvores profundas para capturar padrões sutis

**Implementação**: `get_classifier()` em `src/models/classifiers.py`

### 4.2 Tratamento de Desbalanceamento

**Decisão**: Usar `class_weight='balanced'` + opção de SMOTE

**Justificativa**:
- **class_weight='balanced'**: Penaliza erros na classe minoritária
- **SMOTE**: Gera amostras sintéticas da classe minoritária
  - Útil quando minoritária < 10% do dataset
  - Interpolação no espaço de features

**Trade-off**:
- class_weight: Mais rápido, não aumenta tamanho do dataset
- SMOTE: Pode melhorar recall da classe minoritária, mas risco de overfitting

**Implementação**: 
- `train_model()` usa class_weight
- `train_with_smote()` adiciona SMOTE pipeline

### 4.3 Validação Cruzada

**Decisão**: 5-fold stratified cross-validation

**Justificativa**:
- **Stratified**: Mantém proporção de classes em cada fold
- **k=5**: Compromisso entre viés e variância
- Estima generalização sem holdout set separado

**Implementação**: `cross_validate_model()` em `src/models/evaluation.py`

---

## 5. Métricas de Avaliação

### 5.1 Métricas Escolhidas

**Decisão**: Usar conjunto completo de métricas

1. **Accuracy**: Métrica geral
2. **Precision**: Evitar falsos positivos (alarmes falsos)
3. **Recall**: Detectar todos os spoofings (crítico para segurança)
4. **F1-Score**: Harmônica entre precision e recall
5. **ROC-AUC**: Avalia performance em diferentes thresholds

**Justificativa**:
- Aplicação de segurança: **Recall alto é crítico** (não perder ataques)
- Precision também importante (evitar alarmes falsos excessivos)
- ROC-AUC avalia qualidade geral do classificador

### 5.2 Matriz de Confusão

**Decisão**: Visualizar matriz de confusão em todas as avaliações

**Interpretação no contexto GPS**:
- **True Positive (TP)**: Spoofing detectado corretamente ✓
- **False Negative (FN)**: Spoofing não detectado ✗ (CRÍTICO)
- **False Positive (FP)**: Alarme falso ✗ (prejudica confiança)
- **True Negative (TN)**: Autêntico identificado corretamente ✓

---

## 6. Limitações e Trabalhos Futuros

### 6.1 Limitações Atuais

1. **Simplified Doppler**: Não implementado busca completa Doppler
2. **Single PRN**: Pipeline foca em um PRN por vez
3. **Simplified C/N0**: Método simplificado (não considera coerência)
4. **No Multipath**: Não distingue spoofing de multipath severo

### 6.2 Melhorias Futuras

1. **Multi-PRN Analysis**: Correlação entre múltiplos satélites
2. **Temporal Tracking**: Análise de transições autêntico→spoofed
3. **Deep Learning**: CNN/LSTM para padrões temporais complexos
4. **Receiver Clock Bias**: Análise de saltos no relógio do receptor
5. **Vector Tracking**: Análise de consistência geométrica

---

## 7. Integração com Dataset Real (FGI-SpoofRepo)

### 7.1 Estrutura Esperada

```
data/
├── FGI-SpoofRepo/
│   ├── scenario1/
│   │   ├── authentic.bin
│   │   └── spoofed.bin
│   └── scenario2/
│       └── ...
```

### 7.2 Adaptações Necessárias

1. **Formato de Arquivo**:
   - FGI usa formato int16 interleaved I/Q
   - `read_iq_binary()` já suporta este formato

2. **Metadados**:
   - Taxa de amostragem pode variar (5-25 MHz)
   - PRNs visíveis variam por cenário
   - Timestamps de início/fim de spoofing devem ser extraídos

3. **Rotulagem**:
   - Implementar lógica de rotulagem baseada em timestamps
   - Separar janelas autênticas e spoofed

---

## 8. Reprodutibilidade

Todas as operações estocásticas usam `random_state=42`:
- Split treino/teste
- Inicialização de modelos
- Geração de dados sintéticos
- SMOTE

Isso garante reprodutibilidade entre execuções.

---

## 9. Referências

1. **GPS Signal Structure**: IS-GPS-200 (Interface Specification)
2. **C/A Code Generation**: Gold Codes and GPS
3. **Spoofing Detection**: Psiaki & Humphreys (2016)
4. **Signal Processing**: Oppenheim & Schafer - Discrete-Time Signal Processing
5. **Machine Learning**: Hastie et al. - Elements of Statistical Learning

---

**Autores**: Equipe ES413 - Sinais e Sistemas  
**Data**: Dezembro 2024  
**Versão**: 1.0
=======

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
>>>>>>> main
