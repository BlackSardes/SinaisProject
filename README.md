<<<<<<< HEAD
# GPS Spoofing Detection Project

**Disciplina**: ES413 - Sinais e Sistemas  
**Instituição**: Cin/UFPE  
**Objetivo**: Detecção de ataques de spoofing em sinais GPS usando processamento de sinais e machine learning

---

## 📋 Índice

- [Visão Geral](#visão-geral)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Instalação](#instalação)
- [Uso Rápido](#uso-rápido)
- [Datasets](#datasets)
- [Pipeline Completa](#pipeline-completa)
- [Módulos](#módulos)
- [Notebooks](#notebooks)
- [Testes](#testes)
- [Documentação](#documentação)
- [Contribuindo](#contribuindo)

---

## 🎯 Visão Geral

Este projeto implementa uma pipeline completa para detecção de spoofing em sinais GPS L1 C/A, incluindo:

- **Pré-processamento robusto**: Normalização, filtragem, remoção de DC, windowing
- **Extração de features**: Métricas de correlação (FWHM, P/S ratio, assimetria), C/N0, features temporais
- **Classificação**: Random Forest, SVM, MLP com tratamento de desbalanceamento
- **Avaliação**: Métricas completas (accuracy, precision, recall, F1, ROC-AUC)
- **Visualizações**: Confusion matrix, ROC curves, feature distributions

**Diferencial**: Todas as decisões técnicas são fundamentadas em conceitos de **Sinais e Sistemas** (veja [DECISIONS.md](docs/DECISIONS.md)).

---

## 📁 Estrutura do Projeto

```
SinaisProject/
├── data/                          # Dados GPS (não versionados)
│   └── .gitkeep
├── docs/                          # Documentação
│   └── DECISIONS.md              # Justificativas técnicas detalhadas
├── notebooks/                     # Jupyter notebooks
│   ├── EDA.ipynb                 # Análise exploratória
│   ├── feature_demo.ipynb        # Demonstração de features
│   └── training_eval.ipynb       # Treinamento e avaliação
├── scripts/                       # Scripts executáveis
│   └── run_pipeline.py           # Pipeline completa
├── src/                          # Código fonte
│   ├── preprocessing/            # Pré-processamento de sinais
│   │   ├── signal_io.py         # Leitura de sinais (bin, csv, mat)
│   │   ├── normalization.py     # Normalização e remoção DC
│   │   ├── filtering.py         # Filtros passa-banda, notch
│   │   ├── windowing.py         # Segmentação
│   │   ├── cn0_estimation.py    # Estimação C/N0
│   │   └── resampling.py        # Reamostragem
│   ├── features/                 # Extração de features
│   │   ├── correlation.py       # Correlação cruzada/auto
│   │   ├── correlation_features.py  # Features do perfil (FWHM, P/S, etc)
│   │   ├── temporal_features.py # Features temporais
│   │   └── feature_pipeline.py  # Pipeline de features
│   ├── models/                   # Modelos de classificação
│   │   ├── classifiers.py       # Random Forest, SVM, MLP
│   │   ├── training.py          # Treinamento (com/sem SMOTE)
│   │   ├── evaluation.py        # Métricas e avaliação
│   │   └── persistence.py       # Save/load modelos
│   └── utils/                    # Utilitários
│       ├── plots.py             # Visualizações
│       └── synthetic_data.py    # Gerador de dados sintéticos
├── tests/                        # Testes unitários
├── results/                      # Resultados (não versionados)
├── requirements.txt              # Dependências Python
├── environment.yml               # Ambiente conda
└── README.md                     # Este arquivo
```

---

## 🚀 Instalação

### Opção 1: pip (Recomendado)

```bash
# Clone o repositório
git clone https://github.com/BlackSardes/SinaisProject.git
cd SinaisProject

# Crie ambiente virtual
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# Instale dependências
pip install -r requirements.txt
```

### Opção 2: conda

```bash
# Clone o repositório
git clone https://github.com/BlackSardes/SinaisProject.git
cd SinaisProject

# Crie ambiente conda
conda env create -f environment.yml
conda activate gps-spoofing
```

### Dependências Principais

- **Essenciais**: numpy, scipy, pandas, scikit-learn, matplotlib, seaborn
- **ML Avançado**: imbalanced-learn (SMOTE)
- **Persistência**: joblib
- **Notebooks**: jupyter
- **Testes**: pytest

**Nota**: TensorFlow/Keras e librosa estão comentados em `requirements.txt` e `environment.yml` (dependências pesadas opcionais).

---

## ⚡ Uso Rápido

### 1. Executar Pipeline com Dados Sintéticos

```bash
# Pipeline completa com dados sintéticos
python scripts/run_pipeline.py --synthetic --n-authentic 200 --n-spoofed 200

# Com SMOTE para balanceamento
python scripts/run_pipeline.py --synthetic --use-smote

# Escolher modelo diferente
python scripts/run_pipeline.py --synthetic --model svm
```

**Saída**: Modelo treinado, relatório de avaliação, visualizações em `results/`

### 2. Usar Módulos Individualmente

```python
# Gerar dados sintéticos
from src.utils.synthetic_data import generate_synthetic_dataset
signals, labels = generate_synthetic_dataset(n_authentic=100, n_spoofed=100)

# Extrair features
from src.features.feature_pipeline import build_feature_vector
features_df = build_feature_vector(signals, fs=5e6, prn=1)
features_df['label'] = labels

# Treinar modelo
from src.models.training import train_model
X = features_df.drop(['segment_id', 'label'], axis=1).values
y = features_df['label'].values
model, info = train_model(X, y, model_name='random_forest')

# Avaliar
from src.models.evaluation import evaluate_model
metrics = evaluate_model(model, info['X_test'], info['y_test'])
print(f"Accuracy: {metrics['test_accuracy']:.3f}")
```

---

## 📊 Datasets

### Dados Sintéticos (Inclusos)

O projeto inclui gerador de sinais GPS sintéticos para testes:
- Código C/A válido
- Ruído Gaussiano
- Opção de adicionar sinal de spoofing com delay e potência configuráveis

```python
from src.utils.synthetic_data import generate_synthetic_gps_signal

# Sinal autêntico
signal_auth = generate_synthetic_gps_signal(duration_s=0.5, cn0_db=45)

# Sinal com spoofing
signal_spoof = generate_synthetic_gps_signal(
    duration_s=0.5, cn0_db=45, add_spoofing=True,
    spoofing_delay_chips=0.3, spoofing_power_ratio=2.0
)
```

### Dataset Real: FGI-SpoofRepo (Opcional)

Para usar dados reais, baixe o **FGI-SpoofRepo**:

1. **Download**: https://github.com/nlsfi/FGI-GSRx/tree/master/Spoofing%20Dataset

2. **Estrutura recomendada**:
```
data/
└── FGI-SpoofRepo/
    ├── scenario1/
    │   ├── signal_authentic.bin
    │   └── signal_spoofed.bin
    ├── scenario2/
    └── ...
```

3. **Parâmetros típicos**:
   - Formato: int16 interleaved I/Q
   - Taxa de amostragem: 5-26 MHz (varia por cenário)
   - PRNs: múltiplos (1-32)

4. **Carregar sinal**:
```python
from src.preprocessing.signal_io import load_signal

signal = load_signal(
    'data/FGI-SpoofRepo/scenario1/signal.bin',
    file_format='binary',
    count_samples=int(0.5 * 5e6)  # 0.5s @ 5 MHz
)
```

**Nota**: Implementar rotulagem automática baseada em timestamps do FGI (veja `docs/DECISIONS.md` seção 7).

---

## 🔬 Pipeline Completa

A pipeline segue estas etapas:

```
1. Carregamento/Geração de Sinais
   ↓
2. Pré-processamento
   - Normalização de potência
   - Remoção DC
   - Filtragem (opcional)
   ↓
3. Segmentação (Windowing)
   - Janelas de 0.5-1s
   - Overlap de 50%
   ↓
4. Extração de Features
   - Correlação com código C/A
   - Features do perfil: FWHM, P/S ratio, assimetria
   - C/N0 e variação temporal
   - Features estatísticas
   ↓
5. Pré-processamento de Features
   - Imputação de valores faltantes
   - Padronização (StandardScaler)
   - PCA (opcional)
   ↓
6. Treinamento
   - Random Forest (class_weight='balanced')
   - Opção: SMOTE para balanceamento
   - Validação cruzada
   ↓
7. Avaliação
   - Métricas: accuracy, precision, recall, F1, ROC-AUC
   - Confusion matrix
   - Feature importance
   ↓
8. Persistência
   - Salvar modelo (.pkl)
   - Salvar metadados (.json)
   - Salvar visualizações
```

---

## 🧩 Módulos

### src/preprocessing

Funções de pré-processamento de sinais GPS:
- `load_signal()`: Carrega sinais de múltiplos formatos (.bin, .csv, .mat)
- `normalize_signal()`: Normalização de potência
- `bandpass_filter()`: Filtro passa-banda Butterworth
- `window_segment()`: Segmentação em janelas
- `estimate_cn0_from_correlation()`: Estimação de C/N0

### src/features

Extração de features para classificação:
- `compute_cross_correlation()`: Correlação rápida via FFT
- `extract_correlation_features()`: FWHM, P/S ratio, assimetria, etc.
- `extract_temporal_features()`: Features estatísticas do sinal
- `build_feature_vector()`: Pipeline completa de features

### src/models

Treinamento e avaliação de modelos:
- `train_model()`: Treina Random Forest/SVM/MLP
- `train_with_smote()`: Treina com balanceamento SMOTE
- `evaluate_model()`: Métricas completas
- `save_model()`, `load_model()`: Persistência

### src/utils

Utilitários e visualizações:
- `plot_confusion_matrix()`: Matriz de confusão
- `plot_roc_curves()`: Curvas ROC
- `plot_feature_distributions()`: Distribuição de features por classe
- `generate_synthetic_gps_signal()`: Gerador de sinais sintéticos

---

## 📓 Notebooks

### 1. EDA.ipynb - Análise Exploratória

- Visualização de sinais GPS
- Análise de perfis de correlação
- Distribuição de features
- Comparação autêntico vs spoofed

### 2. feature_demo.ipynb - Demonstração de Features

- Extração passo-a-passo de features
- Visualização de FWHM, P/S ratio
- Análise de sensibilidade
- Interpretação física

### 3. training_eval.ipynb - Treinamento e Avaliação

- Treinamento de múltiplos modelos
- Comparação de performance
- Análise de feature importance
- Tuning de hiperparâmetros

**Para executar**:
=======

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

>>>>>>> main
```bash
jupyter notebook notebooks/
```

<<<<<<< HEAD
---

## 🧪 Testes

Execute testes unitários:

```bash
# Todos os testes
pytest tests/

# Com cobertura
pytest tests/ --cov=src --cov-report=html

# Teste específico
pytest tests/test_features.py::test_fwhm_computation
```

**Testes implementados**:
- Geração de código C/A
- Normalização de sinal
- Cálculo de FWHM
- Pipeline mínima com dados sintéticos
- Persistência de modelos

---

## 📚 Documentação

### docs/DECISIONS.md

Documento **essencial** que explica:
- Fundamentos de Sinais e Sistemas
- Justificativa para cada decisão técnica
- Interpretação física de cada feature
- Trade-offs de modelos e métodos
- Limitações e trabalhos futuros

**Leitura obrigatória** para entender o projeto em profundidade.

---

## 🤝 Contribuindo

1. Fork o repositório
2. Crie branch para feature: `git checkout -b feature/nova-feature`
3. Commit: `git commit -m 'Add nova feature'`
4. Push: `git push origin feature/nova-feature`
5. Abra Pull Request

**Diretrizes**:
- Docstrings em todas as funções
- Testes para novas funcionalidades
- Justificar decisões técnicas

---

## 📄 Licença

Este projeto é de código aberto para fins educacionais (ES413).

---

## 👥 Autores

Equipe ES413 - Sinais e Sistemas  
Centro de Informática - UFPE

---

## 📞 Contato

Para dúvidas ou sugestões, abra uma **Issue** no GitHub.

---

**Última atualização**: Dezembro 2024
=======
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
>>>>>>> main
