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
```bash
jupyter notebook notebooks/
```

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
