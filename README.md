# SinaisProject - Detecção de Spoofing em Sinais GPS

Pipeline robusto para detecção de ataques de spoofing em sinais GNSS (GPS) utilizando análise de correlação, métricas de qualidade de sinal (SQMs), e aprendizado de máquina.

**Disciplina**: ES413 - Sinais e Sistemas (Cin/UFPE)

---

## 📋 Sumário

- [Características](#-características)
- [Instalação](#-instalação)
- [Uso Rápido](#-uso-rápido)
- [Estrutura do Projeto](#-estrutura-do-projeto)
- [Datasets](#-datasets)
- [Notebooks](#-notebooks)
- [Testes](#-testes)
- [Documentação Técnica](#-documentação-técnica)
- [Contribuindo](#-contribuindo)

---

## ✨ Características

- **Pré-processamento Completo**: Normalização, filtragem, correção Doppler, remoção de outliers
- **Extração de Features Avançada**: 
  - Métricas SQM (Signal Quality Monitoring): Peak-to-Secondary, FWHM, Asymmetry
  - Métricas de potência: C/N0, SNR, Noise Floor
  - Features estatísticas: Skewness, Kurtosis
- **Modelos de ML**:
  - Random Forest (priorizado) com balanceamento automático
  - SVM e MLP Neural Network como alternativas
  - Suporte para SMOTE (balanceamento sintético)
- **Gerador de Dados Sintéticos**: Permite execução offline sem datasets grandes
- **Suporte para Datasets Reais**: FGI-SpoofRepo e TEXBAT
- **Visualizações Avançadas**: Perfis de correlação, ROC curves, distribuições de features
- **Pipeline End-to-End**: Script automatizado para execução completa

---

## 🚀 Instalação

### Pré-requisitos

- Python 3.9+
- pip ou conda

### Instalação via pip

```bash
# Clone o repositório
git clone https://github.com/BlackSardes/SinaisProject.git
cd SinaisProject

# Instale as dependências
pip install -r requirements.txt
```

### Instalação via conda

```bash
# Clone o repositório
git clone https://github.com/BlackSardes/SinaisProject.git
cd SinaisProject

# Crie ambiente conda
conda env create -f environment.yml
conda activate sinais-gps-spoofing
```

### Dependências Opcionais

Para usar TensorFlow/Keras ou Librosa (análise avançada):
```bash
pip install tensorflow keras librosa
```

Ou descomente as linhas correspondentes em `requirements.txt` ou `environment.yml`.

---

## 🎯 Uso Rápido

### 1. Executar Pipeline Completo com Dados Sintéticos

```bash
python scripts/script_run_pipeline.py --mode synthetic --num-samples 200
```

**Saída**: Modelo treinado, métricas e visualizações em `data/processed/`

### 2. Executar com Dataset Real (TEXBAT)

```bash
python scripts/script_run_pipeline.py \
  --mode texbat \
  --data-dir data/raw/texbat \
  --spoof-time 17.0 \
  --num-samples 500
```

### 3. Usar nos Notebooks

```bash
jupyter notebook notebooks/
```

Abra:
- `EDA.ipynb`: Análise exploratória de dados
- `feature_demo.ipynb`: Demonstração de extração de features
- `training_eval.ipynb`: Treinamento e avaliação de modelos

### 4. Usar como Biblioteca Python

```python
from src.utils.synthetic_data import generate_synthetic_dataset
from src.preprocessing.signal_processing import generate_ca_code
from src.features.pipeline import build_feature_vector
from src.models.train import train_model

# Gerar dados
signals, labels, metadata = generate_synthetic_dataset(
    num_authentic=100, num_spoofed=100, fs=5e6
)

# Extrair features
prn_code = generate_ca_code(prn=1)
features = build_feature_vector(signals[0], prn_code, fs=5e6)

# Treinar modelo
model, metrics = train_model(X_train, y_train, model_name='random_forest')
```

---

## 📁 Estrutura do Projeto

```
SinaisProject/
├── data/
│   ├── raw/              # Dados brutos (FGI, TEXBAT) - não versionados
│   └── processed/        # Features, modelos treinados
├── notebooks/
│   ├── EDA.ipynb         # Análise exploratória
│   ├── feature_demo.ipynb # Demonstração de features
│   └── training_eval.ipynb # Treinamento de modelos
├── src/
│   ├── preprocessing/    # Pré-processamento de sinais
│   │   └── signal_processing.py
│   ├── features/         # Extração de features
│   │   ├── correlation.py
│   │   ├── temporal.py
│   │   └── pipeline.py
│   ├── models/           # Treinamento e avaliação
│   │   ├── train.py
│   │   └── persistence.py
│   └── utils/            # Utilitários
│       ├── plots.py
│       ├── synthetic_data.py
│       └── data_loader.py
├── scripts/
│   └── script_run_pipeline.py  # Script de execução completa
├── tests/                # Testes pytest
│   ├── test_preprocessing.py
│   ├── test_features.py
│   └── test_pipeline.py
├── docs/
│   └── DECISIONS.md      # Decisões técnicas e fundamentos
├── requirements.txt      # Dependências Python
├── environment.yml       # Ambiente conda
└── README.md             # Este arquivo
```

---

## 📊 Datasets

### Dados Sintéticos (Padrão)

O pipeline inclui um gerador de dados sintéticos GPS que permite execução sem downloads:

```python
from src.utils.synthetic_data import generate_synthetic_dataset

signals, labels, metadata = generate_synthetic_dataset(
    num_authentic=100,
    num_spoofed=100,
    fs=5e6,
    duration=0.5
)
```

**Características**:
- Sinais autênticos: C/N0 40-50 dB-Hz, Doppler ±5 kHz
- Sinais spoofed: Power attacks (5-15 dB acima), secondary peaks

### FGI-SpoofRepo (Dataset Real)

**Fonte**: Finnish Geospatial Institute  
**Link**: https://github.com/Finnish-Geospatial-Institute/FGI-SpoofRepo

**Instruções de Instalação**:

1. Baixe o dataset do GitHub (>10 GB)
2. Extraia para `data/raw/fgi-spoof-repo/`
3. Estrutura esperada:
   ```
   data/raw/fgi-spoof-repo/
   ├── scenario_1/
   │   ├── authentic/
   │   │   └── *.bin
   │   └── spoofed/
   │       └── *.bin
   └── scenario_2/
       └── ...
   ```

**Uso**:
```python
from src.utils.data_loader import load_fgi_dataset

signals, labels, metadata = load_fgi_dataset('data/raw/fgi-spoof-repo')
```

### TEXBAT (Dataset de Referência)

**Descrição**: Texas Spoofing Test Battery - dataset acadêmico

**Características**:
- Formato: Binário int16 interleaved I/Q
- Taxa de amostragem: 5 MHz (configurável)
- Ground truth: Time-based (início do spoofing em timestamp conhecido)

**Instruções**:

1. Obtenha o dataset TEXBAT (contato: instituições acadêmicas)
2. Coloque arquivos `.bin`/`.dat` em `data/raw/texbat/`
3. Configure o tempo de início do spoofing (padrão: 17.0s)

**Uso**:
```python
from src.utils.data_loader import load_texbat_dataset

signals, labels, metadata = load_texbat_dataset(
    'data/raw/texbat',
    fs=5e6,
    spoof_start_time=17.0
)
```

---

## 📓 Notebooks

### 1. EDA.ipynb - Análise Exploratória

**Conteúdo**:
- Visualização de sinais GPS (tempo e frequência)
- Constelações IQ
- Perfis de correlação autênticos vs spoofed
- Distribuições de C/N0

**Execução**:
```bash
jupyter notebook notebooks/EDA.ipynb
```

### 2. feature_demo.ipynb - Demonstração de Features

**Conteúdo**:
- Extração passo-a-passo de features
- Análise de importância de features
- Correlação entre features
- Distribuições por classe

### 3. training_eval.ipynb - Treinamento e Avaliação

**Conteúdo**:
- Treinamento de Random Forest, SVM, MLP
- Comparação de modelos
- Métricas detalhadas (confusion matrix, ROC curves)
- Persistência de modelos

---

## 🧪 Testes

O projeto inclui testes unitários e de integração com pytest.

### Executar Todos os Testes

```bash
pytest tests/ -v
```

### Executar Testes Específicos

```bash
# Testes de preprocessing
pytest tests/test_preprocessing.py -v

# Testes de features
pytest tests/test_features.py -v

# Testes de pipeline completo
pytest tests/test_pipeline.py -v
```

### Coverage

```bash
pytest tests/ --cov=src --cov-report=html
```

Abra `htmlcov/index.html` para ver relatório detalhado.

---

## 📚 Documentação Técnica

Consulte [`docs/DECISIONS.md`](docs/DECISIONS.md) para:

- Fundamentos de Sinais e Sistemas aplicados
- Justificativa para escolha de features
- Teoria de correlação e códigos C/A
- Estimativa de C/N0 e limitações
- Escolha de modelos de ML
- Referências bibliográficas

**Tópicos principais**:
- Geração de Códigos C/A (Gold Sequences)
- Propriedades de Autocorrelação
- Métricas SQM (Peak-to-Secondary, FWHM, Asymmetry)
- C/N0 vs SNR
- Random Forest vs SVM vs MLP
- Balanceamento de classes (SMOTE vs Class Weight)

---

## 🔧 Configuração Avançada

### Personalizar Parâmetros do Pipeline

Edite `scripts/script_run_pipeline.py` ou use flags CLI:

```bash
python scripts/script_run_pipeline.py \
  --mode synthetic \
  --num-samples 500 \
  --model random_forest \
  --use-smote \
  --fs 5e6 \
  --duration 0.5 \
  --output-dir results/ \
  --random-seed 42
```

**Parâmetros disponíveis**:
- `--mode`: synthetic, fgi, texbat
- `--model`: random_forest, svm, mlp
- `--use-smote`: Ativar SMOTE para balanceamento
- `--fs`: Frequência de amostragem (Hz)
- `--duration`: Duração dos segmentos (segundos)
- `--spoof-time`: Tempo de início do spoofing (TEXBAT)

### Adicionar Novos Modelos

1. Implemente em `src/models/train.py`
2. Adicione à função `train_model()` com parâmetros padrão
3. Atualize documentação

---

## 🤝 Contribuindo

Contribuições são bem-vindas!

### Diretrizes

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/nova-feature`)
3. Commit suas mudanças (`git commit -m 'Add nova feature'`)
4. Push para a branch (`git push origin feature/nova-feature`)
5. Abra um Pull Request

### Checklist de PR

- [ ] Código segue estilo do projeto (PEP8)
- [ ] Testes adicionados para novas funcionalidades
- [ ] Todos os testes passam (`pytest`)
- [ ] Documentação atualizada (README, DECISIONS.md)
- [ ] Docstrings em funções públicas

---

## 📄 Licença

Este projeto foi desenvolvido para fins acadêmicos (ES413 - Cin/UFPE).

---

## 📧 Contato

**Projeto**: SinaisProject  
**Repositório**: https://github.com/BlackSardes/SinaisProject  
**Disciplina**: ES413 - Sinais e Sistemas (Cin/UFPE)

---

## 🙏 Agradecimentos

- **Docentes de ES413**: Pelos fundamentos de Sinais e Sistemas
- **Finnish Geospatial Institute**: Pelo dataset FGI-SpoofRepo
- **Comunidade de GNSS Security**: Pelas referências e datasets

---

## 📝 Changelog

### v1.0.0 (2024-12-06)

**Implementado**:
- ✅ Pipeline completa de pré-processamento
- ✅ Extração de features SQM e potência
- ✅ Modelos de ML (Random Forest, SVM, MLP)
- ✅ Gerador de dados sintéticos
- ✅ Suporte para FGI-SpoofRepo e TEXBAT
- ✅ Notebooks de análise e treinamento
- ✅ Testes unitários e de integração
- ✅ Documentação técnica completa

**Próximos passos**:
- Multi-PRN fusion
- Temporal features (LSTM)
- Real-time processing
- GUI para visualização

---

**Desenvolvido com ❤️ para detecção de spoofing GPS**
