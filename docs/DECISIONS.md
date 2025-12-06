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
