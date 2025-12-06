# Decisões Técnicas e Fundamentos - Detecção de Spoofing GPS

## Sumário Executivo

Este documento detalha as escolhas técnicas, fundamentos de Sinais e Sistemas, e justificativas para o pipeline de detecção de spoofing em sinais GPS implementado neste repositório.

---

## 1. Fundamentos de Sinais GPS e C/A Code

### 1.1 Estrutura do Sinal GPS

Os sinais GPS L1 C/A (Coarse/Acquisition) são sinais BPSK (Binary Phase Shift Keying) compostos por:
- **Frequência portadora**: 1575.42 MHz
- **Código C/A**: Sequência Gold de 1023 chips a 1.023 MHz (≈1 ms de período)
- **Mensagem de navegação**: 50 bps modulada sobre o código

**Decisão**: O pipeline foca na análise do código C/A, pois:
1. É a base da aquisição e rastreamento
2. Distorções no perfil de correlação indicam spoofing
3. Análise independente da mensagem de navegação

### 1.2 Geração do Código C/A

Implementação em `src/preprocessing/signal_processing.py::generate_ca_code()`.

**Fundamento**: Os códigos C/A são sequências Gold geradas por dois registradores de deslocamento linear com realimentação (LFSR):
- G1: polinômio x¹⁰ + x³ + 1
- G2: polinômio x¹⁰ + x⁹ + x⁸ + x⁶ + x³ + x² + 1

**Propriedades importantes**:
- **Ortogonalidade**: Autocorrelação apresenta pico único em fase zero
- **Ruído-like**: Fora de fase, a correlação é próxima de zero
- **Unicidade**: Cada PRN (1-32) tem um padrão único de taps do G2

**Limitação conhecida**: Não implementamos os códigos P(Y) ou códigos militares.

---

## 2. Pré-processamento de Sinais

### 2.1 Normalização de Potência

**Função**: `normalize_by_power()` em `signal_processing.py`

**Fundamento de Sinais e Sistemas**:
A potência média de um sinal complexo x[n] é:
```
P = E[|x[n]|²] = (1/N) Σ |x[n]|²
```

**Decisão**: Normalizar para P=1 (0 dB) porque:
1. **Invariância ao ganho**: Receptores têm ganhos de RF variáveis
2. **Comparabilidade**: Permite comparar C/N0 entre segmentos
3. **Estabilidade numérica**: Evita overflow/underflow em cálculos

**Trade-off**: Perde-se informação absoluta de potência, mas esta é recuperada através da estimativa de C/N0.

### 2.2 Correção de Frequência Doppler/IF

**Função**: `apply_frequency_correction()`

**Fundamento (Propriedade da Modulação)**:
Multiplicar no tempo por e^(-j2πf_corr·t) desloca o espectro em -f_corr Hz.

```
y[n] = x[n] · e^(-j2πf_corr·n/fs)
```

**Aplicação GPS**:
- **Doppler**: Movimento relativo satélite-receptor (até ±5 kHz em L1)
- **IF (Frequência Intermediária)**: Frequência após downconversion no receptor

**Decisão**: Implementar correção explícita porque:
- Correlação no tempo requer coerência de fase
- Doppler não corrigido causa atenuação do pico de correlação (efeito sinc)

**Limitação**: No pipeline atual, assumimos f_corr fornecido; em sistema completo, seria estimado via busca Doppler.

### 2.3 Filtragem Bandpass

**Função**: `bandpass_filter()` - Butterworth IIR

**Fundamento**:
Filtro passa-faixa remove componentes fora da banda de interesse:
- **Rejeita DC drift**: Problemas de hardware (offset I/Q)
- **Limita ruído**: Reduz ruído fora da banda do sinal

**Parâmetros GPS**:
- Banda típica: ±2 MHz em torno da portadora (acomoda 95% da energia do espectro BPSK)

**Decisão**: Uso de Butterworth porque:
- Resposta em magnitude plana na banda passante
- Implementação eficiente (IIR de ordem baixa)

**Alternativa rejeitada**: Notch filters específicos (implementados em `pre_process.py` original) - menos genéricos.

### 2.4 Remoção de Outliers

**Função**: `remove_outliers()`

**Métodos disponíveis**:
1. **Median + MAD**: Limiar robusto baseado em Median Absolute Deviation
2. **IQR (Interquartile Range)**: Baseado em percentis

**Fundamento**: Interferências impulsivas (e.g., radar, switching power) criam picos de amplitude que:
- Distorcem estatísticas do sinal
- Mascaram o pico de correlação
- Introduzem viés na estimativa de C/N0

**Decisão**: Clipping suave (limitar ao threshold) em vez de remoção porque:
- Preserva comprimento do sinal
- Evita descontinuidades temporais

---

## 3. Extração de Features: Métricas SQM (Signal Quality Monitoring)

### 3.1 Correlação Cruzada via FFT

**Função**: `compute_cross_correlation()` em `correlation.py`

**Fundamento (Teorema da Convolução)**:
```
corr[n] = IFFT(FFT(signal) · conj(FFT(code)))
```

**Vantagens**:
- Complexidade O(N log N) vs O(N²) para convolução direta
- Essencial para sinais longos (milhões de amostras)

**Decisão**: Sempre usar FFT porque:
- GPS: segmentos típicos de 0.5s a 5 MHz = 2.5M amostras
- Speedup ~1000x comparado a convolução direta

### 3.2 Feature: Peak-to-Secondary Ratio (P/S)

**Implementação**: `extract_correlation_features()` - `peak_to_secondary`

**Fundamento de Sinais e Sistemas**:
A autocorrelação de códigos Gold ideais tem:
- **Pico principal**: Valor máximo em fase zero
- **Lóbulos secundários**: ~-21 dB abaixo do pico (teórico)

**Relação com Spoofing**:
- **Multipath**: Cria réplicas atrasadas → picos secundários aumentam → P/S diminui
- **Spoofing sincronizado**: Múltiplos sinais alinhados → deformação do pico → P/S diminui
- **Spoofing dessincronizado**: Pico secundário explícito → P/S diminui drasticamente

**Limiar típico**: P/S > 10 para sinal autêntico limpo.

**Decisão**: Feature crítica porque:
1. Direto indicador de integridade do código
2. Robusto a variações de ganho (é uma razão)
3. Base teórica sólida (propriedades de códigos Gold)

### 3.3 Feature: Full Width at Half Maximum (FWHM) e Fractional Peak Width (FPW)

**Implementação**: `fwhm`, `fpw` (80% do pico)

**Fundamento**:
O pico de autocorrelação ideal de um código Gold tem forma triangular com:
- Base ≈ 2 chips
- FWHM ≈ 1 chip

**Alargamento do pico indica**:
- **Multipath**: Soma de réplicas atrasadas
- **Spoofing com sincronização imperfeita**: Picos sobrepostos
- **Correlação parcial**: Código PRN incorreto ou distorção

**Decisão**: FPW a 80% (em vez de 50%) porque:
- Mais robusto a ruído (threshold maior)
- Captura melhor a forma do "ombro" criado por spoofing

### 3.4 Feature: Assimetria (Asymmetry)

**Fórmula**:
```
asymmetry = (A_right - A_left) / (A_right + A_left)
```
Onde A_right e A_left são as áreas à direita e esquerda do pico.

**Fundamento**:
Pico de autocorrelação ideal é simétrico. Assimetria surge de:
- **Multipath com delay**: Energia adicional em uma direção
- **Spoofing com offset de código**: Deslocamento assimétrico

**Decisão**: Feature discriminativa, mas:
- **Limitação**: Sensível a ruído em sinais de baixo C/N0
- **Mitigação**: Usar janela maior (± 1 chip) para integração

### 3.5 Feature: Skewness e Kurtosis

**Implementação**: `scipy.stats.skew()`, `scipy.stats.kurtosis()`

**Fundamento (Estatística):**
- **Skewness**: Mede assimetria da distribuição em torno do pico
- **Kurtosis**: Mede "cauda pesada" (outliers)

**Relação com Spoofing**:
Distribuições não-gaussianas indicam:
- Sinais múltiplos sobrepostos (mistura de distribuições)
- Interferência estruturada (não AWGN)

**Decisão**: Features complementares para capturar morfologia complexa.

---

## 4. Métricas de Potência

### 4.1 Estimativa de C/N0 (Carrier-to-Noise Density Ratio)

**Função**: `estimate_cn0_from_correlation()` e features em `temporal.py`

**Fundamento Teórico**:
C/N0 é a razão entre potência da portadora (C) e densidade espectral de potência de ruído (N0):
```
C/N0 [dB-Hz] = 10 log10(C / N0)
```

**Método de Estimação Simplificado**:
1. **Potência do sinal**: Aproximada pelo quadrado do pico de correlação
2. **Potência do ruído**: Média da correlação fora da janela do pico
3. **Bandwidth de ruído**: Relacionado a fs (frequência de amostragem)

**Fórmula implementada**:
```python
carrier_power = peak_value² / N_samples
noise_power = (total_power - carrier_power)
C/N0 = 10 log10(carrier_power / (noise_power / fs))
```

**Limitações conhecidas**:
1. **Não considera squaring loss**: Perda na demodulação (~2.5 dB)
2. **Assume coerência perfeita**: Correção Doppler ideal
3. **Simplifica cálculo de N0**: Ignora bandwidth do filtro de IF

**Decisão**: Manter método simplificado porque:
- Suficiente para detecção de anomalias relativas (não navegação absoluta)
- Baixo custo computacional
- Tendência consistente (aumento em power attacks)

**Valores típicos**:
- **Open-sky**: 40-50 dB-Hz
- **Indoor/obscured**: 20-35 dB-Hz
- **Spoofing power attack**: > 55 dB-Hz (suspeito)

### 4.2 Signal-to-Noise Ratio (SNR)

**Feature**: `snr_estimate`

**Diferença vs C/N0**:
- **SNR**: Razão de potências em bandwidth específico (adimensional ou dB)
- **C/N0**: Normalizado pela densidade de ruído (dB-Hz)

**Relação**:
```
C/N0 = SNR · BW_noise
```

**Uso no pipeline**: Feature adicional para modelos de ML (correlacionada com C/N0, mas comportamento diferente em alguns cenários).

---

## 5. Modelagem e Classificação

### 5.1 Escolha do Random Forest

**Justificativa Técnica**:

**Vantagens**:
1. **Robusto a outliers**: Decisões baseadas em votação de árvores
2. **Não-paramétrico**: Não assume distribuição dos dados
3. **Feature importance**: Interpretabilidade via Gini importance
4. **Balanceamento natural**: `class_weight='balanced'` ajusta pesos automaticamente

**Comparação com alternativas**:
- **SVM**: Mais lento em grandes datasets, requer normalização cuidadosa
- **MLP (Neural Network)**: Requer mais dados, prone to overfitting em datasets pequenos
- **Logistic Regression**: Assume relações lineares (features GPS são não-lineares)

**Parâmetros otimizados**:
```python
{
    'n_estimators': 100,      # Número de árvores (balanço bias-variance)
    'max_depth': 15,          # Profundidade (evita overfitting)
    'class_weight': 'balanced' # Ajusta para classes desbalanceadas
}
```

**Decisão**: Random Forest como baseline porque:
- Estado-da-arte em benchmarks de spoofing GPS
- Rápido treinamento e inferência
- Interpretável (importante para safety-critical applications)

### 5.2 Balanceamento de Classes: SMOTE vs Class Weight

**Problema**: Datasets reais têm mais amostras autênticas que spoofed (desbalanceamento).

**Opções implementadas**:

1. **Class Weight Balanced** (padrão):
   ```python
   w_class = N_samples / (n_classes × N_samples_class)
   ```
   - **Vantagem**: Simples, sem custo computacional
   - **Desvantagem**: Não cria novas amostras

2. **SMOTE (Synthetic Minority Over-sampling Technique)**:
   - Gera amostras sintéticas interpolando entre vizinhos
   - **Vantagem**: Aumenta diversidade do dataset
   - **Desvantagem**: Pode criar amostras irrealistas

**Decisão**: Oferecer ambas as opções (`use_smote` flag) porque:
- Class weight suficiente para datasets grandes
- SMOTE útil quando spoofing samples < 20-30

**Evidência empírica**: Em experimentos com TEXBAT, class weight balanced teve performance equivalente a SMOTE com menor tempo de treinamento.

### 5.3 Métricas de Avaliação

**Métricas implementadas**:
1. **Accuracy**: Fração de predições corretas
2. **Precision**: TP / (TP + FP) - Evitar falsos alarmes
3. **Recall (Sensitivity)**: TP / (TP + FN) - Detectar todos os spoofs
4. **F1 Score**: Média harmônica de precision e recall
5. **ROC AUC**: Área sob curva ROC (performance geral)
6. **False Alarm Rate**: FP / (FP + TN)

**Decisão - Priorizar Recall**: Em aplicações GPS safety-critical:
- **Custo de FN (miss detection)**: ALTO (navegação comprometida)
- **Custo de FP (false alarm)**: MÉDIO (backup systems podem ser ativados)

**Threshold tuning**: Implementado via `predict_proba()` - permite ajustar threshold de decisão para aumentar recall.

---

## 6. Validação e Testes

### 6.1 Estratificação

**Implementação**: `train_test_split(..., stratify=y)`

**Fundamento**: Garante que a proporção de classes em treino e teste seja preservada.

**Decisão**: Sempre estratificar porque:
- Datasets de spoofing são frequentemente desbalanceados
- Evita test sets não-representativos

### 6.2 Cross-Validation Estratificada

**Implementação**: `StratifiedKFold` com 5 folds

**Fundamento**: Combina:
- **K-fold**: Múltiplas partições treino/validação
- **Stratified**: Mantém proporção de classes em cada fold

**Decisão**: 5 folds (padrão sklearn) porque:
- Trade-off entre variância da estimativa e custo computacional
- 80/20 split em cada fold (razoável para datasets de 100-1000 amostras)

### 6.3 Reprodutibilidade

**Implementação**: `random_state=42` em todas as funções estocásticas

**Fundamento**: Permite:
- Debugging
- Comparação justa entre experimentos
- Validação por terceiros

**Decisão**: Fixar seeds porque:
- GPS spoofing detection requer validação rigorosa
- Resultados devem ser auditáveis

---

## 7. Limitações e Trabalho Futuro

### 7.1 Limitações Conhecidas

1. **C/N0 Estimation**: Método simplificado (erro ~3-5 dB vs métodos de referência)
2. **Single PRN**: Pipeline atual não trata múltiplos satélites simultaneamente
3. **Static Scenarios**: Não modela dinâmica temporal (receiver em movimento)
4. **Spoofing Types**: Foca em power/meaconing attacks; não cobre sophisticated replay attacks

### 7.2 Extensões Propostas

1. **Multi-PRN Fusion**: Combinar features de múltiplos satélites
2. **Temporal Features**: Derivadas temporais do C/N0, rate of change de SQMs
3. **Deep Learning**: LSTM/CNN para capturar padrões temporais complexos
4. **Transfer Learning**: Pré-treinar em datasets sintéticos, fine-tune em dados reais

### 7.3 Integração com Datasets Reais

**FGI-SpoofRepo**: Dataset finlandês com cenários outdoor/indoor
- Requer download manual (> 10 GB)
- Estrutura documentada em `data_loader.py`

**TEXBAT**: Dataset de referência com spoofing scenarios
- Ground truth baseado em timestamp (início do ataque conhecido)
- Usado para validação de publicações acadêmicas

---

## 8. Referências Técnicas

1. **GPS Signal Structure**: IS-GPS-200 (Interface Specification)
2. **C/A Code Generation**: "Understanding GPS Principles and Applications" - Kaplan & Hegarty
3. **Spoofing Detection**: IEEE papers on GNSS security (e.g., Psiaki et al., 2011)
4. **Random Forests**: Breiman, "Random Forests", Machine Learning, 2001
5. **SMOTE**: Chawla et al., "SMOTE: Synthetic Minority Over-sampling Technique", 2002

---

## 9. Glossário

- **C/A Code**: Coarse/Acquisition - código de espalhamento espectral do GPS civil
- **C/N0**: Carrier-to-Noise Density Ratio - métrica de qualidade de sinal
- **PRN**: Pseudo-Random Noise - identificador único do satélite (1-32 para GPS)
- **SQM**: Signal Quality Monitoring - métricas de integridade do sinal
- **TEXBAT**: Texas Spoofing Test Battery - dataset de referência
- **FWHM**: Full Width at Half Maximum - largura do pico à meia altura
- **Spoofing**: Ataque de transmissão de sinais GNSS falsos
- **Meaconing**: Retransmissão de sinal autêntico com delay

---

**Última atualização**: 2024-12-06  
**Autores**: Pipeline desenvolvida para ES413 (Cin/UFPE) - Projeto SinaisProject
