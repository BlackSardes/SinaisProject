# II - Extração de Características de Sinais GNSS (SQMs e Potência)

Este módulo implementa a extração de características críticas focadas na **Integridade do Sinal e Análise da Correlação**, conforme exigido para a detecção de *spoofing* em receptores GPS.

## 📊 Classificação por Necessidade

As classificações refletem a criticidade das funções para o projeto:
**🔴 CRÍTICAS**: `extract_correlation_sqms`, `extract_power_metrics` (São as features do projeto)
**🟡 IMPORTANTES**: `generate_local_code_oversampled`, `load_and_label_segment` (Preparam o dado para a extração)
**🟢 ÚTEIS**: `run_feature_extraction_pipeline` (Gerencia a produção do dataset)

---

## 🔍 Justificativa das Features (Conexão Sinais/Sistemas)

A detecção de *spoofing* reside na análise das **Métricas de Monitoramento da Qualidade do Sinal (SQMs)** e das métricas de potência. O ataque se manifesta como uma distorção na forma ou um aumento não natural na energia do pico de correlação.

| Grupo de Feature | Feature Escolhida | Relação com Sinais e Sistemas | Impacto do Spoofing |
| :--- | :--- | :--- | :--- |
| **SQMs (Morfologia)** | **`sqm_peak_to_secondary` (P/S Ratio)** | Reflete a **Ortogonalidade** e a pureza da Função de Autocorrelação (ACF). | **Diminui drasticamente** devido à criação de "ombros" ou picos secundários. |
| **SQMs (Morfologia)** | **`sqm_asymmetry`** | Quantifica a **Inclinação** da ACF. Idealmente zero. | Ataques sincronizados causam uma **distorção assimétrica** no pico. |
| **SQMs (Morfologia)** | **`sqm_fpw` (Fractional Peak Width)** | Mede a **Largura do Lóbulo Principal** da ACF. | **Aumenta** se o receptor estiver rastreando múltiplos sinais ligeiramente defasados (pico alargado). |
| **Potência** | **`power_c_n0`** | Relação entre a **Potência da Portadora (C)** e a **Densidade Espectral de Ruído ($N_0$)**. | **Aumenta abruptamente** na maioria dos ataques de alta potência. |
| **Potência** | **`power_noise_floor`** | Estimativa da densidade espectral de ruído ($N_0$). | Variações indicam se o ataque introduziu ruído de banda larga. |

---

## Funções Principais

### 🔴
#### ➡️ `extract_correlation_sqms(corr_magnitude: np.ndarray, samples_per_chip: int) -> Dict[str, float]`

**Descrição**: Extrai as Métricas de Monitoramento da Qualidade do Sinal (SQMs) do perfil de magnitude da função de correlação (ACF). Essas métricas são a **base morfológica** para a classificação.

**Necessidade**: **CRÍTICA** - As distorções do pico de correlação são a manifestação física do ataque.

**Características Extraídas**:
- `sqm_peak_value`
- `sqm_peak_to_secondary` (P/S Ratio)
- `sqm_fpw` (Fractional Peak Width)
- `sqm_asymmetry`
- `sqm_secondary_peak_value`

---

#### ➡️ `extract_power_metrics(signal_processed: np.ndarray, peak_value: float, secondary_peak_value: float, fs: float) -> Dict[str, float]`

**Descrição**: Extrai métricas relacionadas à potência e ao ruído do sinal.

**Necessidade**: **CRÍTICA** - O ataque de *spoofing* geralmente eleva a potência do sinal (Power Attack) ou introduz ruído.

**Características Extraídas**:
- `power_c_n0` (Carrier-to-Noise Density Ratio)
- `power_noise_floor` (Potência do Ruído de Fundo)
- `power_mean_real` (Média da parte real do sinal I/Q)
- `power_std_amplitude` (Desvio padrão da amplitude do sinal)

---

### 🟡
#### ➡️ `generate_local_code_oversampled(prn_number: int, fs: float, samples_in_segment: int, ca_chip_rate: float = 1.023e6) -> np.ndarray`

**Descrição**: Gera o código PRN local (referência) reamostrado, essencial para a correlação (Módulo I).

**Necessidade**: **IMPORTANTE** - Fornece o código de referência com a taxa de amostragem correta para a FFT/Correlação.

---

#### ➡️ `load_and_label_segment(file_path: str, segment_index: int, segment_size: int, fs: float) -> Tuple[Optional[np.ndarray], int]`

**Descrição**: Função utilitária que carrega um segmento I/Q e aplica a lógica de rotulagem do TEXBAT baseada no tempo do segmento.

**Necessidade**: **IMPORTANTE** - Modulariza a complexa lógica de leitura de arquivos grandes e a rotulagem de tempo ("antes do 150s" = Autêntico).

---

### 🟢
#### ➡️ `run_feature_extraction_pipeline()`

**Descrição**: Função de produção que gerencia o loop sobre todos os arquivos e segmentos do dataset, aplicando as etapas de Pré-processamento (Módulo I) e Extração (Módulo II) e construindo o DataFrame final de características.

**Necessidade**: **ÚTIL** - Garante a rastreabilidade e a execução em lote do processo, com a ordem correta das chamadas de função.

---