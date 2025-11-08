# Módulo de Pré-processamento de Sinais GNSS (GPS)

Este módulo contém funções essenciais para o pré-processamento de sinais GPS em Banda Base (formato I/Q complexo), especificamente otimizado para mitigar interferências e garantir coerência para a etapa de correlação.

## 📊 Classificação por Necessidade

**🔴 CRÍTICAS**: `read_iq_data`, `apply_frequency_correction`, `normalize_by_power`
**🟡 IMPORTANTES**: `apply_pulse_blanking`, `apply_fdpb_filter`, `generate_ca_code`
**🟢 ÚTEIS**: `pipeline_preprocessamento_segmento`

-----

## Funções Principais

### 🔴

#### ➡️ `read_iq_data(filepath: str, start_offset_samples: int, count_samples: int) -> Optional[np.ndarray]`

**Descrição**: Carrega um segmento de dados I/Q binários (int16 intercalados) de um arquivo RAW (como os do TEXBAT).

**Necessidade**: **CRÍTICA** - O formato binário I/Q não é suportado nativamente pelo NumPy para leitura segmentada e complexa.

**Parâmetros**:

  - `filepath`: Caminho para o arquivo binário (`.bin` ou `.dat`).
  - `start_offset_samples`: Posição inicial (em amostras complexas) para começar a leitura (para janelamento).
  - `count_samples`: Número de amostras complexas (I+jQ) a serem lidas.

**Retorna**:

  - Array numpy complexo (`float32`) com os dados I/Q, ou `None` se a leitura falhar.

**Exemplo de uso**:

```python
# Lê 0.5s de dados (12.5 milhões de amostras a 25MHz)
fs = 25e6
num_samples = int(0.5 * fs)
signal = read_iq_data('ds1.bin', 0, num_samples)
print(f"Tipo do sinal: {signal.dtype}") # complex64
```

**Observações**:

  - Essencial para o **janelamento** (leitura por segmentos) em grandes datasets como o TEXBAT.
  - `np.int16` são convertidos para `np.float32` para cálculos de ponto flutuante.
  - Lida com I e Q intercalados (`I, Q, I, Q, ...`) e reconstrói o sinal complexo (`I + 1j * Q`).

-----

#### ➡️ `apply_frequency_correction(signal: np.ndarray, fs: float, freq_correction: float) -> np.ndarray`

**Descrição**: Corrige o desvio de frequência Doppler e a Frequência Intermediária (IF) do sinal.

**Necessidade**: **CRÍTICA** - Sem coerência de frequência, o pico de correlação no domínio do tempo é nulo.

**Conceito (Sinais e Sistemas)**: Aplica a **Propriedade da Modulação (Shifting Property)** no domínio do tempo, multiplicando o sinal por um oscilador complexo de frequência negativa ($\mathbf{e}^{-j 2 \pi f_{corr} t}$).

**Parâmetros**:

  - `signal`: Array numpy complexo.
  - `fs`: Frequência de amostragem em Hz.
  - `freq_correction`: Frequência total a ser removida (IF + Doppler estimado).

**Retorna**:

  - Sinal corrigido (coerente).

**Validações Implementadas**:

  - **Consistência de fase**: Garante que o vetor de tempo seja calculado corretamente para o tamanho do segmento.

**Exemplo de uso**:

```python
# Corrige o sinal para 0 Hz (assumindo IF+Doppler = 0)
signal_coherent = apply_frequency_correction(signal, fs=25e6, freq_correction=0e6) 

# Em um cenário real, freq_correction seria o resultado de uma busca.
```

-----

#### ➡️ `normalize_by_power(signal: np.ndarray) -> np.ndarray`

**Descrição**: Normaliza o sinal de forma que sua potência média ($\mathbf{E}[|x|^2]$) seja aproximadamente 1 (ou 0 dB).

**Necessidade**: **CRÍTICA** - Variações de ganho de hardware afetam a amplitude absoluta.

**Conceito (Sinais e Sistemas)**: Padronização da energia média do sinal.

**Por que é crucial**:

  - O modelo de ML não deve confundir uma variação de ganho do receptor com uma anomalia causada por *spoofing*.
  - Essencial para o cálculo preciso da métrica **C/N0** (Carrier-to-Noise Density Ratio), que é uma *feature* chave.

**Parâmetros**:

  - `signal`: Array numpy complexo.

**Retorna**:

  - Sinal normalizado.

**Proteções Implementadas**:

  - Proteção contra divisão por zero (`power > 1e-12`).

**Exemplo de uso**:

```python
signal_norm = normalize_by_power(signal_coherent)
# np.mean(np.abs(signal_norm)**2) será ≈ 1
```

-----

### 🟡

#### ➡️ `apply_pulse_blanking(signal: np.ndarray, threshold_factor: float = 4.0) -> np.ndarray`

**Descrição**: Mitigação de Interferência Pulsada (PB), limitando a amplitude de picos de alta energia no domínio do tempo.

**Necessidade**: **IMPORTANTE** - Picos de ruído alteram as estatísticas do sinal e degradam o C/N0.

**Conceito (Sinais e Sistemas)**: Processamento não-linear, atuando como um "limiter".

**Tipos mitigados**: Interferências impulsivas (p. ex., radar, fontes de energia comutadas).

**Parâmetros**:

  - `signal`: Sinal complexo.
  - `threshold_factor`: Limite do PB (padrão: 4.0 \* desvio padrão da amplitude).

**Retorna**:

  - Sinal com os pulsos suprimidos (limitados).

-----

#### ➡️ `apply_fdpb_filter(signal: np.ndarray, threshold_factor: float = 3.5) -> np.ndarray`

**Descrição**: Mitigação de Interferência no Domínio da Frequência (FDPB), suprimindo componentes espectrais anômalos (RFI de banda estreita).

**Necessidade**: **IMPORTANTE** - RFI de banda estreita aumenta o chão de ruído e distorce o pico de correlação.

**Conceito (Sinais e Sistemas)**: Filtragem adaptativa no domínio da frequência (FFT/IFFT). Utiliza o **MAD (Median Absolute Deviation)** para um limiar de ruído robusto.

**Parâmetros**:

  - `signal`: Sinal complexo.
  - `threshold_factor`: Fator multiplicador para o limiar espectral robusto.

**Retorna**:

  - Sinal sem as componentes de frequência de interferência suprimidas.

-----

#### ➡️ `generate_ca_code(prn_number: int) -> np.ndarray`

**Descrição**: Gera o código C/A (Code-Acquisition) Gold Sequence para o satélite PRN especificado.

**Necessidade**: **IMPORTANTE** - A sequência de código é o **sinal local** que será correlacionado com o sinal recebido na próxima etapa.

**Conceito (Sinais e Sistemas)**: Geração de Sequências Pseudo-Aleatórias (PN) e Códigos Ortogonais.

**Parâmetros**:

  - `prn_number`: Número do Satélite (1 a 32).

**Retorna**:

  - Array numpy (1023 chips) com valores +1 ou -1.

-----

### 🟢

#### ➡️ `pipeline_preprocessamento_segmento(file_path: str, segment_index: int, fs: float, prn: int, ...) -> Optional[np.ndarray]`

**Descrição**: Sequência completa e otimizada de pré-processamento para um único segmento de sinal.

**Necessidade**: **ÚTIL** - Garante a ordem correta e reprodutível das operações para o *loop* principal de extração de características.

**Ordem Otimizada de Processamento**:

1.  **Leitura do Segmento** (`read_iq_data`)
2.  **Correção de Frequência** (`apply_frequency_correction`)
3.  **Mitigação de Pulso** (`apply_pulse_blanking`)
4.  **Mitigação de RFI (FDPB)** (`apply_fdpb_filter`)
5.  **Normalização de Potência** (`normalize_by_power`)

**Exemplo de uso**:

```python
# Esta função agrupa todas as etapas 2 a 5 para uso no loop da Célula II.
signal_processed = pipeline_preprocessamento_segmento(
    filepath, 
    start_offset_samples, 
    num_samples, 
    fs, 
    center_freq, 
    test_doppler_freq
)
```