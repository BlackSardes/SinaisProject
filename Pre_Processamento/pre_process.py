## 1. PRÉ-PROCESSAMENTO (LEITURA, FILTRAGEM RFI E NORMALIZAÇÃO)

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate, iirfilter, lfilter, butter
import pandas as pd
from scipy.fft import fft, fftfreq

# --- PARÂMETROS GLOBAIS DO TEXBAT ---
folder_path = r"Caminho/TEXBAT"  # MUDAR: Pasta com ds1.bin, ds2.bin, etc.
fs = 25e6                      # Taxa de amostragem em Hz (25 MHz para TEXBAT)
prn_to_track = 1                # Focar no PRN 1
center_freq = 0e6               # Frequência central (IF)
test_doppler_freq = 0           # Início da busca Doppler (0 Hz para teste)
ca_chip_rate = 1.023e6

# --- Janelamento de Dados ---
segment_duration_s = 0.5        # Duração de cada segmento (janela) em segundos
num_samples_per_segment = int(fs * segment_duration_s)

# --- MAPA DE ROTULAGEM (LABELING) DO TEXBAT (SIMPLIFICADO) ---
SPOOF_START_TIME_S = 150.0

# --- FUNÇÕES DE PRÉ-PROCESSAMENTO AVANÇADO ---

def apply_notch_filter(signal, fs, f0, Q):
    """
    Aplica um filtro Notch IIR (Band-stop) para remover interferência de banda estreita (RFI).
    
    Args:
        signal (np.array): Sinal complexo (I + 1j*Q).
        fs (float): Frequência de amostragem.
        f0 (float): Frequência central do notch (onde a RFI está).
        Q (float): Fator de Qualidade (define a largura da banda de corte).
    
    Returns:
        np.array: Sinal filtrado.
    """
    # Projeta o filtro IIR de segunda ordem (ordem=2)
    b, a = iirfilter(2, [f0 - f0/(2*Q), f0 + f0/(2*Q)], rs=60, btype='bandstop', fs=fs, output='ba')
    
    # Aplica o filtro separadamente em I e Q
    I_filtered = lfilter(b, a, np.real(signal))
    Q_filtered = lfilter(b, a, np.imag(signal))
    
    return I_filtered + 1j * Q_filtered

def normalize_by_power(signal):
    """Normaliza o sinal de forma que sua potência média seja 1 (ou 0 dB)."""
    power = np.mean(np.abs(signal)**2)
    if power > 1e-12: # Evita divisão por zero
        return signal / np.sqrt(power)
    return signal

# Função de Leitura (Mantida)
def read_iq_data(file_path, start_offset_samples, count_samples):
    """Lê dados IQ de um arquivo binário de int16, com offset (para janelamento)."""
    bytes_per_iq_pair = 4 
    start_offset_bytes = start_offset_samples * bytes_per_iq_pair
    count_int16 = 2 * count_samples
    
    try:
        with open(file_path, "rb") as f:
            f.seek(start_offset_bytes)
            raw = np.fromfile(f, dtype=np.int16, count=count_int16)
        
        if raw.size < count_int16:
             return None 
             
        I = raw[0::2].astype(np.float32)
        Q = raw[1::2].astype(np.float32)
        signal = I + 1j * Q
        return signal

    except Exception as e:
        print(f"Aviso: Erro ao ler segmento do arquivo {os.path.basename(file_path)}: {e}")
        return None

def generate_ca_code(prn_number):
    # ... (Sua função de geração de código C/A - Mantida) ...
    g2_shifts = {
        1: (2,6), 2: (3,7), 3: (4,8), 4: (5,9), 5: (1,9),
        6: (2,10), 7: (1,8), 8: (2,9), 9: (3,10), 10: (2,3),
        11: (3,4), 12: (5,6), 13: (6,7), 14: (7,8), 15: (8,9),
        16: (9,10), 17: (1,4), 18: (2,5), 19: (3,6), 20: (4,7),
        21: (5,8), 22: (6,9), 23: (1,3), 24: (4,6), 25: (5,7),
        26: (6,8), 27: (7,9), 28: (8,10), 29: (1,6), 30: (2,7),
        31: (3,8), 32: (4,9)
    }
    if prn_number not in g2_shifts:
        raise ValueError("PRN fora do intervalo (1-32) neste gerador simplificado.")
    s1, s2 = g2_shifts[prn_number]
    g1 = np.ones(10, dtype=int)
    g2 = np.ones(10, dtype=int)
    ca = np.zeros(1023, dtype=int)
    for i in range(1023):
        ca[i] = (g1[-1] ^ (g2[-s1] ^ g2[-s2]))
        new_g1 = g1[2] ^ g1[9]
        new_g2 = g2[1] ^ g2[2] ^ g2[5] ^ g2[7] ^ g2[8] ^ g2[9]
        g1 = np.roll(g1, 1); g1[0] = new_g1
        g2 = np.roll(g2, 1); g2[0] = new_g2
    return 1 - 2*ca
    
# --- Demostração de Pré-Processamento no Primeiro Segmento (Visualização) ---

# Tente carregar o primeiro segmento do primeiro arquivo
dat_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.bin', '.dat'))]
if not dat_files:
     raise SystemExit("Nenhum arquivo de dados TEXBAT (.bin ou .dat) encontrado.")
selected_file = os.path.join(folder_path, dat_files[0])
signal_original = read_iq_data(selected_file, 0, num_samples_per_segment)

if signal_original is not None:
    # 1. Correção de Frequência (Beating Frequency Removal)
    t = np.arange(signal_original.size) / fs
    mixer = np.exp(-1j * 2 * np.pi * (center_freq + test_doppler_freq) * t)
    signal_mixed = signal_original * mixer
    
    # 2. Filtragem RFI (Exemplo: Filtro Notch em 1 MHz)
    # RFI é comum no GNSS, especialmente se a banda for gravada.
    f_rfi = 1e6 # Hipótese de uma RFI em 1 MHz
    print(f" Aplicando Filtro Notch em {f_rfi/1e6:.1f} MHz para suprimir RFI...")
    signal_filtered = apply_notch_filter(signal_mixed, fs, f_rfi, Q=30)
    
    # 3. Normalização de Potência
    signal_processed = normalize_by_power(signal_filtered)
    
    print("\n--- Visualização do Pré-Processamento ---")
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Pré-Processamento: Espectro e Normalização', fontsize=16, fontweight='bold')
    
    # Espectro Original (Após Mixagem)
    freqs = fftfreq(len(signal_mixed), 1/fs)
    fft_orig = np.abs(fft(signal_mixed))
    mask = freqs > 0
    axes[0,0].semilogy(freqs[mask]/1e6, fft_orig[mask], 'b-', linewidth=0.8)
    axes[0,0].set_title('Espectro - Sinal Misturado (Original)')
    axes[0,0].set_xlabel('Frequência (MHz)'); axes[0,0].set_ylabel('Magnitude')
    axes[0,0].grid(True, alpha=0.3)
    
    # Espectro Filtrado (Após Notch)
    fft_filt = np.abs(fft(signal_filtered))
    axes[0,1].semilogy(freqs[mask]/1e6, fft_filt[mask], 'g-', linewidth=0.8)
    axes[0,1].set_title(f'Espectro - Sinal Filtrado (Notch em {f_rfi/1e6:.1f} MHz)')
    axes[0,1].set_xlabel('Frequência (MHz)'); axes[0,1].set_ylabel('Magnitude')
    axes[0,1].grid(True, alpha=0.3)

    # Distribuição Antes e Depois da Normalização
    axes[1,0].hist(np.real(signal_filtered), bins=50, alpha=0.7, color='blue', density=True, label='Antes (Re)')
    axes[1,0].hist(np.real(signal_processed), bins=50, alpha=0.7, color='red', density=True, label='Depois (Re)')
    axes[1,0].set_title('Normalização (Distribuição Real)')
    axes[1,0].set_xlabel('Amplitude'); axes[1,0].set_ylabel('Densidade')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Estatísticas de Potência
    P_filt = np.mean(np.abs(signal_filtered)**2)
    P_proc = np.mean(np.abs(signal_processed)**2)
    axes[1,1].bar(['Antes (P={:.2e})'.format(P_filt), 'Depois (P={:.2e})'.format(P_proc)], [P_filt, P_proc], color=['blue', 'red'])
    axes[1,1].set_title('Potência Média do Sinal')
    axes[1,1].set_ylabel('Potência')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()