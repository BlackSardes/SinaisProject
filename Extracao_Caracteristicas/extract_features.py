import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import correlate
from scipy.fft import fft, fftfreq
import time as time_module
from typing import Tuple, Optional, Dict, Any

folder_path = r"Caminho/TEXBAT"  # caminho do arquivo


fs = 25e6                      # taxa de amostragem em Hz
prn_to_track = 1                # satélite PRN a ser analisado
ca_chip_rate = 1.023e6          # taxa do código C/A

# janelamento
segment_duration_s = 0.5        # duração de cada segmento (janela) em segundos
num_samples_per_segment = int(fs * segment_duration_s)
SPOOF_START_TIME_S = 150.0      # tempo de corte para rotulagem 

def read_iq_data(file_path, start_offset_samples, count_samples):
    
    if not os.path.exists(file_path) or count_samples <= 0:
        return None
    
    return np.random.randn(count_samples) + 1j * np.random.randn(count_samples)

def generate_ca_code(prn_number):
    
    return 1 - 2 * np.random.randint(0, 2, 1023)

def apply_frequency_correction(signal, fs, freq_correction): return signal
def apply_pulse_blanking(signal): return signal
def apply_fdpb_filter(signal): return signal
def normalize_by_power(signal):
    power = np.mean(np.abs(signal)**2)
    return signal / np.sqrt(power) if power > 1e-12 else signal

def generate_local_code_oversampled(prn_number: int, fs: float, samples_in_segment: int, ca_chip_rate: float = 1.023e6) -> np.ndarray:
    
    ca_code = generate_ca_code(prn_number)
    samples_per_chip = round(fs / ca_chip_rate)
    replicated_ca_full = np.repeat(ca_code, samples_per_chip)
    return replicated_ca_full[:samples_in_segment]

def extract_correlation_sqms(corr_magnitude: np.ndarray, samples_per_chip: int) -> Dict[str, float]:
    
    peak_index = np.argmax(corr_magnitude)
    peak_value = corr_magnitude[peak_index]

    # 1. Razão P/S (Peak-to-Secondary Ratio)
    peak_window_samples = int(2 * samples_per_chip)
    temp_corr = corr_magnitude.copy()
    
    temp_corr = np.roll(temp_corr, -peak_index) 
    temp_corr[:peak_window_samples] = 0         
    temp_corr = np.roll(temp_corr, peak_index) 
    
    secondary_peak = np.max(temp_corr)
    peak_to_secondary = peak_value / secondary_peak if secondary_peak > 0 else 999.0

    # 2. Largura Fracionária (FPW) - a 80% da altura
    frac_level = 0.8 * peak_value
    above_frac = np.where(corr_magnitude > frac_level)[0]
    fpw = above_frac[-1] - above_frac[0] if above_frac.size > 0 else 0

    # 3. Assimetria (Area-based Asymmetry)
    left_area = np.sum(corr_magnitude[peak_index - samples_per_chip: peak_index])
    right_area = np.sum(corr_magnitude[peak_index + 1: peak_index + samples_per_chip + 1])
    asymmetry = (right_area - left_area) / (right_area + left_area) if (right_area + left_area) != 0 else 0.0

    return {
        "sqm_peak_value": float(peak_value),
        "sqm_peak_to_secondary": peak_to_secondary,
        "sqm_fpw": fpw,
        "sqm_asymmetry": float(asymmetry),
        "sqm_secondary_peak_value": float(secondary_peak)
    }

def extract_power_metrics(signal_processed: np.ndarray, peak_value: float, secondary_peak_value: float, fs: float) -> Dict[str, float]:
    
    # 1. Potência do Ruído (Noise Floor Power)
    total_power = np.mean(np.abs(signal_processed)**2)
    carrier_power_proxy = peak_value**2 / signal_processed.size
    noise_power_est = total_power - carrier_power_proxy
    if noise_power_est <= 0: noise_power_est = 1e-12
    
    # 2. C/N0 (Carrier-to-Noise Density Ratio)
    C_N0_estimate = 10 * np.log10(carrier_power_proxy / (noise_power_est / fs))

    # 3. Estatísticas Descritivas
    mean_real = np.mean(np.real(signal_processed))
    std_amplitude = np.std(np.abs(signal_processed))

    return {
        "power_c_n0": C_N0_estimate,
        "power_noise_floor": noise_power_est,
        "power_mean_real": float(mean_real),
        "power_std_amplitude": float(std_amplitude)
    }

def load_and_label_segment(file_path: str, segment_index: int, segment_size: int, fs: float) -> Tuple[Optional[np.ndarray], int]:
    
    signal = read_iq_data(file_path, segment_index, segment_size)
    
    segment_time_start = segment_index / fs
    filename = os.path.basename(file_path).lower()
    
    if 'cleanstatic' in filename:
        label = 0 # 0 = Autêntico
    elif segment_time_start < SPOOF_START_TIME_S:
        label = 0 # 0 = Autêntico (antes do spoofing começar)
    else:
        label = 1 # 1 = Spoofed (após o spoofing começar)

    return signal, label


def run_feature_extraction_pipeline():

    features_df_all = pd.DataFrame()
    
    try:
        dat_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.bin', '.dat'))]
    except FileNotFoundError:
        dat_files = ['cleanstatic.bin', 'ds1.bin', 'ds1.bin'] 
        print("AVISO: Pasta não encontrada. Usando nomes de arquivo MOCK para teste de lógica.")
        
    if not dat_files:
        print(f"ERRO: Nenhuma base de dados encontrada.")
        return features_df_all

    print(f"Iniciando extração de features em {len(dat_files)} arquivos.")
    start_time_completo = time_module.time()

    for idx_file, filename in enumerate(dat_files):
        file_path = os.path.join(folder_path, filename)
        
        total_samples_iq_simulado = int(420 * fs) 
        
        local_code = generate_local_code_oversampled(prn_to_track, fs, num_samples_per_segment)
        
        for segment_index in range(0, total_samples_iq_simulado, num_samples_per_segment):
            
            signal, label = load_and_label_segment(file_path, segment_index, num_samples_per_segment, fs)
            if signal is None or signal.size < num_samples_per_segment:
                continue
                
            signal_mixed = apply_frequency_correction(signal, fs, 0)
            signal_processed = normalize_by_power(signal_mixed) # Skip PB/FDPB mocks
            
            fft_signal = np.fft.fft(signal_processed)
            fft_code = np.fft.fft(local_code)
            corr_fft = fft_signal * np.conj(fft_code)
            corr_magnitude = np.abs(np.fft.ifft(corr_fft))
            
            samples_per_chip = round(fs / ca_chip_rate)
            sqm_features = extract_correlation_sqms(corr_magnitude, samples_per_chip)
            power_features = extract_power_metrics(signal_processed, sqm_features['sqm_peak_value'], sqm_features['sqm_secondary_peak_value'], fs)
            
            features_completas = {
                "filename": filename,
                "segment_start_s": segment_index / fs,
                "prn": prn_to_track,
                "label": label
            }

            features_completas.update(sqm_features)
            features_completas.update(power_features)
            
            new_row = pd.DataFrame([features_completas])
            features_df_all = pd.concat([features_df_all, new_row], ignore_index=True)
            
            if len(features_df_all) > 50 and filename != 'cleanstatic.bin': break

        print(f"  Arquivo {idx_file+1}/{len(dat_files)} ({filename}) processado. Total de features: {len(features_df_all)}")
        
    tempo_total = time_module.time() - start_time_completo

    print("\n" + "="*70)
    print("MÓDULO II: EXTRAÇÃO DE CARACTERÍSTICAS CONCLUÍDA")
    print(f"Tempo Total de Execução: {tempo_total:.2f} segundos.")
    print(f"Total de Registros (Segmentos): {len(features_df_all)}")

    if not features_df_all.empty:
        print("\nAMOSTRA DO DATASET DE FEATURES:")
        print(features_df_all.head().to_string())
    
    return features_df_all

if __name__ == '__main__':
    df_results = run_feature_extraction_pipeline()