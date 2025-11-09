import os
import numpy as np
import pandas as pd
import time as time_module
from typing import Tuple, Optional, Dict
from scipy.fft import fft, fftfreq

folder_path = r"Caminho/DO_ARQUIVO_BINARIO"
fs = 5e6
prn_to_track = 1                
ca_chip_rate = 1.023e6

segment_duration_s = 0.5
num_samples_per_segment = int(fs * segment_duration_s)
SPOOF_START_TIME_S = 17.0      
TOTAL_SAMPLES_IQ_S = int(60 * fs)

def read_iq_data(file_path, start_offset_samples, count_samples):
    
    bytes_per_iq_pair = 4
    start_offset_bytes = start_offset_samples * bytes_per_iq_pair
    count_int16 = 2 * count_samples 
    
    try:
        with open(file_path, "rb") as f:
            f.seek(start_offset_bytes)
            raw = np.fromfile(f, dtype=np.int16, count=2 * count_samples)
        
        if raw.size < count_int16:
             return None 
             
        I = raw[0::2].astype(np.float32)
        Q = raw[1::2].astype(np.float32)
        signal = I + 1j * Q
        return signal

    except Exception as e:
        return None

def generate_ca_code(prn_number):
    g2_shifts = {
        1: (2,6), 2: (3,7), 3: (4,8), 4: (5,9), 5: (1,9),
        6: (2,10), 7: (1,8), 8: (2,9), 9: (3,10), 10: (2,3),
        11: (3,4), 12: (5,6), 13: (6,7), 14: (7,8), 15: (8,9),
        16: (9,10), 17: (1,4), 18: (2,5), 19: (3,6), 20: (4,7),
        21: (5,8), 22: (6,9), 23: (1,3), 24: (4,6), 25: (5,7),
        26: (6,8), 27: (7,9), 28: (8,10), 29: (1,6), 30: (2,7),
        31: (3,8), 32: (4,9)
    }
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

def apply_frequency_correction(signal, fs, freq_correction): return signal

def normalize_by_power(signal):
    power = np.mean(np.abs(signal)**2)
    return signal / np.sqrt(power) if power > 1e-12 else signal

def generate_local_code_oversampled(prn_number: int, fs: float, samples_in_segment: int, ca_chip_rate: float = 1.023e6):
    ca = generate_ca_code(prn_number)
    samples_per_chip = int(fs // ca_chip_rate)
    local_code = np.repeat(ca, samples_per_chip)

    repeats = int(np.ceil(samples_in_segment / len(local_code)))
    local_code = np.tile(local_code, repeats)
    return local_code[:samples_in_segment]

def extract_correlation_sqms(corr_magnitude: np.ndarray, samples_per_chip: int) -> Dict[str, float]:
    peak_index = np.argmax(corr_magnitude)
    peak_value = corr_magnitude[peak_index]

    peak_window_samples = int(2 * samples_per_chip)
    temp_corr = corr_magnitude.copy()
    
    temp_corr = np.roll(temp_corr, -peak_index) 
    temp_corr[:peak_window_samples] = 0         
    temp_corr = np.roll(temp_corr, peak_index) 
    
    secondary_peak = np.max(temp_corr)
    peak_to_secondary = peak_value / secondary_peak if secondary_peak > 0 else 999.0

    frac_level = 0.8 * peak_value
    above_frac = np.where(corr_magnitude > frac_level)[0]
    fpw = above_frac[-1] - above_frac[0] if above_frac.size > 0 else 0

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
    total_power = np.mean(np.abs(signal_processed)**2)
    carrier_power_proxy = peak_value**2 / signal_processed.size
    noise_power_est = total_power - carrier_power_proxy
    if noise_power_est <= 0: noise_power_est = 1e-12
    
    C_N0_estimate = 10 * np.log10(carrier_power_proxy / (noise_power_est / fs))

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
    
    if segment_time_start < SPOOF_START_TIME_S:
        label = 0 
    else:
        label = 1 

    return signal, label


def run_feature_extraction_pipeline():

    features_df_all = pd.DataFrame()
    
    try:
        dat_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.bin', '.dat'))]
    except FileNotFoundError:
        dat_files = ['spoofer.bin'] 
        print("AVISO: Pasta não encontrada. Usando o nome de arquivo 'spoofer.bin' para teste de lógica.")
        
    if not dat_files:
        print(f"ERRO: Nenhuma base de dados encontrada.")
        return features_df_all

    print(f"Iniciando extração de features em {len(dat_files)} arquivo(s).")
    start_time_completo = time_module.time()

    for idx_file, filename in enumerate(dat_files):
        file_path = os.path.join(folder_path, filename)
        total_samples_iq = TOTAL_SAMPLES_IQ_S
        local_code = generate_local_code_oversampled(prn_to_track, fs, num_samples_per_segment)
        
        for segment_index in range(0, total_samples_iq, int(num_samples_per_segment / 2)):
            
            signal, label = load_and_label_segment(file_path, segment_index, num_samples_per_segment, fs)
            if signal is None or signal.size < num_samples_per_segment:
                continue
                
            signal_mixed = apply_frequency_correction(signal, fs, 0)
            signal_processed = normalize_by_power(signal_mixed)
            
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
    features_df_all = run_feature_extraction_pipeline()