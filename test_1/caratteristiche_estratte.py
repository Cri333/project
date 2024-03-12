import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
from scipy.signal import find_peaks, welch
import seaborn as sns


# funzioni

import scipy.stats as stats
import scipy.signal as signal
def create_resp_features(resp, band):
    frequencies, powers = signal.welch(
        x=resp,
        fs=1.0,
        window="hann",
        nperseg=int(len(resp)),
        noverlap=int(len(resp) / 2)
    )

    fbands = {'ulf': (0.0, 0.1), 'vlf': (0.1, 0.2), 'lf': (0.2, 0.3), 'hf': (0.3, 0.4)}

    return np.sum(
        powers[np.where(
            (frequencies >= fbands[band][0]) &
            (frequencies <= fbands[band][1])
        )]
    )


def create_gsr_features(gsr, return_type):
    peaks, _ = signal.find_peaks(gsr)
    widths, width_heights, left_lps, right_lps = signal.peak_widths(
        gsr,
        peaks,
        rel_height=1
    )

    if return_type == "frequency":
        return len(peaks)
    elif return_type == "magnitude":
        return sum(gsr[peaks] - width_heights)
    elif return_type == "duration":
        return sum(peaks - left_lps)
    elif return_type == "area":
        return sum(0.5 * (gsr[peaks] - width_heights) * (peaks - left_lps))


def create_hrv_feature(hr):
    periods = np.linspace(0.01, 0.5, 50)
    angular_frequencies = (2 * np.pi) / periods
    timestamp = np.linspace(1 / 15.5, len(hr) * (1 / 15.5), num=len(hr))

    try:
        lomb = signal.lombscargle(timestamp, hr, angular_frequencies, normalize=True)
        ratio = sum(lomb[0:8]) / sum(lomb[14:])
    except ZeroDivisionError:
        print("Failed to calculate Lomb, returning mean instead.")
        ratio = np.mean(hr)

    return ratio

def calculate_speed_features(group):
    array = group['Speed'].to_numpy()
    speed_change = array.max() - array.min()
    return pd.Series({
        'Speed_mean': array.mean(),
        'Speed_std': array.std(),
        'Speed_min': array.min(),
        'Speed_max': array.max(),
        'Speed_range': array.ptp(),
        'Speed_var': array.var(),
        'Speed_change': speed_change,
    })

def calculate_steering_features(group):
    array = group['Steering'].to_numpy()
    steering_change = array.max() - array.min()
    return pd.Series({
        'Steering_mean': array.mean(),
        'Steering_std': array.std(),
        'Steering_min': array.min(),
        'Steering_max': array.max(),
        'Steering_range': array.ptp(),
        'Steering_var': array.var(),
        'Steering_change': steering_change,
    })

def calculate_brake_features(group):
    array = group['Brake'].to_numpy()
    breaking_change = array.max() - array.min()
    breaking_freq = (array > array.mean()).sum()
    return pd.Series({
        'Breaking_sum': array.sum(),
        'Breaking_mean': array.mean(),
        'Breaking_std': array.std(),
        'Breaking_min': array.min(),
        'Breaking_max': array.max(),
        'Breaking_range': array.ptp(),
        'Breaking_var': array.var(),
        'Breaking_change': breaking_change,
        'Breaking_freq': breaking_freq,
    })

def calculate_acceleration_features(group):
    array = group['Acceleration'].to_numpy()
    jerk = np.diff(array) if len(array) > 1 else np.array([])
    return pd.Series({
        'Acceleration_mean': array.mean() if len(array) > 0 else np.nan,
        'Acceleration_std': array.std() if len(array) > 0 else np.nan,
        'Acceleration_min': array.min() if len(array) > 0 else np.nan,
        'Acceleration_max': array.max() if len(array) > 0 else np.nan,
        'Acceleration_range': array.ptp() if len(array) > 0 else np.nan,
        'Acceleration_var': array.var() if len(array) > 0 else np.nan,
        'Acceleration_Jerk_mean': jerk.mean() if len(jerk) > 0 else np.nan,
        'Acceleration_Jerk_std': jerk.std() if len(jerk) > 0 else np.nan,
        'Acceleration_Jerk_min': jerk.min() if len(jerk) > 0 else np.nan,
        'Acceleration_Jerk_max': jerk.max() if len(jerk) > 0 else np.nan,
        'Acceleration_Jerk_range': jerk.ptp() if len(jerk) > 0 else np.nan,
        'Acceleration_Jerk_var': jerk.var() if len(jerk) > 0 else np.nan,
    })

def calculate_heart_rate_features(group):
    hr_array = group['Heart.Rate'].to_numpy()
    fft = np.fft.rfft(hr_array)
    psd = np.abs(fft) ** 2
    peak_diffs = np.diff(signal.argrelmax(hr_array)[0])
    hr_diffs = np.diff(hr_array)
    return pd.Series({
        'Heart.Rate_ratio': np.mean(np.diff(hr_array) / hr_array[:-1]),
        'Heart.Rate_HRV': create_hrv_feature(hr_array),
        'Heart.Rate_mean': hr_array.mean(),
        'Heart.Rate_median': np.median(hr_array),
        'Heart.Rate_std': hr_array.std(),
        'Heart.Rate_var': hr_array.var(),
        'Heart.Rate_sum': hr_array.sum(),
        'Heart.Rate_range': hr_array.ptp(),
        'Heart.Rate_max': hr_array.max(),
        'Heart.Rate_min': hr_array.min(),
        'Heart.Rate_rms': np.sqrt(np.mean(hr_array**2)),
        'Heart.Rate_entropy': stats.entropy(hr_array),
        'Heart.Rate_iqr': stats.iqr(hr_array),
        'Heart.Rate_psd': psd.sum(),
        'Heart.Rate_psd_mean': psd.mean(),
        'Heart.Rate_psd_median': np.median(psd),
        'Heart.Rate_mean_peak_diff':  np.sqrt(np.mean(np.square(peak_diffs))) if len(peak_diffs) > 0 else np.nan,
        'Heart.Rate_std_peak_diff': peak_diffs.std() if len(peak_diffs) > 0 else np.nan,
        'Heart.Rate_num_peak_diff_gt_50': (peak_diffs > 50).sum() if len(peak_diffs) > 0 else np.nan,

        'Heart.Rate_mean_diff_sqrt': np.mean(np.sqrt(np.abs(hr_diffs))) if len(hr_diffs) > 0 else np.nan,
        'Heart.Rate_std_diff': hr_diffs.std() if len(hr_diffs) > 0 else np.nan,
        'Heart.Rate_num_diff_gt_50': (hr_diffs > 50).sum() if len(hr_diffs) > 0 else np.nan,
    })

def calculate_breathing_rate_features(group):
    br_array = group['Breathing.Rate'].to_numpy()
    fft = np.fft.rfft(br_array)
    psd = np.abs(fft) ** 2
    peak_diffs = np.diff(signal.argrelmax(br_array)[0])
    br_diffs = np.diff(br_array)
    return pd.Series({
        'Breathing.Rate_ratio': np.mean(np.diff(br_array) / br_array[:-1]),
        'Breathing.Rate_freq_very_low': create_resp_features(br_array, 'ulf'),
        'Breathing.Rate_freq_low': create_resp_features(br_array, 'vlf'),
        'Breathing.Rate_freq_high': create_resp_features(br_array, 'lf'),
        'Breathing.Rate_freq_very_high': create_resp_features(br_array, 'hf'),
        'Breathing.Rate_mean': br_array.mean(),
        'Breathing.Rate_median': np.median(br_array),
        'Breathing.Rate_std': br_array.std(),
        'Breathing.Rate_var': br_array.var(),
        'Breathing.Rate_sum': br_array.sum(),
        'Breathing.Rate_range': br_array.ptp(),
        'Breathing.Rate_max': br_array.max(),
        'Breathing.Rate_min': br_array.min(),
        'Breathing.Rate_rms': np.sqrt(np.mean(br_array**2)),
        'Breathing.Rate_entropy': stats.entropy(br_array),
        'Breathing.Rate_iqr': stats.iqr(br_array),
        'Breathing.Rate_psd': psd.sum(),
        'Breathing.Rate_psd_mean': psd.mean(),
        'Breathing.Rate_psd_median': np.median(psd),
        'Breathing.Rate_mean_peak_diff':  np.sqrt(np.mean(np.square(peak_diffs))) if len(peak_diffs) > 0 else np.nan,
        'Breathing.Rate_std_peak_diff': peak_diffs.std() if len(peak_diffs) > 0 else np.nan,
        'Breathing.Rate_num_peak_diff_gt_50': (peak_diffs > 50).sum() if len(peak_diffs) > 0 else np.nan,

        'Breathing.Rate_mean_diff_sqrt': np.mean(np.sqrt(np.abs(br_diffs))) if len(br_diffs) > 0 else np.nan,
        'Breathing.Rate_std_diff': br_diffs.std() if len(br_diffs) > 0 else np.nan,
        'Breathing.Rate_num_diff_gt_50': (br_diffs > 50).sum() if len(br_diffs) > 0 else np.nan,
    })

def calculate_perinasal_perspiration_features(group):
    pp_array = group['Perinasal.Perspiration'].to_numpy()
    fft = np.fft.rfft(pp_array)
    psd = np.abs(fft) ** 2
    peak_diffs = np.diff(signal.argrelmax(pp_array)[0])
    pp_diffs = np.diff(pp_array)
    return pd.Series({
        'Perinasal.Perspiration_mean': pp_array.mean(),
        'Perinasal.Perspiration_median': np.median(pp_array),
        'Perinasal.Perspiration_std': pp_array.std(),
        'Perinasal.Perspiration_var': pp_array.var(),
        'Perinasal.Perspiration_sum': pp_array.sum(),
        'Perinasal.Perspiration_range': pp_array.ptp(),
        'Perinasal.Perspiration_max': pp_array.max(),
        'Perinasal.Perspiration_min': pp_array.min(),
        'Perinasal.Perspiration_rms': np.sqrt(np.mean(pp_array**2)),
        'Perinasal.Perspiration_entropy': stats.entropy(pp_array),
        'Perinasal.Perspiration_iqr': stats.iqr(pp_array),
        'Perinasal.Perspiration_psd': psd.sum(),
        'Perinasal.Perspiration_psd_mean': psd.mean(),
        'Perinasal.Perspiration_psd_median': np.median(psd),
        'Perinasal.Perspiration_mean_peak_diff':  np.sqrt(np.mean(np.square(peak_diffs))) if len(peak_diffs) > 0 else np.nan,
        'Perinasal.Perspiration_std_peak_diff': peak_diffs.std() if len(peak_diffs) > 0 else np.nan,
        'Perinasal.Perspiration_num_peak_diff_gt_50': (peak_diffs > 50).sum() if len(peak_diffs) > 0 else np.nan,

        'Perinasal.Perspiration_mean_diff_sqrt': np.mean(np.sqrt(np.abs(pp_diffs))) if len(pp_diffs) > 0 else np.nan,
        'Perinasal.Perspiration_std_diff': pp_diffs.std() if len(pp_diffs) > 0 else np.nan,
        'Perinasal.Perspiration_num_diff_gt_50': (pp_diffs > 50).sum() if len(pp_diffs) > 0 else np.nan,
    })

import scipy.stats as stats
import scipy.signal as signal

def calculate_palm_eda_features(group):
    eda_array = group['Palm.EDA'].to_numpy()
    fft = np.fft.rfft(eda_array)
    psd = np.abs(fft) ** 2
    peak_diffs = np.diff(signal.argrelmax(eda_array)[0])
    eda_diffs = np.diff(eda_array)
    return pd.Series({
        'Palm.EDA_peak_count': create_gsr_features(eda_array, "frequency"),
        'Palm.EDA_peak_amplitude_mean': create_gsr_features(eda_array, "magnitude"),
        'Palm.EDA_peak_duration_mean': create_gsr_features(eda_array, "duration"),
        'Palm.EDA_peak_area_mean': create_gsr_features(eda_array, "area"),
        'Palm.EDA_mean': eda_array.mean(),
        'Palm.EDA_median': np.median(eda_array),
        'Palm.EDA_std': eda_array.std(),
        'Palm.EDA_var': eda_array.var(),
        'Palm.EDA_sum': eda_array.sum(),
        'Palm.EDA_range': eda_array.ptp(),
        'Palm.EDA_max': eda_array.max(),
        'Palm.EDA_min': eda_array.min(),
        'Palm.EDA_rms': np.sqrt(np.mean(eda_array**2)),
        'Palm.EDA_entropy': stats.entropy(eda_array),
        'Palm.EDA_iqr': stats.iqr(eda_array),
        'Palm.EDA_psd': psd.sum(),
        'Palm.EDA_psd_mean': psd.mean(),
        'Palm.EDA_psd_median': np.median(psd),
        'Palm.EDA_mean_peak_diff':  np.sqrt(np.mean(np.square(peak_diffs))) if len(peak_diffs) > 0 else np.nan,
        'Palm.EDA_std_peak_diff': peak_diffs.std() if len(peak_diffs) > 0 else np.nan,
        'Palm.EDA_num_peak_diff_gt_50': (peak_diffs > 50).sum() if len(peak_diffs) > 0 else np.nan,

        'Palm.EDA_mean_diff_sqrt': np.mean(np.sqrt(np.abs(eda_diffs))) if len(eda_diffs) > 0 else np.nan,
        'Palm.EDA_std_diff': eda_diffs.std() if len(eda_diffs) > 0 else np.nan,
        'Palm.EDA_num_diff_gt_50': (eda_diffs > 50).sum() if len(eda_diffs) > 0 else np.nan,
    })



def calculate_features(group):
    heart_rate_features = calculate_heart_rate_features(group)
    breathing_rate_features = calculate_breathing_rate_features(group)
    palm_eda_features = calculate_palm_eda_features(group)
    perinasal_perspiration_features = calculate_perinasal_perspiration_features(group)
    speed_features = calculate_speed_features(group)
    acceleration_features = calculate_acceleration_features(group)
    brake_features = calculate_brake_features(group)
    steering_features = calculate_steering_features(group)

    return pd.concat(
        [heart_rate_features, breathing_rate_features, palm_eda_features, perinasal_perspiration_features,
         speed_features, acceleration_features, brake_features, steering_features
         ])

df = pd.read_csv('df_sistema_valori_esperimento2.csv', dtype={14: str})
df['Group_ID'] = (df[['Drive']].diff().ne(0).any(axis=1)).cumsum()
df['Time_Reset'] = df.groupby('Group_ID')['Time'].transform(lambda x: x - x.min())
df['Time_Interval'] = (df['Time_Reset'] // 10).astype(int) * 10
df['Time_Interval'] = df['Time_Interval'].apply(
    lambda x: '{:02d}:{:02d}:{:02d}'.format(x // 3600, (x // 60) % 60, x % 60))
'''
df['Group_ID'] = (df[['Drive']].diff().ne(0).any(axis=1)).cumsum()
df['Time_Reset'] = df.groupby('Group_ID')['Time'].transform(lambda x: x - x.min())
df['Time_Start'] = (df['Time_Reset'] // 15).astype(int) * 15
df['Time_Start'] = df['Time_Start'].apply(
    lambda x: '{:02d}:{:02d}:{:02d}'.format(x // 3600, (x // 60) % 60, x % 60))
df['Time_End'] = ((df['Time_Reset'] // 15).astype(int) * 15) + 30
df['Time_End'] = df['Time_End'].apply(
    lambda x: '{:02d}:{:02d}:{:02d}'.format(x // 3600, (x // 60) % 60, x % 60))'''
gruppi = df.groupby(['driver', 'stress', 'Time_Interval'])

df_stats = gruppi.apply(calculate_features)
df_stats = df_stats.reset_index()
df_stats.to_csv('driver_stress_Time2.csv', index=False)