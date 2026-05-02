import numpy as np
import librosa


def extract_features(file_path: str) -> np.ndarray:

    # Load an audio file and return a 1-D NumPy array of 7 features.

    waveform, sr = librosa.load(file_path, sr=None)

    rms_frames   = librosa.feature.rms(y=waveform)[0]
    avg_rms      = float(np.mean(rms_frames))
    avg_zcr      = float(np.mean(librosa.feature.zero_crossing_rate(y=waveform)[0]))
    avg_centroid = float(np.mean(librosa.feature.spectral_centroid(y=waveform, sr=sr)[0]))
    avg_flatness = float(np.mean(librosa.feature.spectral_flatness(y=waveform)[0]))

    threshold        = np.percentile(rms_frames, 20)
    noise_frames     = rms_frames[rms_frames <= threshold]
    noise_floor_mean = float(np.mean(noise_frames))
    noise_floor_var  = float(np.var(noise_frames))
    near_zero_ratio  = float(np.sum(rms_frames < 1e-4) / len(rms_frames))

    return np.array([
        avg_rms,
        avg_zcr,
        avg_centroid,
        avg_flatness,
        noise_floor_mean,
        noise_floor_var,
        near_zero_ratio,
    ], dtype=np.float64)


# Human-readable names — used for feature importance display in the GUI
FEATURE_NAMES = [
    "avg_rms",
    "avg_zcr",
    "avg_centroid",
    "avg_flatness",
    "noise_floor_mean",
    "noise_floor_variance",
    "near_zero_ratio",
]
