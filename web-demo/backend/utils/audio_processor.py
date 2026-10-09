import torch
import torchaudio
import torchaudio.transforms as T
import torchvision.transforms as transforms

TARGET_SR = 22050
TARGET_SAMPLES = 88200
N_MELS = 128
F_MAX = 8000.0
N_FFT = 2048
HOP_LENGTH = 512

MEAN_3CH = [-25.0, 0.0, 0.0]
STD_3CH = [20.0, 20.0, 20.0]

mel_spectrogram_transform = T.MelSpectrogram(
    sample_rate=TARGET_SR,
    n_mels=N_MELS,
    f_max=F_MAX,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH
)
amplitude_to_db_transform = T.AmplitudeToDB(stype='power', top_db=80)
compute_deltas = T.ComputeDeltas()

normalize_transform = transforms.Normalize(mean=MEAN_3CH, std=STD_3CH)

def preprocess_audio(file_path):
    waveform, sr = torchaudio.load(file_path)

    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    if sr != TARGET_SR:
        resampler = T.Resample(orig_freq=sr, new_freq=TARGET_SR)
        waveform = resampler(waveform)

    num_samples = waveform.shape[1]
    if num_samples > TARGET_SAMPLES:
        waveform = waveform[:, :TARGET_SAMPLES]
    elif num_samples < TARGET_SAMPLES:
        num_repeats = (TARGET_SAMPLES // num_samples) + 1
        waveform_repeated = waveform.repeat(1, num_repeats)
        waveform = waveform_repeated[:, :TARGET_SAMPLES]

    mel_spec = mel_spectrogram_transform(waveform)
    mel_spec_db = amplitude_to_db_transform(mel_spec)

    delta = compute_deltas(mel_spec_db)
    delta2 = compute_deltas(delta)

    features = torch.cat([mel_spec_db, delta, delta2], dim=0)

    features_normalized = normalize_transform(features)

    tensor_features = features_normalized.unsqueeze(0)

    return tensor_features
