import os
import torch
import torchaudio
import torchaudio.transforms as T
from config import INTERIM_DIR, FEATURE_DIR, TARGET_SR, N_MELS, F_MAX, N_FFT, HOP_LENGTH

mel_spectrogram_transform = T.MelSpectrogram(
    sample_rate=TARGET_SR,
    n_mels=N_MELS,
    f_max=F_MAX,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH
)
amplitude_to_db_transform = T.AmplitudeToDB(stype='power', top_db=80)

compute_deltas = T.ComputeDeltas()


def save_spectrogram_torch(audio_path, output_path):
    try:
        waveform, sample_rate = torchaudio.load(audio_path)

        mel_spec = mel_spectrogram_transform(waveform)
        mel_spec_db = amplitude_to_db_transform(mel_spec)

        delta = compute_deltas(mel_spec_db)
        delta2 = compute_deltas(delta)

        features = torch.cat([mel_spec_db, delta, delta2], dim=0)

        torch.save(features, output_path)

    except Exception as e:
        print(f"Lỗi extract file {audio_path}: {e}")


if __name__ == "__main__":
    print("Bắt đầu trích xuất ...")
    extracted_count = 0
    for root, dirs, files in os.walk(INTERIM_DIR):
        for file in files:
            if file.lower().endswith('.wav'):
                audio_path = os.path.join(root, file)
                rel_path = os.path.relpath(root, INTERIM_DIR)
                output_folder = os.path.join(FEATURE_DIR, rel_path)
                os.makedirs(output_folder, exist_ok=True)

                filename_base = os.path.splitext(file)[0]
                output_path = os.path.join(output_folder, f"{filename_base}.pt")

                if not os.path.exists(output_path):
                    save_spectrogram_torch(audio_path, output_path)
                extracted_count += 1
    print("Hoàn tất!")
