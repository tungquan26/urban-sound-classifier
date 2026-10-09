import os
import torch
import torchaudio
import torchaudio.transforms as T
from config import AUDIO_DIR, INTERIM_DIR, TARGET_SR, TARGET_SAMPLES


def pad_or_truncate_torch(waveform):
    num_samples = waveform.shape[1]

    if num_samples > TARGET_SAMPLES:
        return waveform[:, :TARGET_SAMPLES]

    elif num_samples < TARGET_SAMPLES:
        num_repeats = (TARGET_SAMPLES // num_samples) + 1
        waveform_repeated = waveform.repeat(1, num_repeats)
        return waveform_repeated[:, :TARGET_SAMPLES]

    return waveform


def process_audio(audio_path, output_folder, filename_base):
    try:
        waveform, sr = torchaudio.load(audio_path)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        if sr != TARGET_SR:
            resampler = T.Resample(orig_freq=sr, new_freq=TARGET_SR)
            waveform = resampler(waveform)

        waveform_processed = pad_or_truncate_torch(waveform)

        orig_path = os.path.join(output_folder, f"{filename_base}.wav")
        if not os.path.exists(orig_path):
            torchaudio.save(orig_path, waveform_processed, TARGET_SR)

    except Exception as e:
        print(f"Lỗi preprocess file {audio_path}: {e}")


if __name__ == "__main__":
    print("Bắt đầu preprocess")
    processed_count = 0
    for root, dirs, files in os.walk(AUDIO_DIR):
        for file in files:
            if file.lower().endswith(('.wav', '.mp3', '.ogg', '.flac')):
                audio_path = os.path.join(root, file)
                rel_path = os.path.relpath(root, AUDIO_DIR)
                output_folder = os.path.join(INTERIM_DIR, rel_path)
                os.makedirs(output_folder, exist_ok=True)

                filename_base = os.path.splitext(file)[0]
                process_audio(audio_path, output_folder, filename_base)

                processed_count += 1
    print("Hoàn thành preprocess")
