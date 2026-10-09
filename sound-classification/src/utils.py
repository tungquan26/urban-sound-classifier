import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from config import TRAIN_LOGS_DIR
import torch.nn as nn
import torchaudio.transforms as T

class SpecAugment(nn.Module):
    def __init__(self, freq_mask_param=10, time_mask_param=10):
        super().__init__()
        self.freq_masking = T.FrequencyMasking(freq_mask_param=freq_mask_param)
        self.time_masking = T.TimeMasking(time_mask_param=time_mask_param)

    def forward(self, tensor):
        augmented = self.freq_masking(tensor)
        augmented = self.time_masking(augmented)
        return augmented

def plot_training_history(history, timestamp_str):
    actual_epochs = len(history["train_loss"])
    epochs_range = range(1, actual_epochs + 1)

    plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, history["train_loss"], label='Train Loss', color='blue', linewidth=2)
    plt.plot(epochs_range, history["val_loss"], label='Validation Loss', color='red', linestyle='--', linewidth=2)
    plt.title('Biểu đồ Sai số (Loss)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)

    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, history["train_acc"], label='Train Accuracy', color='blue', linewidth=2)
    plt.plot(epochs_range, history["val_acc"], label='Validation Accuracy', color='green', linestyle='--', linewidth=2)
    plt.title('Biểu đồ Độ chính xác (Accuracy)')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)

    plt.subplot(1, 3, 3)
    plt.plot(epochs_range, history["lr"], label='Learning Rate', color='purple', linewidth=2)
    plt.title('Lịch sử Learning Rate')
    plt.xlabel('Epochs')
    plt.ylabel('LR')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)

    plot_filename = f"training_history_{timestamp_str}.png"
    plot_path = os.path.join(TRAIN_LOGS_DIR, plot_filename)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.show()

def plot_confusion_matrix(cm, class_names, save_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.title('Confusion Matrix - Urban Sound Classification', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
