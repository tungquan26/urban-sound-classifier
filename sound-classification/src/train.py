import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from typing import Tuple
from config import (EPOCHS, LEARNING_RATE, TRAIN_LOGS_DIR, WEIGHT_DECAY, T_MAX,
                    ETA_MIN, DEVICE, MODELS_DIR, FEATURE_DIR, BATCH_SIZE)
from dataset import get_dataloaders
from model_cnn import UrbanSoundCNN
from utils import plot_training_history


def train_one_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer, device: torch.device) -> Tuple[float, float]:
    model.train()
    running_train_loss = 0.0
    correct_train = 0
    total_train = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_train_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    train_loss = running_train_loss / total_train
    train_acc = 100 * correct_train / total_train
    return train_loss, train_acc


def evaluate(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device) -> Tuple[float, float]:
    model.eval()
    running_val_loss = 0.0
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    val_loss = running_val_loss / total_val
    val_acc = 100 * correct_val / total_val
    return val_loss, val_acc


def train():
    print("Start")
    print("-" * 50)

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    train_loader, val_loader, _ = get_dataloaders(FEATURE_DIR, BATCH_SIZE)
    model = UrbanSoundCNN().to(DEVICE)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    scheduler = CosineAnnealingLR(optimizer, T_max=T_MAX, eta_min=ETA_MIN)

    best_val_acc = 0.0
    best_val_loss = 1e9
    best_model_path = os.path.join(MODELS_DIR, f"best_model_{timestamp_str}.pth")

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [], "lr": []
    }

    patience = 10
    epochs_no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        start_time = time.time()
        current_lr = optimizer.param_groups[0]['lr']

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = evaluate(model, val_loader, criterion, DEVICE)

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        saved_msg = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            saved_msg = "[ĐÃ LƯU MÔ HÌNH MỚI]"
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        end_time = time.time()
        mins, secs = divmod(end_time - start_time, 60)

        print(f"Epoch [{epoch:02d}/{EPOCHS}] | LR: {current_lr:.1e} | Thời gian: {int(mins)}m {int(secs)}s")
        print(f"  + Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  + Val. Loss:  {val_loss:.4f} | Val. Acc:  {val_acc:.2f}% {saved_msg}")

        if epochs_no_improve >= patience:
            print(f"\n Kích hoạt Early Stopping tại Epoch {epoch}.")
            break

    print("-" * 50)
    print(f"Hoàn tất!  Mô hình tốt nhất đạt độ chính xác {best_val_acc:.2f}%")
    plot_training_history(history, timestamp_str)


if __name__ == "__main__":
    train()
