import os
import glob
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from config import DEVICE, MATRIX_DIR, MODELS_DIR, FEATURE_DIR
from dataset import get_dataloaders
from model_cnn import UrbanSoundCNN
from utils import plot_confusion_matrix

URBAN_SOUND_CLASSES = [
    "air_conditioner", "car_horn", "children_playing", "dog_bark",
    "drilling", "engine_idling", "gun_shot", "jackhammer",
    "siren", "street_music"
]

def get_latest_model_path(models_dir):
    list_of_files = glob.glob(os.path.join(models_dir, '*.pth'))
    if not list_of_files:
        return None
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file


def evaluate():
    print(f"Bắt đầu quá trình đánh giá trên thiết bị: {DEVICE}")

    model_path = get_latest_model_path(MODELS_DIR)
    if not model_path:
        print(f"Không tìm thấy file model nào trong thư mục {MODELS_DIR}!")
        return
    print(f"Đang tải trọng số từ: {model_path}")

    _, _, test_loader = get_dataloaders(FEATURE_DIR)

    if len(test_loader.dataset) == 0:
        print("Tập Test trống! Vui lòng kiểm tra lại dữ liệu trong FEATURE_DIR/fold10.")
        return

    model = UrbanSoundCNN().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    model.eval()

    criterion = nn.CrossEntropyLoss()

    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    test_loss = running_loss / len(test_loader.dataset)
    test_acc = np.sum(all_preds == all_labels) / len(all_labels) * 100

    print(f"\n[KẾT QUẢ] Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.2f}%")
    report = classification_report(all_labels, all_preds, target_names=URBAN_SOUND_CLASSES)
    print(f"\nCLASSIFICATION REPORT:\n{report}")

    cm = confusion_matrix(all_labels, all_preds)
    cm_filename = f"confusion_matrix_{os.path.basename(model_path).replace('.pth', '')}.png"
    cm_save_path = os.path.join(MATRIX_DIR, cm_filename)

    plot_confusion_matrix(cm, URBAN_SOUND_CLASSES, cm_save_path)

if __name__ == "__main__":
    evaluate()
