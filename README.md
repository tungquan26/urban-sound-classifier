# 🔊 Urban Sound Classifier

Dự án phân loại âm thanh đô thị sử dụng mô hình **CNN (Convolutional Neural Network)** được xây dựng bằng **PyTorch**. Mô hình nhận đầu vào là file âm thanh và phân loại thành **10 lớp** âm thanh môi trường đô thị dựa trên bộ dữ liệu [UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html).

Dự án bao gồm toàn bộ pipeline từ tiền xử lý dữ liệu, trích xuất đặc trưng, huấn luyện, đánh giá mô hình, đến một **Web Demo** cho phép người dùng tải lên file âm thanh và nhận kết quả phân loại trực tiếp trên trình duyệt.

---

## 📁 Cấu Trúc Thư Mục

```
Urban Sound Classifier/
├── sound-classification/          # Module huấn luyện mô hình
│   ├── src/
│   │   ├── config.py              # Cấu hình hyperparameters & đường dẫn
│   │   ├── pre_process.py         # Tiền xử lý audio (resample, pad/truncate)
│   │   ├── extract_features.py    # Trích xuất Mel Spectrogram + Delta
│   │   ├── dataset.py             # Dataset & DataLoader cho PyTorch
│   │   ├── model_cnn.py           # Kiến trúc mô hình CNN
│   │   ├── train.py               # Script huấn luyện mô hình
│   │   ├── evaluate.py            # Script đánh giá trên tập test
│   │   └── utils.py               # SpecAugment, vẽ biểu đồ
│   ├── data/                      # Dữ liệu (không push lên git)
│   │   ├── raw/                   # Audio gốc từ UrbanSound8K
│   │   ├── interim/               # Audio đã tiền xử lý
│   │   └── processed/             # Tensor đặc trưng (.pt)
│   ├── models/                    # File trọng số mô hình (.pth)
│   ├── logs/                      # Biểu đồ training & confusion matrix
│   └── notebooks/                 # Jupyter notebook khám phá dữ liệu
│
├── web-demo/                      # Ứng dụng web demo
│   ├── backend/
│   │   ├── app.py                 # FastAPI server + API /predict
│   │   ├── model/
│   │   │   ├── architecture.py    # Kiến trúc CNN (bản sao cho inference)
│   │   │   └── weights.pth        # Trọng số mô hình đã huấn luyện
│   │   └── utils/
│   │       └── audio_processor.py # Xử lý audio cho inference
│   └── frontend/
│       ├── index.html             # Giao diện web chính
│       ├── css/                   # Stylesheets
│       └── js/                    # Logic upload, biểu đồ, waveform
│
├── requirements.txt               # Danh sách thư viện Python
└── README.md
```

---

## 🏗️ Kiến Trúc Mô Hình

Mô hình **UrbanSoundCNN** là một mạng CNN tùy chỉnh gồm 4 tầng convolution:

| Tầng | Cấu trúc | Output Channels |
|------|-----------|-----------------|
| Conv Block 1 | Conv2d → BatchNorm → ReLU → MaxPool(2×2) | 32 |
| Conv Block 2 | Conv2d → BatchNorm → ReLU → MaxPool(1×2) | 64 |
| Conv Block 3 | Conv2d → BatchNorm → ReLU → MaxPool(1×2) | 128 |
| Conv Block 4 | Conv2d → BatchNorm → ReLU → MaxPool(1×2) | 256 |
| Global Avg Pool | AdaptiveAvgPool2d(1, 1) | 256 |
| Classifier | Linear(256→128) → ReLU → Dropout(0.5) → Linear(128→10) | 10 |

**Đầu vào:** Tensor 3 kênh `(3, 128, 173)` gồm:
- **Kênh 1:** Mel Spectrogram (128 mel bands)
- **Kênh 2:** Delta (đạo hàm bậc 1)
- **Kênh 3:** Delta-Delta (đạo hàm bậc 2)

---

## 📊 Kết Quả Đạt Được

### Tổng quan

| Metric | Giá trị |
|--------|---------|
| **Test Accuracy** | **79.21%** |
| **Test Loss** | 0.5912 |
| **Macro Avg Precision** | 0.83 |
| **Macro Avg Recall** | 0.80 |
| **Macro Avg F1-Score** | 0.80 |

### Chi tiết theo từng lớp

| Lớp âm thanh | Precision | Recall | F1-Score | Số mẫu |
|--------------|-----------|--------|----------|---------|
| Air Conditioner | 0.60 | 0.90 | 0.72 | 100 |
| Car Horn | 0.88 | 0.85 | 0.86 | 33 |
| Children Playing | 0.66 | 0.85 | 0.75 | 100 |
| Dog Bark | 0.82 | 0.67 | 0.74 | 100 |
| Drilling | 0.92 | 0.85 | 0.89 | 100 |
| Engine Idling | 0.94 | 0.63 | 0.76 | 93 |
| **Gun Shot** | **1.00** | **1.00** | **1.00** | 32 |
| **Jackhammer** | **0.98** | **0.99** | **0.98** | 96 |
| Siren | 0.76 | 0.41 | 0.53 | 83 |
| Street Music | 0.75 | 0.88 | 0.81 | 100 |

> **Nhận xét:** Mô hình phân loại rất tốt các lớp **Gun Shot** (F1 = 1.00) và **Jackhammer** (F1 = 0.98). Lớp **Siren** có hiệu suất thấp nhất (F1 = 0.53) do dễ bị nhầm lẫn với các âm thanh khác.

### Cấu hình huấn luyện

| Tham số | Giá trị |
|---------|---------|
| Epochs | 80 (Early Stopping patience = 10) |
| Optimizer | Adam (lr = 1e-4, weight_decay = 1e-4) |
| Scheduler | CosineAnnealingLR (eta_min = 1e-6) |
| Loss | CrossEntropyLoss (label_smoothing = 0.05) |
| Batch Size | 32 |
| Augmentation | SpecAugment (FreqMask = 10, TimeMask = 10) |
| Chia dữ liệu | Fold 1-8: Train / Fold 9: Validation / Fold 10: Test |

---

## 🚀 Hướng Dẫn Sử Dụng

### 1. Yêu cầu hệ thống

- Python 3.10+
- CUDA 12.1 (khuyến nghị, để huấn luyện trên GPU)

### 2. Cài đặt

```bash
# Clone repository
git clone <repository-url>
cd "Urban Sound Classifier"

# Tạo môi trường ảo
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/macOS

# Cài đặt thư viện
pip install -r requirements.txt
```

> **Lưu ý:** PyTorch trong `requirements.txt` sử dụng phiên bản CUDA 12.1. Nếu bạn không có GPU NVIDIA, hãy cài đặt phiên bản CPU từ [pytorch.org](https://pytorch.org/).

### 3. Chuẩn bị dữ liệu

Tải bộ dữ liệu [UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html) và giải nén vào thư mục:

```
sound-classification/data/raw/
├── fold1/
├── fold2/
├── ...
└── fold10/
```

### 4. Pipeline huấn luyện

Chạy lần lượt các bước sau từ thư mục `sound-classification/src/`:

```bash
cd sound-classification/src

# Bước 1: Tiền xử lý audio (resample về 22050Hz, chuẩn hóa độ dài 4s)
python pre_process.py

# Bước 2: Trích xuất đặc trưng (Mel Spectrogram + Delta + Delta2 → tensor .pt)
python extract_features.py

# Bước 3: Huấn luyện mô hình
python train.py

# Bước 4: Đánh giá trên tập test (fold 10)
python evaluate.py
```

Sau khi huấn luyện xong:
- Trọng số mô hình tốt nhất được lưu tại `sound-classification/models/`
- Biểu đồ training history lưu tại `sound-classification/logs/train_logs/`
- Confusion matrix lưu tại `sound-classification/logs/confusion_matrices/`

### 5. Chạy Web Demo

#### Khởi động Backend (FastAPI)

```bash
cd web-demo/backend
uvicorn app:app --reload --port 8000
```

API sẽ chạy tại `http://localhost:8000`. Endpoint dự đoán: `POST /predict`

#### Mở Frontend

Mở file `web-demo/frontend/index.html` trực tiếp bằng trình duyệt, hoặc dùng Live Server (VS Code extension).

#### Sử dụng Web Demo

1. Kéo thả hoặc chọn file âm thanh (`.wav`, `.mp3`) vào khu vực upload
2. Nhấn nút **CLASSIFY AUDIO**
3. Xem kết quả phân loại, độ tin cậy (confidence), biểu đồ phân phối xác suất, và dạng sóng âm thanh

---

## 🛠️ Công Nghệ Sử Dụng

| Thành phần | Công nghệ |
|------------|-----------|
| Deep Learning Framework | PyTorch, torchaudio |
| Xử lý audio | torchaudio, librosa, soundfile |
| Xử lý dữ liệu | NumPy, Pandas, scikit-learn |
| Trực quan hóa | Matplotlib, Seaborn |
| Backend API | FastAPI, Uvicorn |
| Frontend | HTML, TailwindCSS, JavaScript |
| Biểu đồ web | Chart.js |
| Waveform | WaveSurfer.js |

---

## 📌 Ghi Chú

- File trọng số cho web demo nằm tại `web-demo/backend/model/weights.pth` — cần copy từ model đã huấn luyện trước khi chạy demo.
