import os
import shutil
import tempfile
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import torch
import torch.nn.functional as F

from model.architecture import UrbanSoundCNN
from utils.audio_processor import preprocess_audio

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

CLASSES = [
    "Air Conditioner", "Car Horn", "Children Playing",
    "Dog Bark", "Drilling", "Engine Idling",
    "Gun Shot", "Jackhammer", "Siren", "Street Music"
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UrbanSoundCNN(num_classes=10).to(device)
model.load_state_dict(torch.load("model/weights.pth", map_location=device, weights_only=True))
model.eval()

@app.post("/predict")
async def predict_audio(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        temp_path = temp_file.name

    try:
        input_tensor = preprocess_audio(temp_path).to(device)

        with torch.no_grad():
            logits = model(input_tensor)
            probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()

        results = [{"class": CLASSES[i], "prob": float(probs[i]) * 100} for i in range(10)]
        results.sort(key=lambda x: x["prob"], reverse=True)
        top_4 = results[:4]

        return {
            "top_class": top_4[0]["class"],
            "confidence": round(top_4[0]["prob"], 1),
            "distribution": [{"class": item["class"], "prob": round(item["prob"], 1)} for item in top_4]
        }
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

frontend_path = os.path.join(os.path.dirname(__file__), "../frontend")
app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")
