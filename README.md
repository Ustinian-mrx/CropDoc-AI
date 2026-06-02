# CropDoc AI

CropDoc AI is an end-to-end crop leaf disease diagnosis demo built with computer vision, explainable AI, a language-model advice module, and a web interface.

The project currently supports image upload, crop disease classification, Top-3 predictions, Grad-CAM heatmap visualization, structured prevention advice, and SQLite-based diagnosis history.

> This project is for learning, prototyping, and demonstration. The diagnosis and advice are only references and should not replace local agricultural experts or field inspection.

## Features

- Leaf image upload and preview
- PlantVillage disease classification with EfficientNet-B0
- Top-3 prediction probabilities
- Label parsing into crop, disease, health status, and confidence
- Grad-CAM heatmap generation and static image serving
- Structured prevention advice through MIMO API, with rule-based fallback
- SQLite diagnosis history
- Next.js frontend for prediction results, heatmaps, advice, and history records
- FastAPI backend with interactive Swagger docs

## Project Structure

```text
CropDoc AI
├── backend
│   ├── app
│   │   ├── main.py
│   │   └── services
│   │       ├── advice.py
│   │       ├── db.py
│   │       ├── gradcam.py
│   │       ├── inference.py
│   │       ├── label_utils.py
│   │       └── mimo_client.py
│   ├── requirements.txt
│   └── static
│       └── heatmaps
├── frontend
│   ├── app
│   │   └── page.tsx
│   ├── package.json
│   └── package-lock.json
├── ml
│   ├── check_dataset.py
│   ├── config.py
│   ├── inspect_dataloader.py
│   ├── predict_image.py
│   └── train.py
├── data
│   └── PlantVillage
├── models
│   └── efficientnet_b0_plantvillage.pth
└── README.md
```

`data/`, model weights, generated heatmaps, local databases, and environment files are intentionally ignored by Git.

## Tech Stack

### Machine Learning

- Python
- PyTorch
- torchvision
- EfficientNet-B0 transfer learning
- PlantVillage dataset
- Grad-CAM

### Backend

- FastAPI
- Uvicorn
- Pillow
- OpenCV
- SQLite
- httpx
- python-dotenv

### Frontend

- Next.js
- React
- TypeScript
- Tailwind CSS

## Current Model Result

The first trained model uses EfficientNet-B0 with ImageNet pretraining.

```text
Train images: 43444
Validation images: 10861
Classes: 38
Best validation accuracy: 0.9965
Model file: models/efficientnet_b0_plantvillage.pth
```

The validation result is based on the prepared PlantVillage split and should not be treated as real-field performance.

## Requirements

- Python 3.10+
- Node.js 18+
- NVIDIA GPU is recommended for training
- CUDA-enabled PyTorch is recommended for GPU inference/training

The backend can also run on CPU, but inference and Grad-CAM will be slower.

## Dataset Preparation

Expected dataset layout:

```text
data/PlantVillage
├── train
│   ├── Apple___Apple_scab
│   ├── Apple___Black_rot
│   └── ...
└── val
    ├── Apple___Apple_scab
    ├── Apple___Black_rot
    └── ...
```

Check the dataset:

```powershell
cd "D:\CropDoc AI\ml"
.venv\Scripts\activate
python check_dataset.py
```

Expected:

```text
Found 38 class folders.
train split looks good.
val split looks good.
```

## Model Training

Create and activate the ML environment:

```powershell
cd "D:\CropDoc AI\ml"
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

For CUDA wheels, install PyTorch with the matching PyTorch index if needed:

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Train the model:

```powershell
python train.py
```

The trained checkpoint is saved to:

```text
models/efficientnet_b0_plantvillage.pth
```

Test single-image inference:

```powershell
python predict_image.py --image "D:\CropDoc AI\data\PlantVillage\val\Apple___Apple_scab\example.JPG"
```

## Backend Setup

Create and activate the backend environment:

```powershell
cd "D:\CropDoc AI\backend"
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

If PyTorch CUDA wheels fail through `requirements.txt`, install them explicitly:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Create `backend/.env`:

```env
MIMO_API_KEY=your_api_key
MIMO_API_BASE_URL=https://api.xiaomimimo.com/v1/chat/completions
MIMO_MODEL=mimo-v2.5-pro
```

The MIMO configuration is optional. If it is missing or the request fails, the backend uses local rule-based advice.

Start the backend:

```powershell
cd "D:\CropDoc AI\backend"
.venv\Scripts\activate
uvicorn app.main:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

## Frontend Setup

Create `frontend/.env.local`:

```env
NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000
```

Install and start:

```powershell
cd "D:\CropDoc AI\frontend"
npm install
npm run dev
```

Open:

```text
http://localhost:3000
```

## API Overview

### Health Check

```http
GET /health
```

Response:

```json
{
  "status": "ok",
  "service": "cropdoc-backend"
}
```

### Predict

```http
POST /predict
```

Form field:

```text
file: image file
```

Response includes:

- `record_id`
- `filename`
- `model`
- `prediction`
- `top3`
- `explanation.heatmap_url`
- `advice`
- `note`

### History

```http
GET /history?limit=20
```

Response:

```json
{
  "items": [],
  "count": 0
}
```

## Development Flow

Recommended startup order:

1. Start backend on `http://127.0.0.1:8000`
2. Start frontend on `http://localhost:3000`
3. Upload a validation image
4. Check prediction, Top-3, heatmap, advice, and history

## Generated Runtime Files

These are intentionally not committed:

- `data/`
- `models/*.pth`
- `backend/.env`
- `frontend/.env.local`
- `backend/cropdoc.db`
- `backend/static/heatmaps/*.jpg`

## Known Limitations

- PlantVillage images are mostly controlled-condition leaf images, so the model accuracy does not directly represent real-field performance.
- Grad-CAM explains model attention but does not prove biological correctness.
- The MIMO advice module depends on external API availability and JSON response quality.
- This version does not yet include login, user isolation, Docker Compose, or production deployment.

## Roadmap

- User registration and JWT login
- User-specific diagnosis history
- Docker Compose local deployment
- ONNX Runtime inference
- EfficientNetV2 experiment
- README screenshots and demo video
- More robust field-image dataset evaluation
