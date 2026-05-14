from io import BytesIO

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image

from app.services.inference import predictor


app = FastAPI(
    title="CropDoc AI Backend",
    description="Backend service for crop disease diagnosis.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {
        "message": "Welcome to CropDoc AI Backend"
    }


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "cropdoc-backend"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file must be an image.",
        )

    try:
        file_bytes = await file.read()
        image = Image.open(BytesIO(file_bytes))

        top3 = predictor.predict(
            image=image,
            top_k=3,
        )

        best_result = top3[0]

        return {
            "filename": file.filename,
            "model": {
                "name": "efficientnet_b0",
                "version": "0.1.0",
                "image_size": 224,
        },
        "prediction": best_result,
        "top3": top3,
        "explanation": {
            "gradcam_available": False,
            "heatmap_url": None,
        },
        "note": "Prediction is based on the PlantVillage-trained model and should be used as a reference only.",
    }   


    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {exc}",
        )