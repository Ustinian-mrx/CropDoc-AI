from io import BytesIO

from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile, HTTPException
from app.services.advice import build_advice
from PIL import Image

from app.services.inference import predictor
from app.services.db import (
    init_db,
    save_diagnosis_record,
    list_diagnosis_records,
)



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
app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static",
)
init_db()



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

        prediction_output = predictor.predict(
            image=image,
            top_k=3,
        )

        top3 = prediction_output["top3"]
        heatmap_url = prediction_output["heatmap_url"]

        best_result = top3[0]
        advice = build_advice(best_result)
        record_id = save_diagnosis_record(
            filename=file.filename or "unknown",
            prediction=best_result,
            top3=top3,
            advice=advice,
            heatmap_url=heatmap_url,
    )

        return {
            "record_id": record_id,
            "filename": file.filename,
            "model": {
                "name": "efficientnet_b0",
                "version": "0.1.0",
                "image_size": 224,
            },
            "prediction": best_result,
            "top3": top3,
            "explanation": {
                "gradcam_available": True,
                "heatmap_url": heatmap_url,
            },
            "note": "预测是基于plantvillage训练的模型，只应用作参考。实际诊断请结合田间观察和农技人员建议。",
            "advice": advice,
        
        }   


    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {exc}",
        )
        
@app.get("/history")
def get_history(limit: int = 20):
    records = list_diagnosis_records(limit=limit)

    return {
        "items": records,
        "count": len(records),
    }
