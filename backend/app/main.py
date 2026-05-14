from fastapi import FastAPI

app = FastAPI(
    title="CropDoc AI Backend",
    description="Backend service for crop disease diagnosis.",
    version="0.1.0",
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
