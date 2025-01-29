# app/main.py

import os, shutil
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from celery.result import AsyncResult
from .tasks import analyze_with_hf, generate_text_with_hf
from .models_config import MODELS

app = FastAPI(title="HF Analysis & Generation Service")

UPLOAD_DIR = "app/storage/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.post("/analyze-image", summary="Analyze an uploaded image with a Hugging Face model")
async def analyze_image_endpoint(model_key: str = Form(...), file: UploadFile = File(...)):
    """
    Endpoint to upload an image & run image inference via Celery.
    """
    try:
        # Check model
        if model_key not in MODELS:
            raise HTTPException(status_code=400, detail="Unknown model key: %s" % model_key)

        # Save uploaded file
        image_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(image_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Submit Celery task
        task = analyze_with_hf.apply_async(args=[image_path, model_key])
        return {"task_id": task.id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate-text", summary="Generate text from a prompt using a text-generation model")
async def generate_text_endpoint(
    model_key: str = Form(...),
    prompt: str = Form(...),
    max_new_tokens: int = Form(50)
):
    """
    Endpoint to run text-generation with a prompt and a specified model_key.
    """
    try:
        if model_key not in MODELS:
            raise HTTPException(status_code=400, detail="Unknown model key: %s" % model_key)
        # Submit Celery task
        task = generate_text_with_hf.apply_async(args=[prompt, model_key, max_new_tokens])
        return {"task_id": task.id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/task-status/{task_id}", summary="Check the status of a Celery task")
def get_task_status(task_id: str):
    """
    Generic endpoint to retrieve the status of a Celery task.
    Suitable for both image inference and text generation tasks.
    """
    result = AsyncResult(task_id)
    if result.state == "PENDING":
        return {"task_id": task_id, "status": "PENDING"}
    elif result.state == "SUCCESS":
        return {"task_id": task_id, "status": "SUCCESS", "result": result.result}
    elif result.state == "FAILURE":
        return {"task_id": task_id, "status": "FAILURE", "error": str(result.info)}
    else:
        return {"task_id": task_id, "status": result.state}
