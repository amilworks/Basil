from app.celery_app import celery_app
from transformers import (
    pipeline, 
    ViTImageProcessor, 
    ViTForImageClassification
)
from PIL import Image
import os
import requests
import torch
from .models_config import MODELS

@celery_app.task(bind=True, max_retries=3, default_retry_delay=60)
def analyze_with_hf(self, model_key, image_path=None, text_input=None):
    """
    Celery task to run inference on either an image or text input using Hugging Face models.

    Supported 'task_mode's in MODELS:
      1) 'hf_pipeline'         : Use `pipeline(...)` for image-based tasks (segmentation, etc.).
      2) 'vit_custom'          : Custom code for ViT image classification.
      3) 'hf_text_generation'  : Use `pipeline('text-generation', ...)` or custom LLM approach.

    Parameters
    ----------
    model_key : str
        Key referencing a model in the MODELS dict.
    image_path : str, optional
        Path to the input image to analyze (if any).
    text_input : str, optional
        Text prompt or input for a large language model (if any).

    Returns
    -------
    dict
        {
          "status": "success",
          "results": <inference results (label, text, etc.)>
        }
        or
        {
          "status": "error",
          "message": <error message>
        }
    """
    try:
        # 1. Verify the model key is valid
        if model_key not in MODELS:
            raise ValueError("Unknown model key: %s" % model_key)

        model_info = MODELS[model_key]
        hf_repo = model_info["hf_repo"]
        task_mode = model_info["task_mode"]

        # 2. Distinguish between text vs. image input
        if image_path is not None and not os.path.exists(image_path):
            raise FileNotFoundError("Image file not found: %s" % image_path)

        # 3. Branch based on 'task_mode'
        if task_mode == "hf_pipeline":
            # Typical pipeline tasks for images (e.g., segmentation, classification)
            if not image_path:
                raise ValueError("image_path is required for hf_pipeline mode")

            from transformers import pipeline
            task_name = model_info.get("task_name", "image-classification")
            inference_pipeline = pipeline(
                task=task_name,
                model=hf_repo,
            )
            # Load image via PIL
            from PIL import Image
            pil_image = Image.open(image_path).convert("RGB")

            results = inference_pipeline(pil_image)
            return {"status": "success", "results": results}

        elif task_mode == "vit_custom":
            # Custom approach for ViT classification
            if not image_path:
                raise ValueError("image_path is required for vit_custom mode")

            from transformers import ViTImageProcessor, ViTForImageClassification
            processor = ViTImageProcessor.from_pretrained(hf_repo)
            model = ViTForImageClassification.from_pretrained(hf_repo)

            from PIL import Image
            pil_image = Image.open(image_path).convert("RGB")
            inputs = processor(images=pil_image, return_tensors="pt")

            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class_idx = logits.argmax(-1).item()
            label = model.config.id2label[predicted_class_idx]

            return {
                "status": "success",
                "results": {
                    "label": label,
                    "class_idx": predicted_class_idx
                }
            }

        elif task_mode == "hf_text_generation":
            # New mode for text input + text-generation pipeline
            if not text_input:
                raise ValueError("text_input is required for hf_text_generation mode")

            # Option 1: Use pipeline('text-generation', model=...)
            # For Llama-based or GPT-2 style models
            text_pipeline = pipeline(
                "text-generation",
                model=hf_repo,
            )
            # default max_length or pass as param
            generation = text_pipeline(text_input, max_length=200, do_sample=True)
            # Typically returns a list of dicts: [{'generated_text': '...'}, ...]

            return {"status": "success", "results": generation}

        else:
            raise ValueError("Unsupported task_mode: %s" % task_mode)

    except Exception as e:
        return {"status": "error", "message": str(e)}





# app/tasks.py (additional code for text generation)

@celery_app.task(bind=True, max_retries=3, default_retry_delay=60)
def generate_text_with_hf(self, prompt, model_key, max_new_tokens=50):
    """
    Celery task to run text-generation (LLM) inference using the 
    pipeline-based approach. For example, GPT-2 or LLaMA.

    Parameters
    ----------
    prompt : str
        The text prompt to feed into the model.
    model_key : str
        The key referencing a text-generation model in MODELS.
    max_new_tokens : int
        Limit for how many tokens to generate.

    Returns
    -------
    dict
        { "status": "success", "results": <generated text> }
        or
        { "status": "error", "message": <error> }
    """
    try:
        if model_key not in MODELS:
            raise ValueError("Unknown model key: %s" % model_key)

        model_info = MODELS[model_key]
        hf_repo = model_info["hf_repo"]
        task_mode = model_info["task_mode"]
        if task_mode != "text_generation":
            raise ValueError("Model %s is not configured for text_generation" % model_key)

        # Create a pipeline for text-generation
        task_name = model_info.get("task_name", "text-generation")
        generation_pipeline = pipeline(
            task=task_name,
            model=hf_repo,
            trust_remote_code=True,
            # device=0 if GPU is available, e.g. device=0
        )

        # Run generation
        output = generation_pipeline(
            prompt, 
            max_new_tokens=max_new_tokens,
            # other pipeline args (e.g. do_sample, temperature, etc.)
        )

        # Typically output is a list of dict, e.g. [{"generated_text": "..."}]
        # We'll just pass it along
        return {"status": "success", "results": output}

    except Exception as e:
        return {"status": "error", "message": str(e)}
