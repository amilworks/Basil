MODELS = {
    # Typical pipeline-based model
    "dino-seg": {
        "hf_repo": "nielsr/dino-segmentation",
        "task_mode": "hf_pipeline",  # use the pipeline approach
        "task_name": "image-segmentation"
    },
    # Custom code for ViT classification
    "vit-base": {
        "hf_repo": "google/vit-base-patch16-224",
        "task_mode": "vit_custom",   # use our custom ViT approach
    },
    "deep-seek": {
        "hf_repo": "deepseek-ai/DeepSeek-R1",
        "task_mode": "text_generation",
        "task_name": "text-generation"
    },
    "gpt2": {
        "hf_repo": "gpt2",
        "task_mode": "text_generation",
        "task_name": "text-generation"
    },
    # Add more as needed ...
}
