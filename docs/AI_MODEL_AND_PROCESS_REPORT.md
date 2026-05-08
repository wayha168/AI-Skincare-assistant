# Skinme AI Model and Process Report

This document summarizes the AI models, training pipeline, image storage design, and runtime process currently implemented in this project.

---

## 1) System Objective

The Skin Assistant provides:

- Text chat support for skincare questions and recommendations
- Optional image-based skin condition analysis
- Product/ingredient retrieval and recommendation
- Optional backend integration for chat/feedback persistence

The system is designed so you can run the API now, collect/upload image data over time, and train image models later when enough labeled data is available.

---

## 2) AI Models in the Project

### 2.1 Intent Classification Model (NLP)

- Purpose: classify user message intent (`greeting`, `ingredient_info`, `product_recommendation`, etc.)
- Training command: `python main.py train`
- Data source: `data/intent_training.csv` (optional) or built-in training examples
- Artifact location: `models/artifacts/`

### 2.2 Product Models (Text/Image, optional)

- Purpose: product type prediction and product-related support
- Training command:
  - Text model: `python main.py train-products`
  - Image model: `python main.py train-products --image`
- Data source: `data/skinme_products.csv` and product images
- Artifact location: `models/artifacts/`

### 2.3 Skin Condition Ensemble (Image, 2 Models)

- Purpose: analyze uploaded skin photos and predict likely condition
- Inference uses an ensemble of at least 2 models:
  - `resnet18`
  - `efficientnet_b0`
- Predictions are combined by averaging class probabilities.
- Training command: `python main.py train-skin-condition`
- Artifact location:
  - `models/artifacts/skin_condition_model_resnet18.pt`
  - `models/artifacts/skin_condition_model_efficientnet_b0.pt`

---

## 3) Image Storage and Data Flow

### 3.1 Single Folder Strategy

All skin condition image data is stored in one root folder:

- `data/skin_condition_images/`

Structure:

- Labeled data: `data/skin_condition_images/<condition>/image.jpg`
- Unlabeled uploads: `data/skin_condition_images/unlabeled/image.jpg`
- Optional CSV labels template: `data/skin_condition_images/labels.csv.example`
- Upload metadata log: `data/skin_condition_images/uploads_metadata.csv`

### 3.2 Upload Process (`/v1/chat/with-image`)

When user uploads an image:

1. API receives multipart form data.
2. Image is auto-saved into the unified storage folder.
3. If `training_label` is provided, image goes directly to labeled class folder.
4. If `training_label` is missing, image goes to `unlabeled/`.
5. Metadata is appended to `uploads_metadata.csv`.
6. If trained model artifacts exist, image inference is executed and returned in response.

This allows immediate data collection before a final dataset is uploaded/labeled.

---

## 4) Training Pipeline (Skin Condition)

Training function: `train_skin_condition_classifier(...)`

Pipeline summary:

1. Load labeled images from:
   - folder structure under `data/skin_condition_images/<condition>/...`, or
   - CSV mapping (`image_name`, `condition`) if provided.
2. Validate minimum data:
   - at least 2 images
   - at least 2 classes
3. Build transforms:
   - training augmentations (flip, rotation, color jitter)
   - evaluation normalization
4. Split train/validation data (80/20, seeded for reproducibility)
5. Train two models independently:
   - ResNet18
   - EfficientNet-B0
6. Evaluate validation accuracy for each model and save best checkpoints
7. Report ensemble average validation accuracy and target pass/fail status

Default target accuracy is `0.95` and is configurable via CLI.

---

## 5) Runtime Process (API)

Main run command:

- `python main.py`

Runtime behavior:

- Starts FastAPI skin assistant service
- Loads available model artifacts from `models/artifacts/`
- Serves text and image chat endpoints
- Uses retrieval and optional LLM for response generation
- Optionally forwards chat logs/feedback to backend if `SPRING_BACKEND_URL` is configured

If skin condition models are not trained yet:

- API still runs
- image uploads are still stored for future training
- image inference returns no condition until models are trained

---

## 6) Current Status and Expected Next Step

### Current status

- Upload-to-storage flow is implemented
- Unified image data folder is implemented
- 2-model image ensemble is implemented
- Training command and docs are ready

### Next step when dataset is available

1. Upload and organize labeled images into `data/skin_condition_images/<condition>/`
2. (Optional) use CSV labels file in `data/skin_condition_images/`
3. Train:
   - `python main.py train-skin-condition --epochs 10 --batch-size 16`
4. Re-run API:
   - `python main.py`

---

## 7) Notes on Accuracy (95% Target)

- The code supports a 95% validation target and reports whether the run meets it.
- Reaching and sustaining >=95% depends mainly on dataset quality:
  - enough samples per class
  - balanced class distribution
  - clean labels
  - representative real-world images
- If target is not reached, continue collecting/cleaning data and retrain.

---

## 8) Key Commands Quick Reference

- Run API now: `python main.py`
- Train intent model: `python main.py train`
- Train skin condition models: `python main.py train-skin-condition`
- Train with stronger settings: `python main.py train-skin-condition --epochs 10 --batch-size 16`

