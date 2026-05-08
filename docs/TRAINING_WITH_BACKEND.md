# How to Train the AI and Use It With Your Backend

Training runs **locally** in this project (Python). Your Spring (or other) backend does not run the training; it **calls** the Skin Assistant API, which uses the trained models. This guide covers both training and wiring the trained AI to your backend.

---

## 1. Overview

| What | Where it runs | Data source |
|------|----------------|-------------|
| **Intent model** | Local (`python main.py train`) | `data/intent_training.csv` or built-in examples |
| **Product text/image models** | Local (`python main.py train-products`) | `data/skinme_products.csv` (from sync) |
| **Skin condition ensemble (2 models)** | Local (`python main.py train-skin-condition`) | `data/skin_condition_images/` (single folder for upload + training) |
| **API (chat, products, intent)** | Local (`python main.py`) | Trained artifacts + CSV + optional MySQL |
| **Save chat/feedback to DB** | Your backend | Skin Assistant forwards to `SPRING_BACKEND_URL` |

---

## 2. Step-by-Step: Train Then Run With Backend

### 2.1 Prepare product data (for product recommendations and product models)

```bash
# Fetch products from SkinMe API into data/skinme_products.csv and download images
python main.py sync
```

### 2.2 Train the models you need

**Intent model** (classifies user message: greeting, ingredient_info, product_recommendation, etc.):

```bash
python main.py train
```

Optional: use your own examples. Create `data/intent_training.csv` with columns `text` and `intent` (see `data/intent_training.csv.example`), then:

```bash
python main.py train
# Or with explicit paths:
python -m scripts.train --data data/intent_training.csv --output models/artifacts
```

**Product models** (optional; for product-type prediction from text/image):

```bash
python main.py train-products
# With image model (needs PyTorch):
python main.py train-products --image --epochs 5
```

**Skin condition model** (optional; for selfie analysis in API/chat):

- Use one storage root: `data/skin_condition_images/`.
- API uploads from `POST /v1/chat/with-image` are auto-saved there:
  - labeled when `training_label` is sent
  - otherwise in `unlabeled/` for later labeling
- For training, keep labeled images in:
  - `data/skin_condition_images/<condition>/<image>.jpg`
  - or `data/skin_condition_images/labels.csv` with `image_name,condition`

```bash
python main.py train-skin-condition
```

This training now builds an ensemble of **at least 2 models** (`resnet18` + `efficientnet_b0`) and reports whether validation reaches the target accuracy (default `0.95`).

### 2.3 Configure environment for the backend

In `.env` (or your environment):

```env
# So Skin Assistant can forward chat logs and feedback to your backend
SPRING_BACKEND_URL=http://localhost:8080

# Optional: use your MySQL (skinme_db) for product search when "Check with database" is on
MYSQL_HOST=your-host
MYSQL_PORT=3306
MYSQL_USER=your_user
MYSQL_PASSWORD=your_password
MYSQL_DATABASE=skinme_db
```

### 2.4 Start the Skin Assistant API

```bash
python main.py
```

API base: `http://localhost:8000`. The API loads models from `models/artifacts/` and uses `data/` CSVs (and optional MySQL).

### 2.5 Use the trained AI from your backend

- Your backend (e.g. Spring) calls the Skin Assistant API:
  - **POST /v1/chat** — get assistant reply (uses intent + ingredients + products; LLM if `GEMINI_API_KEY` or `GOOGLE_API_KEY` set).
  - **GET /v1/intent?q=...** — intent prediction (uses trained intent model).
  - **GET /v1/products?concern=...** — product search (uses synced/SkinMe data or MySQL if `use_database=true`).
- If `SPRING_BACKEND_URL` is set, the Skin Assistant API forwards **POST /v1/chat/log** and **POST /v1/feedback** to your backend so you can persist chat and feedback in your database. Implement those endpoints in Spring as in [SPRING_BACKEND_INTEGRATION.md](SPRING_BACKEND_INTEGRATION.md).

---

## 3. Training Using Data From Your Backend

The intent model can be trained on real user messages. If your backend stores chat logs (e.g. user message + reply), you can:

1. **Export from your DB** into a CSV with at least:
   - `text` — user message
   - `intent` — one of: `greeting`, `ingredient_info`, `products_with_ingredient`, `product_recommendation`, `general_ingredient_search`, `other`

2. **Save as** `data/intent_training.csv` (see `data/intent_training.csv.example` for format).

3. **Train:**

   ```bash
   python main.py train
   ```
   or  
   `python -m scripts.train --data /path/to/your_export.csv --output models/artifacts`

If you don’t have intents in your DB, you can label a sample of messages and use that CSV; the rest can stay in the backend for analytics only.

---

## 4. Quick Reference

| Goal | Command |
|------|---------|
| Sync product data from SkinMe API | `python main.py sync` |
| Train intent model | `python main.py train` |
| Train product (text ± image) models | `python main.py train-products` or `... --image` |
| Train skin condition model | `python main.py train-skin-condition` |
| Run API (use trained models + backend URL) | `python main.py` |
| Run API after syncing product data | `python main.py --sync-first` |

All training is local; the backend only receives chat/feedback and calls the Skin Assistant API.
