# Chatbot API — Endpoints for Frontend

Use this base URL and endpoints so the frontend can connect to the chatbot.

---

## Base URL (after deploy)

```
https://chatbot.skinme.store
```

For local development:

```
http://localhost:8000
```

---

## Endpoints for chat

### 1. Health check

Check that the chatbot API is up.

| Method | URL | Response |
|--------|-----|----------|
| GET | `{BASE}/v1/health` | `{"status":"ok","service":"skin-assistant"}` |

---

### 2. Send a text message (main chat)

| Method | URL |
|--------|-----|
| POST | `{BASE}/v1/chat` |

**Headers:** `Content-Type: application/json`

**Body (JSON):**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| message | string | Yes | User message (1–2000 chars) |
| history | array | No | Previous messages `[{ "role": "user" \| "assistant", "content": "..." }]` (optional; DB history used when session_id is set) |
| use_llm | boolean | No | Use Gemini for replies when API key is set (default: true) |
| use_database | boolean | No | Use DB for product recommendations (default: false) |
| session_id | string | No | Chat session ID (e.g. UUID). If set, turn is saved to DB and history is loaded from DB. |
| user_id | string | No | Logged-in user ID from your auth. Stored in DB. |
| user_email | string | No | Logged-in user email. Stored in DB. |
| user_name | string | No | Logged-in user name. Stored in DB. |

**Example request:**

```json
{
  "message": "Recommend products for dry skin",
  "session_id": "abc-123-session-uuid",
  "user_id": "auth-user-uuid",
  "user_email": "user@example.com",
  "user_name": "Jane",
  "use_llm": true,
  "use_database": true
}
```

**Example response:**

```json
{
  "reply": "Here are some suggestions:\n\n**Ingredients that may help:**\n..."
}
```

---

### 3. Send a message with a skin image

| Method | URL |
|--------|-----|
| POST | `{BASE}/v1/chat/with-image` |

**Content-Type:** `multipart/form-data`

**Form fields:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| message | string | No | Optional text (e.g. "What do you recommend?") |
| image | file | Yes | Image file (jpg, png, webp) |
| session_id | string | No | Same as above |
| user_id | string | No | Same as above |
| user_email | string | No | Same as above |
| user_name | string | No | Same as above |
| use_llm | boolean | No | Default true |
| use_database | boolean | No | Default false |
| training_label | string | No | Optional label (e.g. `acne`) to save uploaded image directly into labeled training folder |

Every uploaded image is now saved automatically to one centralized training folder:

`data/skin_condition_images/`

- If `training_label` is provided: saved to `data/skin_condition_images/<training_label>/...`
- If no label is provided: saved to `data/skin_condition_images/unlabeled/...` for later labeling

**Example response:**

```json
{
  "reply": "...",
  "image_analysis": "acne (85%)"
}
```

---

### 4. Other useful endpoints

| Method | URL | Description |
|--------|-----|-------------|
| GET | `{BASE}/v1/ingredients/search?q=niacinamide` | Search ingredients |
| GET | `{BASE}/v1/ingredients/{name}` | Get one ingredient |
| GET | `{BASE}/v1/products?concern=dry&use_database=true` | Search products |
| GET | `{BASE}/v1/routes` | List all routes |
| GET | `{BASE}/docs` | OpenAPI (Swagger) UI |

---

## Frontend integration checklist

1. **Base URL**  
   Set in your app config, e.g.  
   `CHATBOT_API_URL = 'https://chatbot.skinme.store'`  
   (or env like `VITE_CHATBOT_API_URL`).

2. **Session ID**  
   Generate a UUID per chat session (e.g. when user opens chat) and send it with every `POST /v1/chat` and `POST /v1/chat/with-image` so history is loaded and saved in DB.

3. **Logged-in user**  
   After login, send `user_id` (and optionally `user_email`, `user_name`) in the chat request body so messages are stored under that user.

4. **CORS**  
   The API allows all origins. If you restrict later, allow `https://skinme.store` and `https://chatbot.skinme.store`.

---

## Example: fetch (JavaScript)

```javascript
const BASE = 'https://chatbot.skinme.store';

// Text chat
const res = await fetch(`${BASE}/v1/chat`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    message: 'Recommend products for dry skin',
    session_id: sessionId,
    user_id: user?.id,
    user_email: user?.email,
    user_name: user?.name,
    use_llm: true,
    use_database: true,
  }),
});
const data = await res.json();
console.log(data.reply);
```

---

## Example: with image (FormData)

```javascript
const form = new FormData();
form.append('message', 'What do you recommend?');
form.append('image', imageFile);
form.append('session_id', sessionId);
if (user?.id) form.append('user_id', user.id);

const res = await fetch(`${BASE}/v1/chat/with-image`, {
  method: 'POST',
  body: form,
});
const data = await res.json();
console.log(data.reply, data.image_analysis);
```
