# ChefGPT AI 🍳

> IBM watsonx.ai + Granite Models + Multi-Agent RAG Recipe Generator

A Flask web application that uses IBM watsonx.ai (Granite model) with a
multi-agent Retrieval-Augmented Generation (RAG) pipeline to generate
personalised recipes, suggest ingredient substitutions, estimate nutrition,
and produce categorised shopping lists.

---

## Project Files

| File | Purpose |
|------|---------|
| `app.py` | Flask backend – all agents & API routes |
| `index.html` | Frontend UI |
| `style.css` | Stylesheet |
| `requirements.txt` | Python dependencies |
| `.env.example` | Template for environment variables (safe to commit) |
| `.env` | **Your real secrets – NEVER commit this file** |

---

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables (API Key — Keep Secret!)
```bash
# Copy the example file
copy .env.example .env       # Windows
cp .env.example .env         # macOS / Linux
```

Now open `.env` in any text editor and fill in your real IBM credentials:
```
WATSONX_API_KEY=your_real_api_key_here
WATSONX_PROJECT_ID=your_real_project_id_here
```

> ⚠️ **IMPORTANT:** `.env` is listed in `.gitignore` and will **never** be
> uploaded to GitHub. Your API key stays on your machine only.

### 5. Run the Application
```bash
python app.py
```

Open your browser at **http://localhost:5000**

---

## How the Multi-Agent System Works

```
User Request
    │
    ▼
Orchestrator Agent
    ├── Agent 1: Recipe Retrieval   (RAG / TF-IDF search)
    ├── Agent 2: Recipe Generation  (IBM Granite LLM)
    ├── Agent 3: Recipe Adaptation  (dietary preferences)
    ├── Agent 4: Nutrition Analysis (calorie estimates)
    └── Agent 5: Shopping List      (categorised ingredients)
```

---

## IBM watsonx.ai Integration

- **Model:** IBM Granite (`ibm/granite-13b-instruct-v2`)
- **Authentication:** IBM Cloud IAM API Key
- **Fallback:** Demo mode with pre-built sample recipes when credentials
  are absent — so the app always works.

---

## ⚠️ Security — Protecting Your API Key

| ✅ DO | ❌ DO NOT |
|-------|----------|
| Store key in `.env` | Hard-code key in `app.py` |
| Add `.env` to `.gitignore` | Commit `.env` to GitHub |
| Share `.env.example` (no real values) | Share your `.env` file |
| Rotate key if accidentally exposed | Leave an exposed key active |

---

## Deployment (Local)

```bash
python app.py
# → Running on http://0.0.0.0:5000
```

---

*Built with IBM watsonx.ai & Flask*
