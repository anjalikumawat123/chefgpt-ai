"""
ChefGPT AI – Flask Backend
===========================
IBM watsonx.ai + Granite Models + Multi-Agent RAG Recipe Generator
"""

import os
import re
import json
import math
import uuid
import textwrap
from io import BytesIO
from pathlib import Path

from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

# ── Optional imports (handled gracefully if absent) ────────────────────────
try:
    import PyPDF2
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    SKLEARN_SUPPORT = True
except ImportError:
    SKLEARN_SUPPORT = False

try:
    import requests as http_requests
    REQUESTS_SUPPORT = True
except ImportError:
    REQUESTS_SUPPORT = False

# ── Load environment variables ──────────────────────────────────────────────
load_dotenv()

WATSONX_API_KEY    = os.getenv("WATSONX_API_KEY", "").strip()
WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID", "").strip()
WATSONX_URL        = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com").strip()
WATSONX_MODEL_ID   = os.getenv("WATSONX_MODEL_ID", "ibm/granite-13b-instruct-v2").strip()
SECRET_KEY         = os.getenv("FLASK_SECRET_KEY", "chefgpt-secret")
MAX_UPLOAD_MB      = int(os.getenv("MAX_UPLOAD_MB", "16"))

# Demo mode: True when IBM credentials are absent or DEMO_MODE=true in .env
_force_demo = os.getenv("DEMO_MODE", "").lower() in ("true", "1", "yes")
DEMO_MODE = _force_demo or not (WATSONX_API_KEY and WATSONX_PROJECT_ID)

# ── Flask app setup ─────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=".", static_url_path="")
app.secret_key = SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

ALLOWED_EXTENSIONS = {"pdf", "txt"}

# ── In-memory knowledge base ─────────────────────────────────────────────────
# Each entry: { "filename": str, "chunks": [str], "chunk_count": int }
knowledge_base: list[dict] = []
stats = {"recipes_generated": 0, "documents_uploaded": 0, "agents_active": 5}

# ══════════════════════════════════════════════════════════════════════════════
# SAMPLE / DEMO DATA
# ══════════════════════════════════════════════════════════════════════════════

SAMPLE_RECIPES = {
    "paneer butter masala": {
        "name": "Paneer Butter Masala",
        "description": "A rich and creamy North Indian curry made with soft paneer cubes in a luscious tomato-butter gravy.",
        "cuisine": "Indian", "difficulty": "Medium",
        "prep_time": "15 mins", "cook_time": "30 mins", "total_time": "45 mins", "servings": 4,
        "ingredients": [
            "250 g paneer, cubed", "3 large tomatoes, pureed", "1 large onion, finely chopped",
            "2 tbsp butter", "1 tbsp cream", "1 tsp ginger-garlic paste",
            "1 tsp red chilli powder", "1 tsp garam masala", "1 tsp kasuri methi",
            "Salt to taste", "Fresh coriander for garnish"
        ],
        "instructions": [
            "Heat butter in a pan over medium heat.",
            "Sauté onion until golden brown, then add ginger-garlic paste.",
            "Add tomato puree and cook until oil separates (about 10 minutes).",
            "Add red chilli powder, garam masala and salt; stir well.",
            "Add paneer cubes and mix gently to coat with the gravy.",
            "Stir in cream and kasuri methi; simmer for 5 minutes.",
            "Garnish with fresh coriander and serve hot with naan or rice."
        ],
        "tips": ["Use fresh paneer for best results.", "Add a pinch of sugar to balance acidity."],
        "nutrition": {"calories": 320, "protein": "14g", "carbs": "18g", "fat": "22g", "fiber": "3g"}
    },
    "vegetable biryani": {
        "name": "Vegetable Biryani",
        "description": "Aromatic basmati rice cooked with mixed vegetables and whole spices.",
        "cuisine": "Indian", "difficulty": "Medium",
        "prep_time": "20 mins", "cook_time": "40 mins", "total_time": "60 mins", "servings": 4,
        "ingredients": [
            "2 cups basmati rice", "1 cup mixed vegetables (carrot, peas, beans)",
            "1 large onion, sliced", "2 tbsp ghee", "1 tsp cumin seeds",
            "2 bay leaves", "4 cloves", "1 cinnamon stick",
            "1 tsp biryani masala", "Salt to taste", "Saffron soaked in warm milk"
        ],
        "instructions": [
            "Soak basmati rice for 30 minutes, then cook until 70% done.",
            "Heat ghee; add whole spices and sauté until fragrant.",
            "Add onion and cook until caramelised. Add vegetables and biryani masala.",
            "Layer half the rice, then the vegetable mixture, then remaining rice.",
            "Drizzle saffron milk on top. Seal and cook on low heat for 20 minutes.",
            "Gently mix before serving. Garnish with fried onions and mint."
        ],
        "tips": ["Always soak rice before cooking.", "Use dum cooking for authentic flavour."],
        "nutrition": {"calories": 380, "protein": "8g", "carbs": "65g", "fat": "10g", "fiber": "5g"}
    },
    "chocolate cake": {
        "name": "Chocolate Cake",
        "description": "A moist, rich chocolate cake perfect for celebrations.",
        "cuisine": "Continental", "difficulty": "Medium",
        "prep_time": "20 mins", "cook_time": "35 mins", "total_time": "55 mins", "servings": 8,
        "ingredients": [
            "1.5 cups all-purpose flour", "1 cup sugar", "0.5 cup cocoa powder",
            "2 eggs", "0.5 cup butter, melted", "1 cup milk",
            "1.5 tsp baking powder", "0.5 tsp baking soda", "1 tsp vanilla extract"
        ],
        "instructions": [
            "Preheat oven to 180°C. Grease and flour a 9-inch round pan.",
            "Sift together flour, cocoa, baking powder and baking soda.",
            "Beat eggs and sugar until fluffy. Add melted butter and vanilla.",
            "Alternately add dry ingredients and milk to the egg mixture.",
            "Pour batter into prepared pan and bake for 30-35 minutes.",
            "Cool completely before frosting."
        ],
        "tips": ["Don't over-mix the batter.", "Insert a toothpick to check doneness."],
        "nutrition": {"calories": 410, "protein": "6g", "carbs": "58g", "fat": "18g", "fiber": "2g"}
    },
    "masala dosa": {
        "name": "Masala Dosa",
        "description": "Crispy South Indian crepe filled with spiced potato filling.",
        "cuisine": "South Indian", "difficulty": "Hard",
        "prep_time": "8 hrs", "cook_time": "30 mins", "total_time": "8.5 hrs", "servings": 4,
        "ingredients": [
            "2 cups rice", "0.5 cup urad dal", "3 large potatoes, boiled",
            "1 onion, sliced", "1 tsp mustard seeds", "8-10 curry leaves",
            "0.5 tsp turmeric", "2 green chillies", "Salt to taste", "Oil for cooking"
        ],
        "instructions": [
            "Soak rice and urad dal separately for 6 hours. Grind and ferment overnight.",
            "For filling: heat oil, add mustard seeds and curry leaves.",
            "Add onion and chillies; sauté until soft.",
            "Add mashed potatoes and turmeric; mix well.",
            "Heat a non-stick pan, pour batter and spread in circles.",
            "Add filling in the centre, fold and serve with coconut chutney."
        ],
        "tips": ["Ferment the batter properly for crispy dosas.", "Use a hot iron skillet."],
        "nutrition": {"calories": 290, "protein": "7g", "carbs": "52g", "fat": "6g", "fiber": "4g"}
    },
    "vegetable pasta": {
        "name": "Vegetable Pasta",
        "description": "Quick Italian-style pasta tossed with fresh vegetables and herbs.",
        "cuisine": "Italian", "difficulty": "Easy",
        "prep_time": "10 mins", "cook_time": "20 mins", "total_time": "30 mins", "servings": 2,
        "ingredients": [
            "200 g penne pasta", "1 zucchini, diced", "1 bell pepper, sliced",
            "1 cup cherry tomatoes", "3 cloves garlic", "2 tbsp olive oil",
            "1 tsp dried oregano", "Salt and pepper", "Parmesan to serve"
        ],
        "instructions": [
            "Cook pasta in salted boiling water until al dente. Reserve 0.5 cup pasta water.",
            "Heat olive oil in a large pan. Sauté garlic until fragrant.",
            "Add zucchini and pepper; cook for 5 minutes.",
            "Add cherry tomatoes and oregano; cook 3 more minutes.",
            "Toss pasta with vegetables, adding pasta water as needed.",
            "Season and serve topped with Parmesan."
        ],
        "tips": ["Reserve pasta water to loosen the sauce.", "Don't overcook the vegetables."],
        "nutrition": {"calories": 340, "protein": "11g", "carbs": "55g", "fat": "9g", "fiber": "6g"}
    },
    "potato sandwich": {
        "name": "Aloo Sandwich",
        "description": "A quick, hearty Indian-style potato sandwich – perfect for breakfast.",
        "cuisine": "Indian", "difficulty": "Easy",
        "prep_time": "10 mins", "cook_time": "10 mins", "total_time": "20 mins", "servings": 2,
        "ingredients": [
            "4 slices bread", "2 large potatoes, boiled and mashed",
            "1 tsp chaat masala", "0.5 tsp red chilli powder",
            "Fresh coriander, chopped", "Green chutney", "Butter"
        ],
        "instructions": [
            "Mix mashed potato with chaat masala, chilli powder and coriander.",
            "Spread green chutney on bread slices.",
            "Spread the potato filling generously on half the slices.",
            "Top with the remaining slices to make sandwiches.",
            "Toast on a buttered griddle until golden on both sides.",
            "Serve hot with ketchup or chutney."
        ],
        "tips": ["Add grated cheese for extra richness.", "Toast on medium heat for even colour."],
        "nutrition": {"calories": 280, "protein": "7g", "carbs": "48g", "fat": "7g", "fiber": "4g"}
    }
}

SUBSTITUTIONS = {
    "egg":     [{"substitute": "Flax Egg (1 tbsp ground flax + 3 tbsp water)", "reason": "Acts as a binder"}],
    "eggs":    [{"substitute": "Yogurt (¼ cup per egg)", "reason": "Adds moisture and binding"}],
    "butter":  [{"substitute": "Coconut Oil", "reason": "Similar fat content and richness"},
                {"substitute": "Applesauce (for baking)", "reason": "Reduces fat while keeping moisture"}],
    "milk":    [{"substitute": "Oat Milk or Almond Milk", "reason": "Dairy-free with similar consistency"}],
    "cream":   [{"substitute": "Coconut Cream", "reason": "Rich and dairy-free"}],
    "paneer":  [{"substitute": "Tofu (firm)", "reason": "Similar texture, vegan-friendly"}],
    "cheese":  [{"substitute": "Nutritional Yeast", "reason": "Adds cheesy flavour without dairy"}],
    "sugar":   [{"substitute": "Honey or Maple Syrup", "reason": "Natural sweeteners with lower GI"},
                {"substitute": "Stevia", "reason": "Zero-calorie sugar substitute"}],
    "flour":   [{"substitute": "Almond Flour", "reason": "Gluten-free alternative"},
                {"substitute": "Oat Flour", "reason": "Nutritious gluten-free option"}],
    "chicken": [{"substitute": "Jackfruit", "reason": "Meaty texture, great for curries"},
                {"substitute": "Chickpeas", "reason": "High-protein plant-based option"}],
    "ghee":    [{"substitute": "Olive Oil", "reason": "Heart-healthy cooking fat"}],
    "yogurt":  [{"substitute": "Cashew Cream", "reason": "Vegan, creamy alternative"}],
}

DIETARY_ADAPTATIONS = {
    "vegetarian": {"remove": ["chicken", "mutton", "fish", "pork", "beef", "meat", "bacon"],
                   "replace": {"chicken": "paneer", "mutton": "mushroom", "fish": "tofu"}},
    "vegan":      {"remove": ["butter", "cream", "milk", "cheese", "ghee", "egg", "honey", "paneer", "yogurt"],
                   "replace": {"butter": "coconut oil", "cream": "coconut cream",
                               "milk": "oat milk", "paneer": "firm tofu", "ghee": "olive oil",
                               "egg": "flax egg", "yogurt": "cashew cream"}},
    "gluten-free":{"remove": ["flour", "bread", "pasta", "wheat"],
                   "replace": {"flour": "almond flour", "pasta": "rice pasta"}},
    "keto":       {"remove": ["rice", "potato", "bread", "sugar", "pasta"],
                   "replace": {"rice": "cauliflower rice", "potato": "turnip", "pasta": "zucchini noodles"}},
    "sugar-free": {"remove": ["sugar", "honey", "jaggery"],
                   "replace": {"sugar": "stevia", "honey": "monk fruit sweetener"}},
}

SHOPPING_CATEGORIES = {
    "Vegetables": ["tomato", "onion", "potato", "carrot", "pea", "bean", "spinach", "zucchini",
                   "capsicum", "pepper", "mushroom", "garlic", "ginger", "cauliflower", "broccoli",
                   "cabbage", "eggplant", "cucumber", "celery", "leek", "corn"],
    "Dairy":      ["milk", "butter", "cream", "cheese", "yogurt", "paneer", "ghee", "curd"],
    "Protein":    ["chicken", "egg", "tofu", "paneer", "lentil", "dal", "chickpea", "fish",
                   "mutton", "beef", "pork", "prawn", "shrimp"],
    "Spices":     ["salt", "pepper", "cumin", "coriander", "turmeric", "chilli", "garam masala",
                   "biryani masala", "chaat masala", "kasuri methi", "saffron", "cardamom",
                   "clove", "cinnamon", "bay leaf", "mustard seed", "oregano", "basil"],
    "Grains":     ["rice", "flour", "pasta", "bread", "oat", "semolina", "suji", "wheat",
                   "noodle", "quinoa", "barley"],
    "Fruits":     ["lemon", "lime", "tomato", "apple", "banana", "mango", "coconut"],
}

# ══════════════════════════════════════════════════════════════════════════════
# RAG UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract raw text from PDF bytes using PyPDF2."""
    if not PDF_SUPPORT:
        return ""
    reader = PyPDF2.PdfReader(BytesIO(file_bytes))
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n".join(pages)


def extract_text_from_txt(file_bytes: bytes) -> str:
    """Decode text file, trying UTF-8 first."""
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            return file_bytes.decode(enc)
        except UnicodeDecodeError:
            continue
    return file_bytes.decode("utf-8", errors="replace")


def clean_text(text: str) -> str:
    """Remove excess whitespace from extracted text."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 80) -> list[str]:
    """Split text into overlapping chunks by word count."""
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk.strip())
        i += chunk_size - overlap
    return chunks


def build_sample_knowledge_base() -> list[str]:
    """Convert SAMPLE_RECIPES into text chunks for demo retrieval."""
    chunks = []
    for recipe in SAMPLE_RECIPES.values():
        text = (
            f"Recipe: {recipe['name']}\n"
            f"Cuisine: {recipe['cuisine']}\n"
            f"Difficulty: {recipe['difficulty']}\n"
            f"Prep Time: {recipe['prep_time']}, Cook Time: {recipe['cook_time']}\n"
            f"Servings: {recipe['servings']}\n"
            f"Ingredients: {', '.join(recipe['ingredients'])}\n"
            f"Instructions: {' '.join(recipe['instructions'])}\n"
            f"Tips: {' '.join(recipe['tips'])}"
        )
        chunks.append(text)
    return chunks


SAMPLE_CHUNKS = build_sample_knowledge_base()


def retrieve_recipe_context(query: str, top_k: int = 3) -> list[dict]:
    """
    RAG Step – retrieve the most relevant chunks for the given query.
    Uses TF-IDF cosine similarity if scikit-learn is available,
    otherwise falls back to simple keyword overlap.
    """
    all_chunks: list[str] = []
    sources:    list[str] = []

    # Add user-uploaded document chunks first
    for doc in knowledge_base:
        for chunk in doc["chunks"]:
            all_chunks.append(chunk)
            sources.append(doc["filename"])

    # Always include sample chunks
    for chunk in SAMPLE_CHUNKS:
        all_chunks.append(chunk)
        sources.append("Sample Recipes (Built-in)")

    if not all_chunks:
        return []

    if SKLEARN_SUPPORT:
        vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
        tfidf_matrix = vectorizer.fit_transform(all_chunks + [query])
        doc_vectors   = tfidf_matrix[:-1]
        query_vector  = tfidf_matrix[-1]
        scores = cosine_similarity(query_vector, doc_vectors).flatten()
        top_indices = scores.argsort()[-top_k:][::-1]
        results = []
        for idx in top_indices:
            if scores[idx] > 0.0:
                results.append({
                    "chunk": all_chunks[idx],
                    "source": sources[idx],
                    "score": round(float(scores[idx]), 4),
                })
        return results
    else:
        # Fallback: keyword overlap scoring
        query_words = set(query.lower().split())
        scored = []
        for i, chunk in enumerate(all_chunks):
            chunk_words = set(chunk.lower().split())
            overlap = len(query_words & chunk_words)
            if overlap > 0:
                scored.append((overlap, i))
        scored.sort(reverse=True)
        return [
            {"chunk": all_chunks[i], "source": sources[i], "score": round(o / max(len(query_words), 1), 4)}
            for o, i in scored[:top_k]
        ]


# ══════════════════════════════════════════════════════════════════════════════
# IBM watsonx.ai INTEGRATION
# ══════════════════════════════════════════════════════════════════════════════

_iam_token_cache: dict = {}

def _get_iam_token() -> str:
    """Fetch (and cache) an IAM bearer token from IBM Cloud."""
    import time
    cached = _iam_token_cache.get("token")
    expiry = _iam_token_cache.get("expiry", 0)
    if cached and time.time() < expiry:
        return cached
    resp = http_requests.post(
        "https://iam.cloud.ibm.com/identity/token",
        data={"grant_type": "urn:ibm:params:oauth:grant-type:apikey", "apikey": WATSONX_API_KEY},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    token = data["access_token"]
    _iam_token_cache["token"] = token
    _iam_token_cache["expiry"] = time.time() + int(data.get("expires_in", 3600)) - 60
    return token


def call_watsonx(prompt: str, max_tokens: int = 700) -> str:
    """
    # ==================================================
    # IBM watsonx.ai / IBM Granite Model Integration
    # ==================================================
    Send a prompt to the IBM Granite model and return the generated text.
    Returns None on failure so callers can fall back to demo mode.
    """
    if not REQUESTS_SUPPORT:
        return None
    try:
        token = _get_iam_token()
        url = f"{WATSONX_URL}/ml/v1/text/generation?version=2023-05-29"
        payload = {
            "model_id": WATSONX_MODEL_ID,
            "project_id": WATSONX_PROJECT_ID,
            "input": prompt,
            "parameters": {
                "decoding_method": "greedy",
                "max_new_tokens": max_tokens,
                "min_new_tokens": 20,
                "stop_sequences": [],
                "repetition_penalty": 1.1,
            },
        }
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        resp = http_requests.post(url, json=payload, headers=headers, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data["results"][0]["generated_text"].strip()
    except Exception as e:
        app.logger.warning(f"IBM watsonx.ai call failed: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# AI AGENTS
# ══════════════════════════════════════════════════════════════════════════════

def recipe_retrieval_agent(query: str) -> dict:
    """
    Agent 1 – Recipe Retrieval Agent
    Searches the knowledge base and returns relevant recipe chunks.
    """
    results = retrieve_recipe_context(query)
    context = "\n\n".join(r["chunk"] for r in results)
    sources  = list({r["source"] for r in results})
    return {
        "agent": "Recipe Retrieval Agent",
        "status": "completed",
        "context": context,
        "sources": sources,
        "results": results,
    }


def recipe_generation_agent(query: str, context: str, preferences: dict) -> dict:
    """
    Agent 2 – Recipe Generation Agent
    Calls IBM Granite (or demo mode) to generate a personalised recipe.
    """
    cuisine   = preferences.get("cuisine", "Any")
    diet      = preferences.get("diet", "No Preference")
    time_pref = preferences.get("cooking_time", "Any")
    difficulty = preferences.get("difficulty", "Any")
    servings  = preferences.get("servings", 2)
    ingredients = preferences.get("ingredients", "")

    prompt = textwrap.dedent(f"""
        You are ChefGPT AI, an intelligent cooking assistant powered by IBM Granite.

        System: You are RecipeAI. Use retrieved recipe context when available.
        Respect dietary restrictions, available ingredients, cuisine, cooking time,
        difficulty and servings. Provide clear step-by-step instructions.

        Retrieved Recipe Context:
        {context or 'No documents uploaded. Use your general knowledge.'}

        User Request: {query}
        Available Ingredients: {ingredients}
        Cuisine Preference: {cuisine}
        Dietary Preference: {diet}
        Max Cooking Time: {time_pref}
        Difficulty: {difficulty}
        Servings: {servings}

        Generate a complete recipe with: Name, Description, Prep Time, Cook Time,
        Servings, Difficulty, Ingredients (list), Step-by-Step Instructions (numbered),
        Cooking Tips, and brief Nutrition Estimate.

        Recipe:
    """).strip()

    generated = None
    used_ibm  = False

    if not DEMO_MODE:
        generated = call_watsonx(prompt, max_tokens=800)
        if generated:
            used_ibm = True

    if not generated:
        # ── Demo mode: pick best matching sample recipe ──────────────────────
        query_lower = query.lower() + " " + ingredients.lower()
        best_key, best_score = None, -1
        for key in SAMPLE_RECIPES:
            score = sum(1 for word in key.split() if word in query_lower)
            if score > best_score:
                best_score, best_key = score, key
        recipe_data = SAMPLE_RECIPES.get(best_key, list(SAMPLE_RECIPES.values())[0])
        generated   = _recipe_dict_to_text(recipe_data)
        recipe_data = recipe_data.copy()
    else:
        recipe_data = _parse_recipe_text(generated)

    stats["recipes_generated"] += 1
    return {
        "agent": "Recipe Generation Agent",
        "status": "completed",
        "recipe_text": generated,
        "recipe_data": recipe_data,
        "used_ibm": used_ibm,
    }


def recipe_adaptation_agent(recipe_text: str, diet: str) -> dict:
    """
    Agent 3 – Recipe Adaptation Agent
    Adapts a recipe to the requested dietary preference.
    """
    diet_key = diet.lower().replace("-", "").replace(" ", "")
    adaptation_map = DIETARY_ADAPTATIONS.get(
        diet.lower().replace(" ", "-"),
        DIETARY_ADAPTATIONS.get(diet_key, {})
    )

    changes = []
    adapted = recipe_text

    for orig, repl in adaptation_map.get("replace", {}).items():
        if orig.lower() in adapted.lower():
            adapted = re.sub(re.escape(orig), repl, adapted, flags=re.IGNORECASE)
            changes.append(f"Replaced '{orig}' with '{repl}'")

    prompt = textwrap.dedent(f"""
        You are ChefGPT AI. Adapt the following recipe to be {diet}.
        List changes made and explain each substitution briefly.

        Original Recipe:
        {recipe_text}

        Dietary Goal: {diet}
        Adapted Recipe (make it {diet}):
    """).strip()

    adapted_text = None
    used_ibm = False
    if not DEMO_MODE:
        adapted_text = call_watsonx(prompt, max_tokens=600)
        if adapted_text:
            used_ibm = True

    if not adapted_text:
        adapted_text = adapted if changes else f"[Demo] {recipe_text}\n\nAdaptation Notes: {', '.join(changes) or 'No major changes needed for ' + diet}"

    return {
        "agent": "Recipe Adaptation Agent",
        "status": "completed",
        "original_diet": "Original",
        "target_diet": diet,
        "adapted_recipe": adapted_text,
        "changes": changes,
        "used_ibm": used_ibm,
    }


def substitution_agent(ingredient: str) -> dict:
    """
    Agent 4 – Ingredient Substitution Agent
    Returns substitution options for a given ingredient.
    """
    key = ingredient.lower().strip()
    substitutes = SUBSTITUTIONS.get(key, [])

    if not substitutes and not DEMO_MODE:
        prompt = f"What are 2 good substitutes for {ingredient} in cooking? Give substitute, quantity, and reason."
        result = call_watsonx(prompt, max_tokens=200)
        if result:
            return {
                "agent": "Ingredient Substitution Agent",
                "status": "completed",
                "ingredient": ingredient,
                "substitutes": [{"substitute": result, "reason": "IBM Granite suggestion"}],
                "used_ibm": True,
            }

    if not substitutes:
        substitutes = [{"substitute": f"Check local store for {ingredient} alternatives",
                        "reason": "No common substitution found in database"}]

    return {
        "agent": "Ingredient Substitution Agent",
        "status": "completed",
        "ingredient": ingredient,
        "substitutes": substitutes,
        "used_ibm": False,
    }


def nutrition_agent(recipe_text: str) -> dict:
    """
    Agent 5 – Nutrition Analysis Agent
    Estimates nutritional information for the generated recipe.
    """
    # Try to find a matching sample recipe for demo nutrition data
    for key, recipe in SAMPLE_RECIPES.items():
        if key in recipe_text.lower() or recipe["name"].lower() in recipe_text.lower():
            n = recipe["nutrition"]
            return {
                "agent": "Nutrition Analysis Agent",
                "status": "completed",
                "nutrition": n,
                "summary": f"Approximately {n['calories']} calories per serving with {n['protein']} protein.",
                "disclaimer": "Nutritional values are AI-generated estimates and are not medical advice.",
                "used_ibm": False,
            }

    if not DEMO_MODE:
        prompt = f"Estimate the nutrition per serving for this recipe (calories, protein, carbs, fat, fiber):\n{recipe_text[:500]}"
        result = call_watsonx(prompt, max_tokens=200)
        if result:
            return {
                "agent": "Nutrition Analysis Agent",
                "status": "completed",
                "nutrition": {"raw": result},
                "summary": result,
                "disclaimer": "Nutritional values are AI-generated estimates and are not medical advice.",
                "used_ibm": True,
            }

    return {
        "agent": "Nutrition Analysis Agent",
        "status": "completed",
        "nutrition": {"calories": "~350", "protein": "~12g", "carbs": "~45g", "fat": "~14g", "fiber": "~5g"},
        "summary": "Estimated ~350 calories per serving. Values may vary.",
        "disclaimer": "Nutritional values are AI-generated estimates and are not medical advice.",
        "used_ibm": False,
    }


def shopping_list_agent(ingredients: list[str], available: str = "") -> dict:
    """
    Agent 6 – Shopping List Agent
    Categorises ingredients and identifies items not in the user's pantry.
    """
    available_items = {a.strip().lower() for a in available.split(",") if a.strip()}
    categorised: dict[str, list[str]] = {cat: [] for cat in SHOPPING_CATEGORIES}
    categorised["Other"] = []

    for item in ingredients:
        item_clean = item.strip().lstrip("0123456789.- ")
        item_lower = item_clean.lower()

        # Skip if user already has it
        already_have = any(avail in item_lower for avail in available_items)
        if already_have:
            continue

        placed = False
        for category, keywords in SHOPPING_CATEGORIES.items():
            if any(kw in item_lower for kw in keywords):
                categorised[category].append(item_clean)
                placed = True
                break
        if not placed:
            categorised["Other"].append(item_clean)

    # Remove empty categories
    categorised = {k: v for k, v in categorised.items() if v}

    return {
        "agent": "Shopping List Agent",
        "status": "completed",
        "shopping_list": categorised,
        "total_items": sum(len(v) for v in categorised.values()),
        "used_ibm": False,
    }


def orchestrator_agent(query: str, preferences: dict) -> dict:
    """
    Master Orchestrator Agent
    Coordinates all sub-agents to produce a complete recipe response.
    """
    workflow = []

    # Step 1: Retrieval
    retrieval = recipe_retrieval_agent(query)
    workflow.append({"step": 1, "agent": "Recipe Retrieval Agent", "status": "completed",
                     "output_summary": f"Found {len(retrieval['results'])} relevant chunks"})

    # Step 2: Generation
    generation = recipe_generation_agent(query, retrieval["context"], preferences)
    workflow.append({"step": 2, "agent": "Recipe Generation Agent", "status": "completed",
                     "output_summary": f"Recipe generated {'via IBM Granite' if generation['used_ibm'] else '(Demo Mode)'}"})

    recipe_text = generation.get("recipe_text", "")
    recipe_data = generation.get("recipe_data", {})

    # Step 3: Adaptation (if dietary preference set)
    diet = preferences.get("diet", "No Preference")
    adaptation = None
    if diet and diet.lower() not in ("no preference", "any", ""):
        adaptation = recipe_adaptation_agent(recipe_text, diet)
        workflow.append({"step": 3, "agent": "Recipe Adaptation Agent", "status": "completed",
                         "output_summary": f"Adapted recipe to {diet}"})

    # Step 4: Nutrition
    nutrition = nutrition_agent(recipe_text)
    workflow.append({"step": 4, "agent": "Nutrition Analysis Agent", "status": "completed",
                     "output_summary": nutrition["summary"]})

    # Step 5: Shopping list
    ing_list = recipe_data.get("ingredients", [])
    if not ing_list:
        ing_list = _extract_ingredients_from_text(recipe_text)
    shopping = shopping_list_agent(ing_list, preferences.get("ingredients", ""))
    workflow.append({"step": 5, "agent": "Shopping List Agent", "status": "completed",
                     "output_summary": f"{shopping['total_items']} items categorised"})

    return {
        "orchestrator": "Master Orchestrator",
        "status": "completed",
        "query": query,
        "workflow": workflow,
        "retrieval": retrieval,
        "generation": generation,
        "adaptation": adaptation,
        "nutrition": nutrition,
        "shopping": shopping,
        "demo_mode": DEMO_MODE or not generation.get("used_ibm"),
        "used_ibm": generation.get("used_ibm", False),
    }


# ══════════════════════════════════════════════════════════════════════════════
# HELPER UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def _recipe_dict_to_text(recipe: dict) -> str:
    lines = [
        f"Recipe Name: {recipe['name']}",
        f"Description: {recipe['description']}",
        f"Cuisine: {recipe['cuisine']}  |  Difficulty: {recipe['difficulty']}",
        f"Prep Time: {recipe['prep_time']}  |  Cook Time: {recipe['cook_time']}  |  Total: {recipe['total_time']}",
        f"Servings: {recipe['servings']}",
        "",
        "Ingredients:",
    ]
    for ing in recipe["ingredients"]:
        lines.append(f"  - {ing}")
    lines.append("")
    lines.append("Instructions:")
    for i, step in enumerate(recipe["instructions"], 1):
        lines.append(f"  {i}. {step}")
    lines.append("")
    lines.append("Cooking Tips:")
    for tip in recipe.get("tips", []):
        lines.append(f"  - {tip}")
    return "\n".join(lines)


def _parse_recipe_text(text: str) -> dict:
    """Best-effort parse of free-form IBM Granite recipe text into a dict."""
    data = {"name": "Generated Recipe", "description": "", "ingredients": [], "instructions": [],
            "tips": [], "prep_time": "N/A", "cook_time": "N/A", "servings": 2, "difficulty": "Medium",
            "cuisine": "Any"}

    name_m = re.search(r"(?:Recipe Name|Name)[:\s]+(.+)", text, re.IGNORECASE)
    if name_m:
        data["name"] = name_m.group(1).strip()

    desc_m = re.search(r"Description[:\s]+(.+?)(?:\n[A-Z]|\Z)", text, re.IGNORECASE | re.DOTALL)
    if desc_m:
        data["description"] = desc_m.group(1).strip()[:300]

    ing_m = re.search(r"Ingredients?[:\s]+(.*?)(?:Instructions?|Steps?|Directions?|Method)", text, re.IGNORECASE | re.DOTALL)
    if ing_m:
        lines = ing_m.group(1).strip().split("\n")
        data["ingredients"] = [l.strip().lstrip("-•* ") for l in lines if l.strip()]

    step_m = re.search(r"(?:Instructions?|Steps?|Directions?|Method)[:\s]+(.*?)(?:Tips?|Notes?|\Z)", text, re.IGNORECASE | re.DOTALL)
    if step_m:
        lines = step_m.group(1).strip().split("\n")
        data["instructions"] = [l.strip().lstrip("0123456789.) ") for l in lines if l.strip()]

    return data


def _extract_ingredients_from_text(text: str) -> list[str]:
    """Extract a simple ingredient list from free-form text."""
    ing_m = re.search(r"Ingredients?[:\s]+(.*?)(?:Instructions?|Steps?|\Z)", text, re.IGNORECASE | re.DOTALL)
    if ing_m:
        lines = ing_m.group(1).strip().split("\n")
        return [l.strip().lstrip("-•* ") for l in lines if l.strip()]
    return []


# ══════════════════════════════════════════════════════════════════════════════
# FLASK ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    """Serve the main HTML page."""
    return send_from_directory(".", "index.html")


@app.route("/style.css")
def serve_css():
    """Serve the stylesheet."""
    return send_from_directory(".", "style.css")


@app.route("/status")
def status():
    """Return current app status (demo mode, IBM connection)."""
    return jsonify({
        "demo_mode": DEMO_MODE,
        "ibm_connected": not DEMO_MODE,
        "model": WATSONX_MODEL_ID if not DEMO_MODE else "Demo Mode",
        "stats": stats,
    })


@app.route("/generate", methods=["POST"])
def generate():
    """
    POST /generate
    Body: { query, ingredients, cuisine, diet, cooking_time, difficulty, servings }
    Returns: full orchestrator result
    """
    data = request.get_json(force=True) or {}
    query = data.get("query", "").strip()
    if not query and not data.get("ingredients", "").strip():
        return jsonify({"error": "Please enter a query or ingredients."}), 400

    if not query:
        query = f"Generate a recipe using {data.get('ingredients')}"

    preferences = {
        "ingredients": data.get("ingredients", ""),
        "cuisine":     data.get("cuisine", "Any"),
        "diet":        data.get("diet", "No Preference"),
        "cooking_time": data.get("cooking_time", "Any"),
        "difficulty":  data.get("difficulty", "Any"),
        "servings":    data.get("servings", 2),
    }

    result = orchestrator_agent(query, preferences)
    return jsonify(result)


@app.route("/upload", methods=["POST"])
def upload():
    """
    POST /upload
    Multipart form with field 'file'.
    Extracts text, chunks it and stores in in-memory knowledge base.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file provided."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Only PDF and TXT files are supported."}), 400

    filename = secure_filename(file.filename)
    file_bytes = file.read()

    ext = filename.rsplit(".", 1)[1].lower()
    if ext == "pdf":
        if not PDF_SUPPORT:
            return jsonify({"error": "PyPDF2 not installed. Only TXT files are supported."}), 400
        raw_text = extract_text_from_pdf(file_bytes)
    else:
        raw_text = extract_text_from_txt(file_bytes)

    if not raw_text.strip():
        return jsonify({"error": "Could not extract text from the uploaded file."}), 400

    text    = clean_text(raw_text)
    chunks  = chunk_text(text)

    # Remove existing entry with same filename
    global knowledge_base
    knowledge_base = [d for d in knowledge_base if d["filename"] != filename]
    knowledge_base.append({"filename": filename, "chunks": chunks, "chunk_count": len(chunks)})
    stats["documents_uploaded"] += 1

    return jsonify({
        "message": f"'{filename}' processed successfully. {len(chunks)} chunks indexed.",
        "filename": filename,
        "chunk_count": len(chunks),
        "status": "Ready",
    })


@app.route("/documents", methods=["GET"])
def list_documents():
    """GET /documents – list all uploaded documents."""
    docs = [{"filename": d["filename"], "chunk_count": d["chunk_count"], "status": "Ready"}
            for d in knowledge_base]
    return jsonify({"documents": docs})


@app.route("/documents/<filename>", methods=["DELETE"])
def delete_document(filename):
    """DELETE /documents/<filename> – remove a document from the knowledge base."""
    global knowledge_base
    before = len(knowledge_base)
    knowledge_base = [d for d in knowledge_base if d["filename"] != filename]
    if len(knowledge_base) < before:
        return jsonify({"message": f"'{filename}' removed."})
    return jsonify({"error": "Document not found."}), 404


@app.route("/search", methods=["POST"])
def search():
    """POST /search – natural language recipe search."""
    data  = request.get_json(force=True) or {}
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"error": "Please enter a search query."}), 400

    results = retrieve_recipe_context(query, top_k=5)
    return jsonify({"query": query, "results": results, "count": len(results)})


@app.route("/adapt", methods=["POST"])
def adapt():
    """POST /adapt – adapt a recipe for a dietary preference."""
    data   = request.get_json(force=True) or {}
    recipe = data.get("recipe", "").strip()
    diet   = data.get("diet", "").strip()
    if not recipe or not diet:
        return jsonify({"error": "Please provide both recipe text and dietary preference."}), 400

    result = recipe_adaptation_agent(recipe, diet)
    return jsonify(result)


@app.route("/substitute", methods=["POST"])
def substitute():
    """POST /substitute – get ingredient substitution suggestions."""
    data       = request.get_json(force=True) or {}
    ingredient = data.get("ingredient", "").strip()
    if not ingredient:
        return jsonify({"error": "Please enter an ingredient."}), 400

    result = substitution_agent(ingredient)
    return jsonify(result)


@app.route("/nutrition", methods=["POST"])
def nutrition():
    """POST /nutrition – estimate nutrition for a recipe."""
    data   = request.get_json(force=True) or {}
    recipe = data.get("recipe", "").strip()
    if not recipe:
        return jsonify({"error": "Please provide recipe text."}), 400

    result = nutrition_agent(recipe)
    return jsonify(result)


@app.route("/shopping-list", methods=["POST"])
def shopping_list():
    """POST /shopping-list – generate a categorised shopping list."""
    data        = request.get_json(force=True) or {}
    ingredients = data.get("ingredients", [])
    available   = data.get("available", "")

    if not ingredients:
        return jsonify({"error": "Please provide a list of ingredients."}), 400

    result = shopping_list_agent(ingredients, available)
    return jsonify(result)


@app.route("/chat", methods=["POST"])
def chat():
    """POST /chat – conversational cooking assistant."""
    data    = request.get_json(force=True) or {}
    message = data.get("message", "").strip()
    if not message:
        return jsonify({"error": "Please enter a message."}), 400

    retrieval = recipe_retrieval_agent(message)
    context   = retrieval["context"]
    sources   = retrieval["sources"]

    prompt = textwrap.dedent(f"""
        You are ChefGPT AI, a friendly and knowledgeable cooking assistant powered by IBM Granite.
        Use the recipe context below when relevant. Keep answers concise and practical.

        Recipe Context:
        {context or 'No documents uploaded. Use your general culinary knowledge.'}

        User: {message}
        ChefGPT AI:
    """).strip()

    response_text = None
    used_ibm = False
    if not DEMO_MODE:
        response_text = call_watsonx(prompt, max_tokens=400)
        if response_text:
            used_ibm = True

    if not response_text:
        response_text = _demo_chat_response(message)

    return jsonify({
        "response": response_text,
        "sources": sources,
        "used_ibm": used_ibm,
        "demo_mode": not used_ibm,
    })


def _demo_chat_response(message: str) -> str:
    """Generate a contextually relevant demo chat response."""
    msg = message.lower()

    if any(w in msg for w in ["paneer", "panir"]):
        r = SAMPLE_RECIPES["paneer butter masala"]
        return (f"Great choice! Here's a quick guide for {r['name']}:\n\n"
                f"Ingredients: {', '.join(r['ingredients'][:5])}...\n\n"
                f"Step 1: {r['instructions'][0]}\n"
                f"Step 2: {r['instructions'][1]}\n\nTip: {r['tips'][0]}")

    if any(w in msg for w in ["cake", "chocolate"]):
        r = SAMPLE_RECIPES["chocolate cake"]
        return (f"Here's a delicious {r['name']} recipe!\n\n"
                f"Key ingredients: {', '.join(r['ingredients'][:4])}\n\n"
                f"{r['instructions'][0]}\n{r['instructions'][1]}\n\nTip: {r['tips'][0]}")

    if any(w in msg for w in ["biryani", "rice"]):
        r = SAMPLE_RECIPES["vegetable biryani"]
        return (f"{r['name']} – aromatic and delicious!\n\n"
                f"You'll need: {', '.join(r['ingredients'][:5])}\n\n"
                f"Quick tip: {r['tips'][0]}")

    if any(w in msg for w in ["substitute", "replace", "instead"]):
        return ("Common substitutions:\n"
                "• Eggs → Flax egg (1 tbsp ground flax + 3 tbsp water)\n"
                "• Butter → Coconut oil\n"
                "• Milk → Oat milk or almond milk\n"
                "• Paneer → Firm tofu\n\nLet me know if you need a specific ingredient!")

    if any(w in msg for w in ["vegan", "vegetarian"]):
        return ("For vegan cooking, great swaps include:\n"
                "• Use coconut oil instead of butter or ghee\n"
                "• Replace paneer with firm tofu\n"
                "• Use oat milk instead of dairy milk\n"
                "• Try jackfruit as a meat substitute in curries!")

    if any(w in msg for w in ["quick", "fast", "20 min", "30 min"]):
        r = SAMPLE_RECIPES["potato sandwich"]
        return (f"For a quick meal, try {r['name']} – ready in just {r['total_time']}!\n\n"
                f"You need: {', '.join(r['ingredients'][:4])}\n\n"
                f"{r['instructions'][0]}")

    return ("I'm ChefGPT AI, your intelligent cooking companion! 🍳\n\n"
            "I can help you with:\n"
            "• Recipe generation from your ingredients\n"
            "• Ingredient substitutions\n"
            "• Dietary adaptations (vegan, gluten-free, etc.)\n"
            "• Nutrition information\n"
            "• Shopping lists\n\n"
            "Try asking: 'What can I make with paneer and tomatoes?' or "
            "'Give me a vegan pasta recipe!'")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  ChefGPT AI – Intelligent Recipe Generator")
    print("  Powered by IBM watsonx.ai & Granite Models")
    print("=" * 60)
    print(f"  Mode   : {'DEMO MODE (no IBM credentials)' if DEMO_MODE else '✓ IBM Granite Connected'}")
    print(f"  Model  : {WATSONX_MODEL_ID}")
    print(f"  Server : http://127.0.0.1:5000")
    print("=" * 60)
    app.run(debug=True, port=5000, use_reloader=False)
