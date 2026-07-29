"""
server.py
─────────
FastAPI backend for the News Sentiment Analysis live dashboard.
Serves the custom HTML/CSS/JS frontend and provides an API endpoint 
for Zero-Shot entity-centric framing detection.
"""

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import logging
import json
from transformers import pipeline
from collections import Counter
from pathlib import Path
import os

try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import spacy.cli
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("webapp")

import mlflow

# Setup MLflow Tracking
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "")
if tracking_uri:
    mlflow.set_tracking_uri(tracking_uri)
    log.info(f"MLflow tracking URI set to {tracking_uri}")
else:
    log.info("No MLFLOW_TRACKING_URI provided. Logging locally to ./mlruns")

mlflow.set_experiment("News-Sentiment-Live")

app = FastAPI(title="News Sentiment API")

# Mount the static directory
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

# Create static dir if it doesn't exist
os.makedirs(STATIC_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Load model globally
import torch
import torch.nn.functional as F

CUSTOM_MODEL_LOADED = False
custom_model = None
custom_tokenizer = None
classifier = None
classifier_light = None
device_int = 0 if torch.cuda.is_available() else -1

if os.getenv("USE_BART", "true").lower() != "true":
    log.info("USE_BART is set to false. Skipping BART-large to save memory.")
else:
    try:
        log.info("Loading Zero-Shot BART...")
        classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=device_int)
    except Exception as e:
        log.error(f"Failed to load BART fallback model: {e}")

if not classifier:
    try:
        log.info("Loading Lightweight Fallback (DistilBERT)...")
        classifier_light = pipeline("zero-shot-classification", model="typeform/distilbert-base-uncased-mnli", device=device_int)
    except Exception as e:
        log.error(f"Failed to load Lightweight fallback model: {e}")

try:
    device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    custom_model_dir = BASE_DIR.parent / "data" / "models" / "roberta_model"
    if not custom_model_dir.exists():
        custom_model_dir = BASE_DIR.parent / "data" / "models" / "dummy_test_roberta_model"
        
    if custom_model_dir.exists():
        import sys
        if str(BASE_DIR.parent) not in sys.path:
            sys.path.insert(0, str(BASE_DIR.parent))
            
        from models.roberta_framing import load_model, IDX_TO_LABEL
        log.info(f"Loading Custom Entity-Aware RoBERTa from {custom_model_dir}...")
        custom_model, custom_tokenizer = load_model(custom_model_dir, device=device_str)
        CUSTOM_MODEL_LOADED = True
    else:
        log.warning(f"No custom model found in {custom_model_dir}. Ensemble will only have BART.")
except Exception as e:
    log.error(f"Could not load custom model ({type(e).__name__}: {e}).")



class AnalysisRequest(BaseModel):
    url: str | None = None
    raw_text: str | None = None
    entity: str | None = None
    outlet: str | None = None


@app.get("/")
async def serve_index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.post("/api/analyze")
async def analyze(req: AnalysisRequest):
    if not CUSTOM_MODEL_LOADED and not classifier and not classifier_light:
        raise HTTPException(status_code=503, detail=f"All models failed to load. CUSTOM={CUSTOM_MODEL_LOADED}")
    
    if not req.url and not req.raw_text:
        raise HTTPException(status_code=400, detail="Must provide either url or raw_text.")
        
    text = ""
    title = "Raw Text Input"
    
    if req.raw_text:
        text = req.raw_text
        if req.outlet:
            title = f"{req.outlet} Article"
            
        # Log to database for data harvesting
        try:
            db_path = BASE_DIR.parent / "data" / "user_queries.jsonl"
            db_path.parent.mkdir(parents=True, exist_ok=True)
            with open(db_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"outlet": req.outlet, "text": text, "entity": req.entity}) + "\n")
        except Exception as e:
            log.warning(f"Failed to log user query: {e}")
    else:
        try:
            import newspaper
            from curl_cffi import requests
            
            # Spoof a real browser to bypass Cloudflare/Akamai 403s
            response = requests.get(req.url, impersonate="chrome120", timeout=15)
            response.raise_for_status()
            
            article = newspaper.Article(req.url)
            article.set_html(response.text)
            article.parse()
            text = article.text
            title = article.title
            if not text:
                raise HTTPException(status_code=400, detail="Failed to extract text from URL.")
        except ImportError:
            raise HTTPException(status_code=500, detail="newspaper3k or curl_cffi is not installed.")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to scrape article: {str(e)}")
        
    full_text = f"{title}\n\n{text}"
    full_text_lower = full_text.lower()
    
    import re
    def clean_ent_text(t: str) -> str:
        return re.sub(r"['’]s$", "", t.strip(), flags=re.IGNORECASE)

    valid_labels = {'PERSON', 'ORG', 'GPE', 'NORP', 'LOC', 'FAC', 'EVENT', 'PRODUCT'}

    title_doc = nlp(title)
    title_entities = [clean_ent_text(ent.text) for ent in title_doc.ents if ent.label_ in valid_labels and len(clean_ent_text(ent.text)) > 1]

    doc = nlp(text[:5000])
    body_entities = [clean_ent_text(ent.text) for ent in doc.ents if ent.label_ in valid_labels and len(clean_ent_text(ent.text)) > 1]

    # Give 3x weight to entities in title
    weighted_entities = title_entities * 3 + body_entities
    
    # Regex fallback for proper nouns if SpaCy misses title entities
    if not weighted_entities:
        proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', title)
        weighted_entities = [p for p in proper_nouns if len(p) > 2]

    # Auto-detect entity if not provided
    entity_clean = req.entity.strip() if req.entity else ""
    
    if not entity_clean:
        if not weighted_entities:
            raise HTTPException(status_code=404, detail="No entity provided, and Auto-NER failed to detect any prominent entities in the text.")
        
        counts = Counter(weighted_entities)
        most_common = [ent for ent, _ in counts.most_common(5)]
        
        # Prefer multi-word entities (e.g. "Shan Masood" over "Masood")
        selected = most_common[0]
        for candidate in most_common:
            if len(candidate.split()) > len(selected.split()):
                selected = candidate
        entity_clean = selected

    entity_lower = entity_clean.lower()
    tokens = [t.strip() for t in entity_lower.split() if len(t.strip()) > 2]
    matched = (entity_lower in full_text_lower) or (len(tokens) > 0 and any(t in full_text_lower for t in tokens))

    if not matched:
        raise HTTPException(
            status_code=404, 
            detail=f"The entity '{entity_clean}' was not found in the article text or headline. (Scraped {len(text)} chars from URL). Please check the spelling."
        )
        
    # Extract Entity-Centric Context: Prioritize paragraphs containing entity_clean or its main tokens
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    entity_tokens = [t.lower() for t in entity_clean.split() if len(t) > 2]
    
    matching_paragraphs = []
    for p in paragraphs:
        p_lower = p.lower()
        if entity_clean.lower() in p_lower or any(t in p_lower for t in entity_tokens):
            matching_paragraphs.append(p)
            
    if matching_paragraphs:
        selected_paragraphs = matching_paragraphs[:3]
    else:
        selected_paragraphs = paragraphs[:2]
        
    context = f"{title}\n\n" + "\n\n".join(selected_paragraphs)
    
    # Truncate if still too long (just in case)
    if len(context) > 2500:
        context = context[:2500] + "..."
        
    try:
        custom_scores = None
        bart_scores = None
        active_classifier = classifier or classifier_light
        fallback_name = "BART-Large" if classifier else ("DistilBERT" if classifier_light else "None")
        engine_name = f"Ensemble (Custom + {fallback_name})" if CUSTOM_MODEL_LOADED else fallback_name
        
        if CUSTOM_MODEL_LOADED and custom_model and custom_tokenizer:
            # Tokenize the context for Custom RoBERTa
            inputs = custom_tokenizer(
                context, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512
            )
            input_ids = inputs["input_ids"].to(device_str)
            attention_mask = inputs["attention_mask"].to(device_str)
            seq_len = input_ids.shape[1]
            
            # Compute REAL token proximity scores relative to entity_clean
            tokens = custom_tokenizer.convert_ids_to_tokens(input_ids[0])
            entity_indices = []
            for i, tok in enumerate(tokens):
                tok_clean = tok.replace('Ġ', '').lower()
                if tok_clean and any(t in tok_clean for t in entity_tokens):
                    entity_indices.append(i)
                    
            proximity_arr = torch.ones(seq_len, dtype=torch.float)
            if entity_indices:
                for i in range(seq_len):
                    min_dist = min(abs(i - e_idx) for e_idx in entity_indices)
                    proximity_arr[i] = 1.0 / (1.0 + min_dist)
            proximity_scores = proximity_arr.unsqueeze(0).to(device_str)
            
            with torch.no_grad():
                out = custom_model(input_ids, attention_mask, proximity_scores)
                logits = out["logits"]
                probs = F.softmax(logits, dim=-1).squeeze().tolist()
                
            custom_scores = {IDX_TO_LABEL[i]: prob for i, prob in enumerate(probs)}
            
        if active_classifier:
            candidate_labels = [
                "supportive or endorsing",
                "critical, blaming, or questioning",
                "neutral, factual reporting",
                "alarmist, crisis, or threatening"
            ]
            
            hypothesis_template = f"In this text, the entity '{entity_clean}' is portrayed in a way that is {{}}."
            
            result = active_classifier(
                context,
                candidate_labels,
                hypothesis_template=hypothesis_template,
                multi_label=False
            )
            label_map = {
                "supportive or endorsing": "Supportive",
                "critical, blaming, or questioning": "Critical",
                "neutral, factual reporting": "Neutral-Reporting",
                "alarmist, crisis, or threatening": "Alarmist"
            }
            bart_scores = {label_map[label]: float(score) for label, score in zip(result["labels"], result["scores"])} # type: ignore
            
        # Compute multi-entity comparative analysis for other top entities in the article
        other_entity_scores = {}
        top_entities = [ent for ent in Counter(weighted_entities).keys() if ent.lower() != entity_clean.lower()][:3]
        
        for o_ent in top_entities:
            o_tokens = [t.lower() for t in o_ent.split() if len(t) > 2]
            o_matching_p = [p for p in paragraphs if o_ent.lower() in p.lower() or any(t in p.lower() for t in o_tokens)]
            o_selected = o_matching_p[:2] if o_matching_p else paragraphs[:1]
            o_context = f"{title}\n\n" + "\n\n".join(o_selected)
            if len(o_context) > 1500:
                o_context = o_context[:1500]
                
            if active_classifier:
                o_res = active_classifier(
                    o_context,
                    candidate_labels,
                    hypothesis_template=f"In this news text, the target entity '{o_ent}' is portrayed in a way that is {{}}.",
                    multi_label=False
                )
                other_entity_scores[o_ent] = {label_map[l]: float(s) for l, s in zip(o_res["labels"], o_res["scores"])}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {str(e)}")
    
    # MLflow Real-Time Tracking
    try:
        with mlflow.start_run():
            mlflow.log_param("engine", engine_name)
            mlflow.log_param("entity", entity_clean)
            mlflow.log_param("context_length", len(context))
            if req.outlet:
                mlflow.log_param("outlet", req.outlet)
            
            if bart_scores:
                for label, score in bart_scores.items():
                    mlflow.log_metric(f"bart_{label.lower().replace('-', '_')}", score)
            if custom_scores:
                for label, score in custom_scores.items():
                    mlflow.log_metric(f"custom_{label.lower().replace('-', '_')}", score)
    except Exception as e:
        log.warning(f"Failed to log to MLflow: {e}")

    return {
        "title": title,
        "target_entity": entity_clean,
        "auto_entity": entity_clean if not req.entity else None,
        "context": context,
        "bart_scores": bart_scores,
        "custom_scores": custom_scores,
        "other_entity_scores": other_entity_scores,
        "engine": engine_name
    }
