"""
auto_annotate.py
────────────────
Module 2.5: Automated Zero-Shot Dataset Annotation

This script reads the preprocessed articles, extracts the primary entities and surrounding text, 
and uses a zero-shot LLM (facebook/bart-large-mnli) to assign a framing label:
Supportive, Critical, Neutral-Reporting, or Alarmist.

This replaces the grueling manual annotation step with a weak-supervision automated dataset generation.
"""

import json
import logging
from pathlib import Path
from tqdm import tqdm
from transformers import pipeline

log = logging.getLogger(__name__)

def annotate_dataset(
    processed_path: Path, 
    output_path: Path, 
    batch_size: int = 8,
    limit: int = 0
) -> int:
    """
    Reads processed_articles.jsonl, annotates them using zero-shot classification,
    and writes to annotated_articles.jsonl.
    
    Args:
        processed_path: Path to input JSONL
        output_path: Path to output JSONL
        batch_size: Batch size for the transformer model
        limit: If > 0, stops after annotating this many articles.
        
    Returns:
        Number of articles successfully annotated.
    """
    if not processed_path.exists():
        log.error(f"[annotate] Cannot find {processed_path}. Run preprocess stage first.")
        return 0

    # 1. Read all processed articles
    articles = []
    with open(processed_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                articles.append(json.loads(line))
                
    if limit > 0:
        articles = articles[:limit]
        
    if not articles:
        log.warning("[annotate] No articles found to annotate.")
        return 0
        
    log.info(f"[annotate] Loaded {len(articles)} articles for zero-shot annotation.")
    
    # 2. Load the zero-shot classifier
    # Using device=0 if CUDA is available, otherwise -1 (CPU)
    import torch
    device = 0 if torch.cuda.is_available() else -1
    
    log.info(f"[annotate] Loading zero-shot classifier (facebook/bart-large-mnli) on device {device}...")
    try:
        classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=device)
    except Exception as e:
        log.warning(f"[annotate] Failed to load facebook/bart-large-mnli (possibly OOM): {e}")
        log.info("[annotate] Falling back to lightweight typeform/distilbert-base-uncased-mnli...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            classifier = pipeline("zero-shot-classification", model="typeform/distilbert-base-uncased-mnli", device=device)
        except Exception as e2:
            log.warning(f"[annotate] Failed to load lightweight model on GPU: {e2}")
            log.info("[annotate] Falling back to CPU...")
            try:
                classifier = pipeline("zero-shot-classification", model="typeform/distilbert-base-uncased-mnli", device=-1)
            except Exception as e3:
                log.error(f"[annotate] All model fallbacks failed: {e3}")
                return 0

    candidate_labels = [
        "supportive or endorsing",
        "critical, blaming, or questioning",
        "neutral, factual reporting",
        "alarmist, crisis, or threatening"
    ]
    
    label_map = {
        "supportive or endorsing": "Supportive",
        "critical, blaming, or questioning": "Critical",
        "neutral, factual reporting": "Neutral-Reporting",
        "alarmist, crisis, or threatening": "Alarmist"
    }
    
    # 3. Handle Resume (Checkpointing)
    annotated_ids = set()
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        annotated_ids.add(json.loads(line).get("article_id"))
                    except:
                        pass
        log.info(f"[annotate] Found existing {output_path.name} with {len(annotated_ids)} annotated articles. Resuming...")
    
    # Filter out already annotated articles
    articles_to_process = [a for a in articles if a.get("article_id") not in annotated_ids]
    
    if not articles_to_process:
        log.info("[annotate] All articles are already annotated.")
        return len(annotated_ids)
        
    log.info(f"[annotate] Beginning annotation of {len(articles_to_process)} remaining articles...")
    annotated_count = len(annotated_ids)
    
    # Open in append mode so we don't overwrite previous progress
    with open(output_path, "a", encoding="utf-8") as out_f:
        for art in tqdm(articles_to_process, desc="Annotating"):
            text = art.get("clean_body", "") or art.get("body", "")
            title = art.get("title", "")
            entities = art.get("entities", [])
            primary_entity = entities[0] if entities else None
            
            if not text:
                continue
                
            # Extract Entity-Centric Context: Prioritize paragraphs containing primary_entity
            if primary_entity:
                paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
                entity_tokens = [t.lower() for t in primary_entity.split() if len(t) > 2]
                matching_p = [p for p in paragraphs if primary_entity.lower() in p.lower() or any(t in p.lower() for t in entity_tokens)]
                selected = matching_p[:3] if matching_p else paragraphs[:2]
                context = f"{title}\n\n" + "\n\n".join(selected)
            else:
                context = f"{title}\n\n{text}"
                
            if len(context) > 2500:
                context = context[:2500]
            
            if primary_entity:
                hypothesis = f"In this news text, the target entity '{primary_entity}' is portrayed in a way that is {{}}."
            else:
                hypothesis = "The framing of this news article is {}."
                
            try:
                result = classifier(
                    context,
                    candidate_labels,
                    hypothesis_template=hypothesis,
                    multi_label=False
                )
                
                best_label = result["labels"][0] # type: ignore
                mapped_label = label_map[best_label]
                
                # Assign the label to the article
                art["label"] = mapped_label
                
                # Write to output file incrementally to save progress
                out_f.write(json.dumps(art) + "\n")
                annotated_count += 1
                
            except Exception as e:
                log.warning(f"[annotate] Failed to annotate article {art.get('id', 'Unknown')}: {e}")
                
    log.info(f"[annotate] Successfully auto-annotated {annotated_count} articles.")
    log.info(f"[annotate] Dataset saved to {output_path}")
    
    return annotated_count
