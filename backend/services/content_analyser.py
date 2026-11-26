import os
import re
import numpy as np
from sentence_transformers import SentenceTransformer, util

_SBERT_MODEL = None
_CUSTOM_MODEL = None
CUSTOM_MODEL_PATH = os.getenv("CUSTOM_MODEL_PATH", "")

# ── Tuning knobs (change these if scores feel too harsh / too generous) ───────
REL_LOW = 0.10            # cosine at or below this = 0 relevance
REL_HIGH = 0.45           # cosine at or above this = 100 relevance
VOCAB_FULL_AT_WORDS = 50  # answers shorter than this get a reduced vocab score
NON_ANSWER_CAP = 15.0     # max content score for hostile / refusing answers

# Multi-word phrases only. Single common words ("i", "so", "like") removed
# because they match almost every answer.
STRUCTURE_MARKERS = {
    "intro": [
        "in my experience", "i would", "i'd like to", "to answer", "i believe",
        "the reason", "when i", "i chose", "i decided", "i was working",
        "i think", "i feel", "my role", "i worked", "i was responsible",
        "at that time", "first of all", "to begin with",
    ],
    "example": [
        "for example", "for instance", "specifically", "one time", "in one case",
        "i remember", "such as", "last year", "during my", "in my project",
        "an example", "once", "in college", "at my internship", "i built",
        "i developed", "i created", "i implemented",
    ],
    "result": [
        "as a result", "which led to", "because of this", "the outcome",
        "we achieved", "therefore", "this led to", "i learned", "resulted in",
        "because", "so that", "this helped", "which helped", "i realized",
        "i realised", "finally", "eventually", "after that", "then i",
    ],
    "conclusion": [
        "in conclusion", "to summarise", "to summarize", "overall", "in summary",
        "ultimately", "to wrap up", "in the end", "that is why", "that's why",
        "this taught me", "all in all", "looking back",
    ],
}

# Hostile / refusing answers get capped. Add your own phrases if needed.
NON_ANSWER_PATTERNS = [
    "why are you asking", "i am offended", "i'm offended", "none of your business",
    "i don't want to answer", "i do not want to answer", "i won't answer",
    "i refuse to", "stupid question", "waste of time", "that's a dumb question",
    "i'm not going to answer", "i am not going to answer",
]


def _get_sbert():
    global _SBERT_MODEL
    if _SBERT_MODEL is None:
        print("[ORION] Loading sentence-transformers model...")
        _SBERT_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _SBERT_MODEL


def _load_custom_model():
    global _CUSTOM_MODEL
    if _CUSTOM_MODEL is None and CUSTOM_MODEL_PATH and os.path.exists(CUSTOM_MODEL_PATH):
        import joblib
        try:
            _CUSTOM_MODEL = joblib.load(CUSTOM_MODEL_PATH)
            print(f"[ORION] Custom model loaded from {CUSTOM_MODEL_PATH}")
        except Exception as e:
            print(f"[ORION] Failed to load custom model: {e}")
    return _CUSTOM_MODEL


def _has_phrase(text: str, phrase: str) -> bool:
    return re.search(r"\b" + re.escape(phrase) + r"\b", text) is not None


def analyse_content(transcript: str, question_text: str) -> dict:
    """
    Returns content metrics and an overall content_score (0-100).
    """
    metrics = {}

    if not transcript or len(transcript.strip()) < 10:
        metrics["content_score"] = 0.0
        return metrics

    lower = transcript.lower()
    words = re.findall(r"\b[a-z']+\b", lower)
    word_count = len(words)

    # ── Relevance (stricter: 0.15 cosine -> 0, 0.55 cosine -> 100) ────────────
    sbert = _get_sbert()
    q_emb = sbert.encode(question_text, convert_to_tensor=True)
    a_emb = sbert.encode(transcript, convert_to_tensor=True)
    cosine = float(util.cos_sim(q_emb, a_emb).item())
    relevance_score = round(
        max(0.0, min(1.0, (cosine - REL_LOW) / (REL_HIGH - REL_LOW))) * 100, 1
    )
    metrics["relevance_score"] = relevance_score

    # ── Structure (whole-phrase matches only) ──────────────────────────────────
    structure_hits = {
        cat: [m for m in markers if _has_phrase(lower, m)]
        for cat, markers in STRUCTURE_MARKERS.items()
    }
    total_matches = sum(len(v) for v in structure_hits.values())
    # 5 different structure phrases = full marks
    structure_score = round(min(total_matches / 5, 1.0) * 100, 1)
    metrics["structure_score"] = structure_score
    metrics["has_example"] = len(structure_hits["example"]) > 0
    metrics["has_conclusion"] = len(structure_hits["conclusion"]) > 0

    # ── Vocabulary (scaled down for very short answers) ───────────────────────
    ttr = len(set(words)) / word_count if word_count else 0.0
    length_factor = min(word_count / VOCAB_FULL_AT_WORDS, 1.0)
    vocab_score = round(min(ttr * 150, 100) * length_factor, 1)
    metrics["vocab_richness"] = round(ttr, 3)
    metrics["vocab_score"] = vocab_score

    # ── Length adequacy (ideal 80-200 words) ──────────────────────────────────
    if 80 <= word_count <= 200:
        length_score = 100.0
    elif word_count < 80:
        length_score = max(0.0, word_count / 80 * 100)
    else:
        length_score = max(50.0, 100 - (word_count - 200) * 0.3)
    metrics["length_score"] = round(length_score, 1)

    # ── Custom model (if available) ───────────────────────────────────────────
    custom_boost = 0.0
    use_custom = False
    custom_model = _load_custom_model()
    if custom_model is not None:
        try:
            embedding = sbert.encode([transcript])
            prediction = custom_model.predict_proba(embedding)[0]
            custom_boost = round(float(prediction[2]) * 100, 1)  # "strong" probability
            metrics["custom_model_score"] = custom_boost
            use_custom = True
        except Exception as e:
            print(f"[ORION] Custom model inference failed: {e}")

    # ── Weighted score ────────────────────────────────────────────────────────
    if use_custom:
        weighted = (0.25 * relevance_score + 0.20 * structure_score +
                    0.15 * vocab_score + 0.15 * length_score + 0.25 * custom_boost)
    else:
        weighted = (0.35 * relevance_score + 0.25 * structure_score +
                    0.20 * vocab_score + 0.20 * length_score)

    # ── Relevance gate: off-topic answers cannot score well on other parts ────
    # relevance 0 -> keeps 30% of the score, relevance 50+ -> keeps 100%
    gate = 0.3 + 0.7 * min(relevance_score / 50.0, 1.0)
    content_score = weighted * gate

    # ── Hostile / refusing answers are capped ─────────────────────────────────
    non_answer = any(p in lower for p in NON_ANSWER_PATTERNS)
    metrics["non_answer"] = non_answer
    if non_answer:
        content_score = min(content_score, NON_ANSWER_CAP)

    metrics["content_score"] = round(max(0.0, min(100.0, content_score)), 1)
    return metrics
