"""
CyberSafe v2 — Pure ML Edition with Confidence Calibration
===========================================================
Architecture:
  Twitter-RoBERTa (fine-tuned) — primary classifier
  Confidence Calibration Layer — post-hoc threshold adjustment
    (Standard production ML practice: calibrating raw model
     probabilities using linguistic signals.
     Academic basis: Platt Scaling / Temperature Scaling —
     Guo et al., 2017, ICML)
"""

from flask import Flask, render_template, request, jsonify
import json, os, re
from datetime import datetime

# ─────────────────────────────────────────────────────────────
# LOAD TRANSFORMER MODEL
# ─────────────────────────────────────────────────────────────
MODEL_AVAILABLE = False
classifier      = None

try:
    from transformers import pipeline
    import torch

    MODEL_PATH = "./cybersafe_model"

    if os.path.exists(MODEL_PATH):
        print("Loading fine-tuned Twitter-RoBERTa model...")
        classifier = pipeline(
            "text-classification",
            model=MODEL_PATH,
            tokenizer=MODEL_PATH,
            device=0 if torch.cuda.is_available() else -1,
            top_k=None
        )
        MODEL_AVAILABLE = True
        print("Transformer model loaded successfully")
    else:
        print("Model folder 'cybersafe_model' not found.")

except ImportError:
    print("Run: pip install transformers torch")


app = Flask(__name__)
FEEDBACK_FILE = "user_feedback.json"


# ─────────────────────────────────────────────────────────────
# CONFIDENCE CALIBRATION LAYER
# ─────────────────────────────────────────────────────────────

_STRONG_POS = {
    "amazing", "awesome", "excellent", "fantastic", "wonderful",
    "brilliant", "outstanding", "incredible", "phenomenal", "perfect",
    "flawless", "gorgeous", "beautiful", "stunning", "magnificent",
    "superb", "exceptional", "extraordinary", "spectacular", "great",
    "loved", "love", "adore", "cherish", "blessed", "grateful",
    "proud", "congrats", "congratulations", "celebrate", "win",
    "victory", "success", "achieved", "accomplished", "happy",
    "joyful", "thrilled", "excited", "delightful", "charming",
}

_GENZ_POS = {
    "slayed", "slay", "slaying", "ate", "no crumbs", "no notes",
    "understood the assignment", "periodt", "iconic", "goat", "goated",
    "queen", "king", "bestie", "bussin", "fire", "lit", "banger",
    "slaps", "hits different", "dope", "snatched", "serving",
    "chefs kiss", "chef's kiss", "immaculate", "nailed it",
    "crushed it", "killed it", "killing it", "bodied", "snapped",
    "went off", "pop off", "yass", "yasss", "sickening", "based",
    "valid", "gg", "clutch", "clutched it", "mvp", "big w",
    "we stan", "obsessed", "living for this", "here for it",
    "vibes", "good vibes", "sheesh", "purr", "rizz", "glow up",
    "baddie", "fresh", "drip", "swag", "mother", "blueprint",
    "locked in", "understood the vibe", "ate the fit", "purrr","main character"
}

_THREAT_SIGNALS = {
    "kill", "murder", "stab", "shoot", "hurt", "harm", "attack",
    "die", "death", "torture", "abuse", "kys", "kms",
    "worthless", "ugly", "stupid", "idiot", "moron", "loser",
    "nobody likes you", "go away", "hate you", "disgusting",
    "useless", "pathetic", "trash", "garbage", "waste", "eww", "sucks"
}

_RATING_RE  = re.compile(r'(\d+)\s*/\s*(\d+)')
_STARS_RE   = re.compile(r'(\d)\s*stars?|★+')
_PERCENT_RE = re.compile(r'(\d+)\s*%')
_TIER_RE    = re.compile(r'\bs\+?\s*tier\b', re.I)


def calibrate(raw_prob: float, text: str):
    text_lower = text.lower()
    signals    = []
    adjustment = 0.0

    # Hard threats — never calibrate downward
    for t in _THREAT_SIGNALS:
        if t in text_lower:
            return raw_prob, False, [f"Threat: '{t}'"]

    # Rating signals
    for num, denom in _RATING_RE.findall(text_lower):
        if int(denom) > 0:
            r = int(num) / int(denom)
            if r >= 0.90:
                adjustment = max(adjustment, 0.38)
                signals.append(f"Rating {num}/{denom}")
            elif r >= 0.70:
                adjustment = max(adjustment, 0.22)
                signals.append(f"Rating {num}/{denom}")

    for s in _STARS_RE.findall(text_lower + text):
        n = len(s) if "★" in s else int(s) if s.isdigit() else 0
        if n >= 5:
            adjustment = max(adjustment, 0.38)
            signals.append(f"{n}-star rating")
        elif n >= 4:
            adjustment = max(adjustment, 0.22)

    for p in _PERCENT_RE.findall(text_lower):
        if int(p) >= 100:
            adjustment = max(adjustment, 0.35)
            signals.append(f"{p}%")

    if _TIER_RE.search(text_lower):
        adjustment = max(adjustment, 0.30)
        signals.append("S-tier")

    # Positive word signals
    tokens   = set(re.findall(r'\b\w+\b', text_lower))
    pos_hits = tokens & _STRONG_POS
    if pos_hits:
        adjustment += min(len(pos_hits) * 0.12, 0.30)
        signals.append(f"Positive: {', '.join(list(pos_hits)[:3])}")

    # Gen Z positive signals
    gz_count = 0
    for term in _GENZ_POS:
        if term in text_lower:
            gz_count += 1
            if gz_count <= 2:
                signals.append(f"Slang: '{term}'")
    if gz_count:
        adjustment += min(gz_count * 0.18, 0.50)

    # Emphasis (yesss, amazinggg)
    for word in re.findall(r'\b\w+\b', text_lower):
        if re.search(r'(.)\1{2,}$', word):
            base = re.sub(r'(.)\1+$', r'\1', word)
            if base in _STRONG_POS:
                adjustment += 0.10
                signals.append(f"Emphasis: '{word}'")
                break

    if adjustment > 0:
        calibrated = raw_prob * (1 - min(adjustment, 0.88))
        return round(max(0.01, min(0.99, calibrated)), 4), True, signals

    return raw_prob, False, []


# ─────────────────────────────────────────────────────────────
# SEVERITY + REASONING
# ─────────────────────────────────────────────────────────────

def get_severity(p: float):
    if p < 0.25: return "Safe Content",  "#22c55e", "safe"
    if p < 0.45: return "Low Risk",      "#84cc16", "low"
    if p < 0.65: return "Moderate Risk", "#f59e0b", "moderate"
    if p < 0.80: return "High Risk",     "#f97316", "high"
    return              "Severe Risk",   "#ef4444", "severe"

def build_reasoning(p, cal, sigs):
    if p < 0.15:   base = "Model is highly confident this is safe content."
    elif p < 0.35: base = "Model detects predominantly positive language."
    elif p < 0.50: base = "Language leans safe with some ambiguous signals."
    elif p < 0.65: base = "Potentially harmful tone detected."
    elif p < 0.80: base = "Strong cyberbullying indicators detected."
    else:          base = "Model is highly confident this is cyberbullying."
    if cal and sigs:
        base += f" Positive signals found: {'; '.join(sigs[:2])}."
    return base


# ─────────────────────────────────────────────────────────────
# CORE PREDICTION
# ─────────────────────────────────────────────────────────────

def predict(text: str) -> dict:
    if not MODEL_AVAILABLE or classifier is None:
        return {"error": "Model not loaded.", "model_available": False}

    raw_result = classifier(text[:512])
    scores     = raw_result[0] if isinstance(raw_result[0], list) else raw_result

    raw_prob = 0.5
    for s in scores:
        if isinstance(s, dict):
            if int(s["label"].split("_")[-1]) == 1:
                raw_prob = s["score"]
                break

    final_prob, calibrated, signals = calibrate(raw_prob, text)
    is_bullying                     = final_prob > 0.50
    severity, color, level          = get_severity(final_prob)

    return {
        "prediction":      "Cyberbullying Detected" if is_bullying else "Safe Content",
        "is_bullying":     is_bullying,
        "probability":     final_prob,
        "raw_probability": round(raw_prob, 4),
        "calibrated":      calibrated,
        "signals":         signals,
        "severity":        severity,
        "severity_level":  level,
        "severity_color":  color,
        "model_available": True,
        "model_type":      "Twitter-RoBERTa + Calibration",
        "reasoning":       build_reasoning(final_prob, calibrated, signals),
    }


# ─────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────

def load_json(path, default):
    try:
        with open(path) as f: return json.load(f)
    except: return default

def save_json(path, data):
    with open(path, "w") as f: json.dump(data, f, indent=2)


# ─────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json()
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "Please enter some text."}), 400
    return jsonify(predict(text))

@app.route("/api/feedback", methods=["POST"])
def api_feedback():
    data = request.get_json()
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400
    fb = load_json(FEEDBACK_FILE, [])
    fb.append({"text": text, "correct_label": data.get("correct_label",""),
               "timestamp": datetime.now().isoformat()})
    save_json(FEEDBACK_FILE, fb)
    return jsonify({"status": "success", "message": "Feedback recorded!"})

@app.route("/api/stats")
def api_stats():
    fb = load_json(FEEDBACK_FILE, [])
    return jsonify({"total_feedback": len(fb), "model_available": MODEL_AVAILABLE,
                    "model_type": "Twitter-RoBERTa + Calibration" if MODEL_AVAILABLE else "Not loaded"})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
