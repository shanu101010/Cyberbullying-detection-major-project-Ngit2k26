"""
Sentiment & Context Engine
--------------------------
Solves the core problems:
1. Unknown positive words being flagged as bullying
2. Gen Z slang misclassification
3. Context-aware ensemble scoring

Architecture:
  - Rule-based sentiment lexicon (positive/negative word lists)
  - Gen Z slang dictionary with contextual labels
  - Negation detection
  - Intensity modifiers
  - Final ensemble score combining all signals
"""

import re
from typing import Tuple, Dict

# ─────────────────────────────────────────────
# 1. POSITIVE WORD LEXICON (domain-expanded)
# ─────────────────────────────────────────────
POSITIVE_WORDS = set([
    # Classic positives
    "awesome", "amazing", "wonderful", "fantastic", "great", "excellent",
    "brilliant", "superb", "outstanding", "magnificent", "incredible",
    "beautiful", "gorgeous", "lovely", "adorable", "cute", "sweet",
    "happy", "joyful", "cheerful", "delightful", "pleased", "glad",
    "love", "loved", "loving", "adore", "cherish", "appreciate",
    "kind", "generous", "caring", "compassionate", "thoughtful",
    "brave", "courageous", "strong", "resilient", "talented",
    "smart", "intelligent", "clever", "creative", "innovative",
    "helpful", "supportive", "encouraging", "inspiring", "motivating",
    "fun", "exciting", "thrilling", "enjoyable", "pleasant",
    "perfect", "flawless", "exceptional", "remarkable", "extraordinary",
    "good", "nice", "cool", "great", "fine", "okay", "well",
    "congrats", "congratulations", "celebrate", "proud", "honor",
    "respect", "admire", "praise", "compliment", "cheer",
    "friend", "friendship", "together", "unity", "team",
    "yes", "absolutely", "definitely", "certainly", "sure",
    "win", "winner", "success", "achieve", "accomplish",
    "hope", "faith", "trust", "believe", "confident",
    # Internet-era positives
    "iconic", "legendary", "goat", "elite", "fire", "lit",
    "valid", "real", "true", "facts", "based",
])

# ─────────────────────────────────────────────
# 2. NEGATIVE / THREAT WORD LEXICON
# ─────────────────────────────────────────────
NEGATIVE_WORDS = set([
    "hate", "loathe", "despise", "detest", "abhor",
    "ugly", "disgusting", "horrible", "terrible", "awful",
    "stupid", "dumb", "idiot", "moron", "fool",
    "worthless", "useless", "pathetic", "pitiful", "weak",
    "kill", "hurt", "harm", "attack", "destroy",
    "threat", "threaten", "bully", "harass", "abuse",
    "racist", "sexist", "discriminate",
    "loser", "failure", "reject",
])

# ─────────────────────────────────────────────
# 3. GEN Z SLANG DICTIONARY
#    Format: word → ("sentiment", score_modifier, explanation)
#    score_modifier: negative = reduces bullying probability
#                    positive = increases bullying probability
# ─────────────────────────────────────────────
GENZ_SLANG = {
    # High compliments (very positive)
    "slayed": ("positive", -0.45, "Gen Z: performed excellently"),
    "slay": ("positive", -0.40, "Gen Z: to perform amazingly"),
    "slaying": ("positive", -0.40, "Gen Z: performing amazingly"),
    "ate": ("positive", -0.45, "Gen Z: did something perfectly"),
    "ate that": ("positive", -0.50, "Gen Z: executed perfectly"),
    "no crumbs": ("positive", -0.40, "Gen Z: left nothing to improve"),
    "period": ("positive", -0.15, "Gen Z: emphasis/full stop, often agreeing"),
    "periodt": ("positive", -0.15, "Gen Z: strong agreement/emphasis"),
    "bestie": ("positive", -0.35, "Gen Z: close friend, term of endearment"),
    "bff": ("positive", -0.30, "Gen Z: best friend"),
    "fav": ("positive", -0.25, "Gen Z: favorite"),
    "iconic": ("positive", -0.40, "Gen Z: legendary, unforgettable"),
    "legend": ("positive", -0.40, "Gen Z: someone amazing"),
    "goat": ("positive", -0.45, "Gen Z: Greatest Of All Time"),
    "queen": ("positive", -0.40, "Gen Z: strong, powerful person - compliment"),
    "king": ("positive", -0.40, "Gen Z: strong, respected person - compliment"),
    "mother": ("positive", -0.35, "Gen Z: highest compliment, ultimate icon"),
    "understood the assignment": ("positive", -0.50, "Gen Z: performed perfectly"),
    "no cap": ("positive", -0.20, "Gen Z: no lie, seriously"),
    "cap": ("negative", 0.10, "Gen Z: a lie or exaggeration"),
    "lowkey": ("neutral", 0.0, "Gen Z: secretly/somewhat"),
    "highkey": ("neutral", 0.0, "Gen Z: openly/very much"),
    "vibe": ("positive", -0.20, "Gen Z: good feeling/atmosphere"),
    "vibing": ("positive", -0.20, "Gen Z: enjoying, getting along"),
    "valid": ("positive", -0.20, "Gen Z: acceptable, makes sense"),
    "hits different": ("positive", -0.30, "Gen Z: feels uniquely good"),
    "main character": ("positive", -0.25, "Gen Z: someone living their best life"),
    "pop off": ("positive", -0.30, "Gen Z: do your thing, go for it"),
    "bussin": ("positive", -0.40, "Gen Z: really good (usually food)"),
    "bussin bussin": ("positive", -0.50, "Gen Z: extremely good"),
    "fire": ("positive", -0.35, "Gen Z: excellent, great"),
    "lit": ("positive", -0.30, "Gen Z: exciting, excellent"),
    "bet": ("positive", -0.15, "Gen Z: okay/agreed/confirmed"),
    "snatched": ("positive", -0.35, "Gen Z: looking great, on point"),
    "on fleek": ("positive", -0.30, "Gen Z: perfect, on point"),
    "lewk": ("positive", -0.25, "Gen Z: a great look/outfit"),
    "serve": ("positive", -0.35, "Gen Z: delivering an amazing performance/look"),
    "serving": ("positive", -0.35, "Gen Z: delivering excellence"),
    "rent free": ("neutral", 0.05, "Gen Z: living in someone's head"),
    "tea": ("neutral", 0.0, "Gen Z: gossip or truth"),
    "spill": ("neutral", 0.0, "Gen Z: share information"),
    "shade": ("negative", 0.15, "Gen Z: subtle disrespect"),
    "throwing shade": ("negative", 0.20, "Gen Z: subtle disrespect"),
    "receipts": ("neutral", 0.0, "Gen Z: evidence/proof"),
    "pressed": ("negative", 0.15, "Gen Z: bothered/upset"),
    "salty": ("negative", 0.15, "Gen Z: bitter/upset"),
    "clout": ("neutral", 0.0, "Gen Z: social influence"),
    "mid": ("negative", 0.10, "Gen Z: mediocre"),
    "l": ("negative", 0.10, "Gen Z: loss/failure"),
    "ratio": ("negative", 0.20, "Gen Z: being overwhelmingly disagreed with"),
    "caught in 4k": ("neutral", 0.05, "Gen Z: caught doing something clearly"),
    "ok boomer": ("negative", 0.15, "Gen Z: dismissing older perspective"),
    "stan": ("positive", -0.25, "Gen Z: to be a big fan of"),
    "simp": ("neutral", 0.05, "Gen Z: overly devoted, sometimes mocking"),
    "sus": ("negative", 0.10, "Gen Z: suspicious"),
    "sksksk": ("positive", -0.20, "Gen Z: excitement/laughter"),
    "and i oop": ("positive", -0.10, "Gen Z: surprised reaction"),
    "it's giving": ("positive", -0.25, "Gen Z: it's emanating a certain vibe"),
    "based": ("positive", -0.25, "Gen Z: admirable/confident"),
    "w": ("positive", -0.20, "Gen Z: win"),
    "fr fr": ("neutral", 0.0, "Gen Z: for real, seriously"),
    "fr": ("neutral", 0.0, "Gen Z: for real"),
    "ngl": ("neutral", 0.0, "Gen Z: not gonna lie"),
    "imo": ("neutral", 0.0, "in my opinion"),
    "idk": ("neutral", 0.0, "I don't know"),
    "lol": ("positive", -0.10, "laughing, lighthearted"),
    "lmao": ("positive", -0.10, "laughing hard, lighthearted"),
    "omg": ("neutral", 0.0, "oh my god - exclamation"),
    "tbh": ("neutral", 0.0, "to be honest"),
    "flex": ("positive", -0.10, "Gen Z: showing off in an admired way"),
    "clap back": ("neutral", 0.05, "Gen Z: witty response to criticism"),
    "glow up": ("positive", -0.35, "Gen Z: impressive positive transformation"),
    "that's so real": ("positive", -0.20, "Gen Z: genuine agreement"),
    "brat": ("positive", -0.15, "Gen Z 2024: confident, carefree attitude"),
    "delulu": ("neutral", 0.05, "Gen Z: delusional, used playfully"),
    "rizz": ("positive", -0.25, "Gen Z: natural charm/charisma"),
    "unalived": ("negative", 0.30, "Gen Z euphemism for death/suicide - serious"),
    "touch grass": ("neutral", 0.10, "Gen Z: go outside, you're too online"),
    # Ratings and approval expressions
    "no notes": ("positive", -0.50, "Gen Z: this is perfect, zero criticism"),
    "chefs kiss": ("positive", -0.45, "Gen Z: absolutely perfect"),
    "chef's kiss": ("positive", -0.45, "Gen Z: absolutely perfect"),
    "immaculate": ("positive", -0.40, "Gen Z: flawless, perfect"),
    "perfection": ("positive", -0.40, "Expressing something is perfect"),
    "flawless": ("positive", -0.35, "Without any flaw — very positive"),
    "nailed it": ("positive", -0.40, "Did something perfectly"),
    "on point": ("positive", -0.30, "Exactly right, perfect"),
    "spot on": ("positive", -0.30, "Exactly correct"),
    "well done": ("positive", -0.30, "Praise for doing something well"),
    "good job": ("positive", -0.35, "Direct praise"),
    "great job": ("positive", -0.40, "Strong praise"),
    # Reclaimed/colloquial terms used as compliments in Gen Z
    "bitch": ("neutral", -0.05, "Gen Z: reclaimed term, often used as endearment"),
    "bitches": ("neutral", -0.05, "Gen Z: reclaimed term, often used as endearment"),
    "baddie": ("positive", -0.30, "Gen Z: confident, attractive person — high compliment"),
    "that girl": ("positive", -0.25, "Gen Z: someone who has their life together — compliment"),
    "girlboss": ("positive", -0.25, "Gen Z: ambitious, successful woman — compliment"),
    "it girl": ("positive", -0.30, "Gen Z: trendsetter, style icon — compliment"),
    "sickening": ("positive", -0.35, "LGBTQ+/Gen Z: so good it's shocking"),
    "killing it": ("positive", -0.35, "Gen Z: performing exceptionally well"),
    "killed it": ("positive", -0.35, "Gen Z: performed exceptionally well"),
    "drip": ("positive", -0.30, "Gen Z: stylish outfit or accessories"),
    "swag": ("positive", -0.25, "Gen Z: confident style and attitude"),
    "fam": ("positive", -0.20, "Gen Z: friend, like family — term of endearment"),
    "bae": ("positive", -0.25, "Gen Z: significant other or close person"),
    "goals": ("positive", -0.25, "Gen Z: something aspirational or admirable"),
    "oof": ("neutral", 0.05, "Gen Z: empathetic reaction to something painful"),
    "yikes": ("neutral", 0.05, "Gen Z: expression of concern or awkwardness"),
    "bruh": ("neutral", 0.02, "Gen Z: expression of disbelief; can be friendly"),
    "periodt": ("positive", -0.15, "Gen Z: strong emphasis, often agreeing"),
}

# ─────────────────────────────────────────────
# 4. CONTEXT MODIFIERS
# ─────────────────────────────────────────────
NEGATIONS = {"not", "no", "never", "dont", "don't", "doesn't", "doesnt",
             "wasn't", "wasnt", "isn't", "isnt", "can't", "cant", "won't", "wont"}

INTENSIFIERS = {"very", "really", "so", "extremely", "absolutely", "totally",
                "completely", "such", "super", "mega", "ultra"}

DIMINISHERS = {"kinda", "sorta", "kind of", "sort of", "slightly", "a bit", "little"}


def _tokenize(text: str):
    """Simple whitespace + punctuation tokenizer."""
    return re.findall(r"\b\w+\b", text.lower())


def _check_phrases(text_lower: str) -> list:
    """Check for multi-word slang phrases."""
    found = []
    for phrase, data in GENZ_SLANG.items():
        if " " in phrase and phrase in text_lower:
            found.append((phrase, data))
    return found


def analyze_sentiment(text: str) -> Dict:
    """
    Full sentiment analysis returning a structured result.

    Returns:
        {
          positive_score: float 0-1,
          negative_score: float 0-1,
          net_sentiment: float (-1 to +1),
          genz_signals: list of detected slang with labels,
          has_strong_positive: bool,
          has_threats: bool,
          explanation: str
        }
    """
    text_lower = text.lower()
    # ── Detect numeric ratings BEFORE cleaning ──────────
    import re as _re
    rating_bonus = 0.0
    # Matches: 10/10, 9/10, 5/5, 10 out of 10, 5 stars, 5 star
    rating_match = _re.search(r'(\d+)\s*/\s*(\d+)', text_lower)
    if rating_match:
        score = int(rating_match.group(1))
        total = int(rating_match.group(2))
        if total > 0:
            ratio = score / total
            if ratio >= 0.8:      # 8/10 and above = positive
                rating_bonus = -0.35
            elif ratio >= 0.6:    # 6/10 = mildly positive
                rating_bonus = -0.15
            elif ratio <= 0.3:    # 3/10 and below = negative
                rating_bonus = 0.20
    # Matches: "5 stars", "5 star"
    stars_match = _re.search(r'(\d)\s*stars?', text_lower)
    if stars_match:
        stars = int(stars_match.group(1))
        if stars >= 4:
            rating_bonus = -0.35
        elif stars <= 2:
            rating_bonus = 0.20
    genz_modifier = rating_bonus  # start modifier with rating bonus
    # ────────────────────────────────────────────────────
    tokens = _tokenize(text_lower)

    pos_count = 0
    neg_count = 0
    genz_signals = []
    threat_words_found = []

    # Check multi-word phrases first
    phrase_hits = _check_phrases(text_lower)
    for phrase, (sentiment, modifier, explanation) in phrase_hits:
        genz_modifier += modifier
        genz_signals.append({
            "term": phrase,
            "sentiment": sentiment,
            "explanation": explanation
        })

    # Token-level analysis with negation window
    negation_window = 0
    for i, token in enumerate(tokens):
        if token in NEGATIONS:
            negation_window = 3  # next 3 words affected
            continue

        is_negated = negation_window > 0
        negation_window = max(0, negation_window - 1)

        # Check Gen Z single-word slang
        if token in GENZ_SLANG:
            sentiment, modifier, explanation = GENZ_SLANG[token]
            if is_negated:
                modifier = -modifier  # flip sentiment under negation
            genz_modifier += modifier
            genz_signals.append({
                "term": token,
                "sentiment": "negated-" + sentiment if is_negated else sentiment,
                "explanation": explanation
            })

        # Positive lexicon
        elif token in POSITIVE_WORDS:
            if is_negated:
                neg_count += 1
            else:
                pos_count += 1

        # Negative lexicon
        elif token in NEGATIVE_WORDS:
            if is_negated:
                pos_count += 0.5  # "not hate" is weakly positive
            else:
                neg_count += 1
                threat_words_found.append(token)

    total = max(len(tokens), 1)
    pos_score = min(pos_count / total, 1.0)
    neg_score = min(neg_count / total, 1.0)
    net_sentiment = (pos_score - neg_score) + (genz_modifier * 0.3)
    net_sentiment = max(-1.0, min(1.0, net_sentiment))

    # Build explanation
    signals = []
    if pos_count > 0:
        signals.append(f"{pos_count} positive word(s) detected")
    if neg_count > 0:
        signals.append(f"{neg_count} negative/threat word(s) detected")
    if genz_signals:
        labels = [f"'{s['term']}' ({s['explanation']})" for s in genz_signals[:3]]
        signals.append("Gen Z slang: " + ", ".join(labels))

    explanation = "; ".join(signals) if signals else "No strong sentiment signals detected"

    return {
        "positive_score": round(pos_score, 3),
        "negative_score": round(neg_score, 3),
        "net_sentiment": round(net_sentiment, 3),
        "genz_signals": genz_signals,
        "has_strong_positive": net_sentiment > 0.3 or pos_count >= 2,
        "has_threats": bool(threat_words_found),
        "threat_words": threat_words_found,
        "explanation": explanation
    }


def ensemble_score(model_prob: float, sentiment: Dict, text: str) -> Tuple[float, str]:
    """
    Combine the H5 model probability with sentiment analysis.

    Strategy:
    - If sentiment is strongly positive → pull score down (less bullying)
    - If Gen Z slang is detected as positive → apply strong correction
    - If threats are detected → keep or raise score
    - Model gets 50% weight, sentiment gets 50%

    Returns: (final_probability, reasoning)
    """
    net = sentiment["net_sentiment"]
    has_strong_pos = sentiment["has_strong_positive"]
    has_threats = sentiment["has_threats"]

    # Convert net sentiment to a "non-bullying" probability adjustment
    # net > 0 → positive → reduces bullying prob
    # net < 0 → negative → increases bullying prob
    sentiment_adjustment = -net * 0.4  # scale factor

    # Strong positive signals from Gen Z slang = heavy correction
    genz_positive = sum(1 for s in sentiment["genz_signals"] if "positive" in s["sentiment"])
    genz_negative = sum(1 for s in sentiment["genz_signals"] if "negative" in s["sentiment"])

    if genz_positive > 0 and not has_threats:
        sentiment_adjustment -= 0.20 * genz_positive  # each positive slang reduces by 20%

    if genz_negative > 0:
        sentiment_adjustment += 0.10 * genz_negative

    # When genz positive signals strongly outnumber negative → high confidence correction
    genz_dominance = genz_positive - genz_negative
    if genz_dominance >= 2 and not has_threats:
        sentiment_adjustment -= 0.20  # extra correction when clearly positive context

    # Ensemble: dynamic weighting based on signal confidence
    # More positive signals → sentiment gets higher weight
    if has_strong_pos and not has_threats:
        model_weight = 0.30
        sentiment_weight = 0.70
    elif genz_positive >= 1 and not has_threats:
        model_weight = 0.35
        sentiment_weight = 0.65
    else:
        model_weight = 0.55
        sentiment_weight = 0.45

    final = (model_prob * model_weight) + (model_prob + sentiment_adjustment) * sentiment_weight
    final = max(0.01, min(0.99, final))

    # Build reasoning string
    reasons = []
    if genz_positive > 0:
        terms = [s["term"] for s in sentiment["genz_signals"] if "positive" in s["sentiment"]]
        reasons.append(f"Gen Z compliments detected: {', '.join(terms)}")
    if has_strong_pos:
        reasons.append("Strong positive sentiment overrides model uncertainty")
    if has_threats:
        reasons.append(f"Threat language detected: {', '.join(sentiment['threat_words'])}")
    if not reasons:
        reasons.append("Standard ensemble of model + sentiment analysis")

    return round(final, 3), " | ".join(reasons)


def get_severity_level(probability: float) -> Tuple[str, str, str]:
    """Returns (label, hex_color, emoji)"""
    if probability < 0.25:
        return "Safe Content", "#22c55e", "✓"
    elif probability < 0.45:
        return "Low Risk", "#84cc16", "◐"
    elif probability < 0.65:
        return "Moderate Risk", "#f59e0b", "⚠"
    elif probability < 0.80:
        return "High Risk", "#f97316", "▲"
    else:
        return "Severe Risk", "#ef4444", "✕"
