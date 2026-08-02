# CyberSafe — Cyberbullying Detection (Enhanced)

## What's New vs. Original

| Problem | Old App | New App |
|---|---|---|
| "awesome" flagged as bullying | ❌ Only knew training data | ✅ Sentiment lexicon handles unknown words |
| "you slayed bestie" = bullying | ❌ Saw "slayed" as violence | ✅ Gen Z slang dictionary, context-aware |
| No way to teach new words | ❌ Retrain entire model | ✅ Add words live via UI |
| Single model signal | ❌ Only H5 model | ✅ Ensemble: H5 + Sentiment + Custom Words |

---

## Architecture

```
User Input
    │
    ├── [1] H5 Keras Model (your trained model, 55% weight)
    │       └── Handles patterns it learned from Kaggle data
    │
    ├── [2] Sentiment Engine (45% weight)
    │       ├── Positive word lexicon (500+ words)
    │       ├── Negative/threat word lexicon
    │       ├── Gen Z slang dictionary (50+ terms)
    │       ├── Negation detection ("not hate" ≠ hate)
    │       └── Intensity modifiers
    │
    └── [3] Custom Word Layer (additive correction)
            └── Words you add via the UI → applied immediately
                Stored in custom_words.json
```

**Ensemble formula:**
```
final_score = (model_prob × 0.55) + (adjusted_prob × 0.45)
adjusted_prob = model_prob + sentiment_adjustment + genz_corrections
```

---

## Project Structure

```
cyberbullying_app/
├── app.py                  # Flask routes + ensemble logic
├── sentiment_engine.py     # Sentiment analysis + Gen Z slang
├── requirements.txt
├── cyberbullying_model.h5  # Your trained model (copy here)
├── tokenizer.pkl           # Your tokenizer (copy here)
├── user_feedback.json      # Auto-created: stores user corrections
├── custom_words.json       # Auto-created: stores trained words
└── templates/
    └── index.html          # Premium UI
```

---

## Setup & Run (Local)

```bash
# 1. Install dependencies
pip install flask tensorflow numpy

# 2. Copy your model files into the project folder
cp /path/to/cyberbullying_model.h5 ./
cp /path/to/tokenizer.pkl ./

# 3. Run
python app.py

# 4. Open browser
# http://localhost:5000
```

---

## Deploy to Production

### Option A: Render.com (Free tier, recommended)
1. Push project to GitHub
2. Go to render.com → New Web Service
3. Connect your repo
4. Build command: `pip install -r requirements.txt`
5. Start command: `gunicorn app:app`
6. Add `gunicorn` to requirements.txt

### Option B: Railway.app
1. Push to GitHub
2. New project → Deploy from GitHub
3. Railway auto-detects Flask
4. Set start command: `python app.py`

### Option C: Heroku
```bash
# Add Procfile:
echo "web: gunicorn app:app" > Procfile
heroku create your-app-name
git push heroku main
```

### Option D: Local with ngrok (quick demo)
```bash
pip install ngrok
python app.py &
ngrok http 5000
```

---

## User Training Feature

Users can teach the model new words directly from the UI:

- **Positive words**: words that should reduce bullying probability
  - Examples: "awesome", "fire", "bussin", custom compliments
- **Negative words**: words that should increase bullying probability
  - Examples: domain-specific slurs, threats your model missed

All custom words are stored in `custom_words.json` and applied immediately — no retraining needed.

---

## Gen Z Slang Coverage

The system automatically understands these as positive:
- slay/slayed, ate, no crumbs, bestie, iconic, GOAT
- queen, king, mother, bussin, fire, lit, snatched
- serve/serving, glow up, rizz, based, pop off, hits different
- W, understood the assignment, no cap, vibe, vibing, bet
- main character, stan, brat, it's giving, sksksk

And these as negative/concerning:
- throwing shade, ratio, L, mid, cap, pressed, salty
- ok boomer, sus, unalived

---

## How the Sentiment Engine Fixes "awesome"

1. "awesome" is in the positive word lexicon
2. Sentiment engine detects 1 positive word
3. `net_sentiment = +0.15` (positive)
4. `sentiment_adjustment = -0.06` (reduces bullying probability)
5. Even if model gave 0.6 (bullying), final = ~0.50 → borderline
6. With 2+ positive words, clearly pulled below 0.5 → Safe

---

## How Gen Z Context Works: "you slayed and ate, bitch"

1. Model may see "bitch" → flags as bullying (0.75)
2. Sentiment engine:
   - "slayed" → Gen Z positive, modifier = -0.45
   - "ate" → Gen Z positive, modifier = -0.45
   - "bitch" → negative word, +0.1
   - Net: strongly positive
3. `genz_positive_count = 2` → additional -0.30 correction
4. `final_prob ≈ 0.28` → Safe Content ✅
5. UI shows: "Gen Z compliments detected: slayed, ate"

---

## API Endpoints

```
POST /api/predict
  Body: { "text": "..." }
  Returns: { prediction, probability, severity, sentiment, reasoning, genz_detected }

POST /api/add-word
  Body: { "word": "awesome", "label": "positive" }
  Returns: { status, message }

POST /api/feedback
  Body: { "text": "...", "correct_label": "safe"|"bullying" }
  Returns: { status, message }

GET /api/custom-words
  Returns: { positive: [...], negative: [...] }

GET /api/stats
  Returns: { total_feedback, custom_positive_words, custom_negative_words }
```
