"""Vitality — Spam & Phishing detector.

predict(text) returns the verdict, the probability, the words that pushed the
message towards spam, and rule-based red flags (urgency, authority, manipulation).
"""
import re
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "spam_model.joblib"
THRESHOLD = 0.5  # probability above which a message is labelled spam/phishing

# Red-flag phrases (English + common Swahili/Sheng). Matching is case-insensitive.
RED_FLAGS = {
    "Urgency": [
        r"\burgent(ly)?\b", r"\bimmediately\b", r"\bact now\b", r"\bright now\b", r"\blast chance\b",
        r"\bfinal (warning|notice|reminder)\b", r"\bwithin \d+\s*(minutes?|mins?|hours?|hrs?|days?)\b",
        r"\bwithin (one|an|a|two|24|48) (hour|day)s?\b", r"\bexpires? (today|tonight|soon|in)\b",
        r"\b(today|tonight) only\b", r"\bbefore (it'?s too late|offer expires)\b", r"\bsuspend(ed)?\b",
        r"\blocked\b", r"\bsuspension\b", r"\bdeadline\b", r"\bharaka\b", r"\bsasa hivi\b", r"\bleo tu\b",
    ],
    "Authority": [
        r"\bgovernment\b", r"\bcourt\b", r"\bpolice\b", r"\bkra\b", r"\bofficial\b", r"\bhr\b",
        r"\bhuman resources\b", r"\bceo\b", r"\bmanagement\b", r"\bexecutive\b", r"\bit (security|department|support)\b",
        r"\bsystem administrator\b", r"\bbank\b", r"\bsafaricom\b", r"\bm-?pesa\b", r"\bapple\b",
        r"\bmicrosoft\b", r"\bsecurity (alert|notice|team)\b", r"\bserikali\b",
    ],
    "Manipulation": [
        r"\bcongratulations\b", r"\byou('ve| have)? (won|been selected|been chosen|qualif(y|ied))\b",
        r"\bwinner\b", r"\b(prize|reward|bonus|gift card|jackpot)\b", r"\bfree\b", r"\bclaim\b",
        r"\bclick (here|the link|now)\b", r"\bverify your (identity|account|details)\b",
        r"\b(send|share|enter|confirm) (your )?(pin|password|otp|id|passport|card details)\b",
        r"\bprocessing fee\b", r"\b(instant|quick) loan\b", r"\bavoid (arrest|legal action|suspension)\b",
        r"\bface arrest\b", r"\bdial \*\d", r"\bumeshinda\b", r"\btuma pin\b",
    ],
}
_COMPILED = {k: [re.compile(p, re.I) for p in v] for k, v in RED_FLAGS.items()}


@lru_cache(maxsize=1)
def _load():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Spam model not found. Train it first: python -m training.train_spam")
    return joblib.load(MODEL_PATH)


def red_flags(text: str) -> dict:
    """Return {flag_name: [matched phrases]} for every red flag found."""
    found = {}
    for name, patterns in _COMPILED.items():
        hits = sorted({m.group(0).strip() for p in patterns for m in p.finditer(text)}, key=str.lower)
        if hits:
            found[name] = hits
    return found


def _top_words(pipeline, text: str, k: int = 6) -> list:
    """Words/phrases in this message that pushed the score most towards spam."""
    word_vec = pipeline.named_steps["features"].transformer_list[0][1]
    clf = pipeline.named_steps["clf"]
    x = word_vec.transform([text])
    n_word = len(word_vec.vocabulary_)
    contrib = x.multiply(clf.coef_[0][:n_word]).toarray()[0]
    names = word_vec.get_feature_names_out()
    order = np.argsort(contrib)[::-1]
    out = []
    for i in order:
        if contrib[i] <= 0 or len(out) == k:
            break
        if all(w in ENGLISH_STOP_WORDS for w in names[i].split()):
            continue  # skip filler words such as "your" or "to"
        out.append(names[i])
    return out


def predict(text: str) -> dict:
    text = (text or "").strip()
    if not text:
        raise ValueError("Please enter a message to check.")
    bundle = _load()
    pipe = bundle["pipeline"]
    p_spam = float(pipe.predict_proba([text])[0][1])
    is_spam = p_spam >= THRESHOLD
    return {
        "verdict": "Spam / Phishing" if is_spam else "Looks safe",
        "spam_probability": round(p_spam, 3),
        "confidence": round(p_spam if is_spam else 1 - p_spam, 3),
        "top_words": _top_words(pipe, text) if is_spam else [],
        "red_flags": red_flags(text),
    }


if __name__ == "__main__":
    for msg in [
        "URGENT: Your M-Pesa account is locked. Send your PIN within 10 minutes to avoid suspension.",
        "Hi, the team meeting has moved to 3pm tomorrow in the main hall.",
    ]:
        print(msg, "\n ->", predict(msg), "\n")
