"""
Feature 5: Hybrid Spam/Ham Classifier (ML + Keyword Boost)

Combines the fine-tuned MuRIL model with keyword-based spam detection.
If the text contains known spam/scam keywords (OTP, claim, verify, etc.),
the spam probability is boosted — fixing cases where the ML model alone
misses phone-call-style spam patterns.

Usage (standalone test):
    python onnx_classifier.py
"""

import os
import re
import sys

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from config import (
    FINE_TUNED_MODEL_DIR,
    TOKENIZER_NAME,
    CLASSIFIER_MAX_LENGTH,
    CLASSIFIER_LABELS,
    SPAM_KEYWORDS,
)

# ─── Keyword-based spam patterns (phone call specific) ───
# These are weighted: higher weight = stronger spam signal
SPAM_KEYWORD_WEIGHTS = {
    # High-confidence spam signals (scam call patterns)
    "otp": 0.4,
    "claim": 0.3,
    "verify": 0.3,
    "verify your": 0.4,
    "account number": 0.4,
    "bank account": 0.4,
    "credit card": 0.4,
    "debit card": 0.4,
    "aadhaar": 0.3,
    "pan card": 0.3,
    "pan number": 0.3,
    "process": 0.2,
    "whatsapp": 0.15,
    "send address": 0.3,
    "enter the": 0.2,

    # Medium-confidence spam signals
    "congratulations": 0.3,
    "selected": 0.25,
    "winner": 0.3,
    "won a": 0.3,
    "prize": 0.3,
    "free": 0.15,
    "offer": 0.2,
    "loan": 0.25,
    "insurance": 0.25,
    "scheme": 0.2,
    "interest rate": 0.25,
    "limited time": 0.25,
    "urgent": 0.2,
    "immediately": 0.2,
    "compromised": 0.3,
    "blocked": 0.2,
    "suspended": 0.3,

    # Tamil/Tanglish spam signals
    "claim pannunga": 0.4,
    "claim maani": 0.4,
    "kedaikum": 0.2,
    "ungalukku": 0.15,
    "thittam": 0.2,
    "jeyicheenga": 0.3,
    "panam": 0.15,
    "vangalam": 0.2,
}

# Minimum keyword boost to flip from Ham to Spam
KEYWORD_BOOST_THRESHOLD = 0.2


class ONNXClassifier:
    """
    Hybrid classifier: MuRIL model + keyword-based spam boost.

    1. Runs MuRIL for base classification (Spam/Ham probabilities)
    2. Scans text for spam keywords and computes a boost score
    3. If keywords are detected, boosts the spam probability
    4. Final decision uses the combined score

    Name kept as ONNXClassifier for backward compatibility.
    """

    def __init__(
        self,
        model_path: str = None,
        tokenizer_path: str = None,
        max_length: int = CLASSIFIER_MAX_LENGTH,
        labels: list = None,
    ):
        if model_path is None:
            model_path = FINE_TUNED_MODEL_DIR
        self.model_path = model_path

        if tokenizer_path is None:
            if os.path.exists(FINE_TUNED_MODEL_DIR):
                tokenizer_path = FINE_TUNED_MODEL_DIR
            else:
                tokenizer_path = TOKENIZER_NAME
        self.tokenizer_path = tokenizer_path

        self.max_length = max_length
        self.labels = labels or CLASSIFIER_LABELS
        self.model = None
        self.tokenizer = None
        self._loaded = False

    def load(self):
        """Load the fine-tuned model and tokenizer."""
        if self._loaded:
            return

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model not found at '{self.model_path}'. "
                f"Run fine_tune.py first."
            )

        print(f"[Classifier] Loading model: {self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path
        )
        self.model.eval()
        self._loaded = True
        print("[Classifier] Ready (PyTorch + Keyword Boost).")

    @staticmethod
    def _softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    @staticmethod
    def _keyword_scan(text: str) -> tuple:
        """
        Scan text for spam keywords and return boost score + matched keywords.

        Returns
        -------
        tuple[float, list[str]]
            (boost_score, list_of_matched_keywords)
        """
        text_lower = text.lower()
        boost = 0.0
        matched = []

        for keyword, weight in SPAM_KEYWORD_WEIGHTS.items():
            # Count occurrences (more occurrences = stronger signal)
            count = text_lower.count(keyword)
            if count > 0:
                # Diminishing returns: first match = full weight, extras = half
                boost += weight + (count - 1) * weight * 0.2
                matched.append(f"{keyword}(×{count})" if count > 1 else keyword)

        # Cap at 0.95 to avoid 100% certainty from keywords alone
        boost = min(boost, 0.95)
        return boost, matched

    @staticmethod
    def _extract_spam_sentences(text: str, matched_keywords: list) -> str:
        """Find the specific sentence(s) containing the spam keywords."""
        import re
        # Split text into sentences using basic punctuation
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        spam_sentences = []
        
        if matched_keywords:
            # Strip frequency counts like '(×2)' from keywords
            raw_keywords = [k.split('(')[0] for k in matched_keywords]
            for s in sentences:
                s_lower = s.lower()
                if any(k in s_lower for k in raw_keywords):
                    spam_sentences.append(s)
        
        if spam_sentences:
            return " ".join(spam_sentences)
            
        # Fallback: if ML model flagged it without keywords, just return the first sentence
        return sentences[0] if sentences else text

    def classify(self, text: str) -> dict:
        """
        Classify text as Spam or Ham using hybrid approach.

        1. MuRIL model gives base probabilities
        2. Keyword scan gives boost score
        3. Combined: spam_prob = max(ml_spam, keyword_boost)

        Returns
        -------
        dict
            {"label", "confidence", "probabilities", "keyword_boost", "keywords_found", "spam_sentence"}
        """
        if not self._loaded:
            self.load()

        # Step 1: ML classification
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0].numpy()

        ml_probs = self._softmax(logits)
        ml_spam_prob = float(ml_probs[1])  # index 1 = Spam

        # Step 2: Keyword boost
        keyword_boost, keywords_found = self._keyword_scan(text)

        # Step 3: Combine — take the higher signal
        final_spam_prob = max(ml_spam_prob, keyword_boost)

        # If keyword boost is significant but ML says Ham, override
        if keyword_boost >= KEYWORD_BOOST_THRESHOLD and ml_spam_prob < 0.5:
            final_spam_prob = keyword_boost
            method = "keyword"
        elif ml_spam_prob >= 0.5:
            final_spam_prob = ml_spam_prob
            method = "model"
        else:
            final_spam_prob = max(ml_spam_prob, keyword_boost)
            method = "combined"

        # Final label
        if final_spam_prob >= 0.5:
            label = "Spam"
            confidence = final_spam_prob
        else:
            label = "Ham"
            confidence = 1.0 - final_spam_prob

        result = {
            "label": label,
            "confidence": confidence,
            "probabilities": {"Ham": 1.0 - final_spam_prob, "Spam": final_spam_prob},
            "method": method,
        }

        if keywords_found:
            result["keywords_found"] = keywords_found
            result["keyword_boost"] = keyword_boost
            
        if label == "Spam":
            result["spam_sentence"] = self._extract_spam_sentences(text, keywords_found)

        return result


# ──────────────────────────── standalone test
def _test():
    """Quick test with sample texts."""
    classifier = ONNXClassifier()

    try:
        classifier.load()
    except FileNotFoundError as e:
        print(f"[!] {e}")
        sys.exit(1)

    test_texts = [
        "Congratulations! You've won a free prize. Call now to claim!",
        "Hey, are you coming to the meeting tomorrow?",
        "Sir, bank loan offer irukku, 2% interest la kedaikum.",
        "Naan late varuvena, 15 minutes la varuven.",
        "Sir our number is OTP sir, OTP number, so we can enter the OTP and enter the address.",
        "Bro, cricket match paakka pogalama?",
        "Sir claim pannunga sir, ungalukku special offer irukku.",
        "Your bank account has been compromised. Verify immediately!",
    ]

    print("=" * 70)
    print("  Hybrid Classifier — Test  (MuRIL + Keyword Boost)")
    print("=" * 70)

    for text in test_texts:
        result = classifier.classify(text)
        keywords = result.get("keywords_found", [])
        boost = result.get("keyword_boost", 0)
        kw_str = f" | Keywords: {', '.join(keywords)} (+{boost:.0%})" if keywords else ""
        print(f"\n  [{result['label']} | {result['confidence']:.0%} | via {result['method']}]{kw_str}")
        print(f'    "{text[:75]}"')


if __name__ == "__main__":
    _test()
