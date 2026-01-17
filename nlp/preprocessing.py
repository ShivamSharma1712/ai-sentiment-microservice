NEGATIONS = {"not", "no", "never", "hardly", "barely"}
POSITIVE = {"good", "great", "excellent", "amazing", "nice"}
NEGATIVE = {"bad", "worst", "awful", "terrible"}

def detect_flags(text: str):
    tokens = text.lower().split()

    negation = False
    mixed = False
    sarcasm = False

    for i in range(len(tokens) - 1):
        if tokens[i] in NEGATIONS and tokens[i + 1] in POSITIVE:
            negation = True
        if tokens[i] in NEGATIONS and tokens[i + 1] in NEGATIVE:
            negation = True

    if any(w in text.lower() for w in POSITIVE) and any(w in text.lower() for w in NEGATIVE):
        mixed = True

    if "yeah right" in text.lower() or "sure" in text.lower():
        sarcasm = True

    return {
        "negation": negation,
        "mixed_sentiment": mixed,
        "sarcasm_possible": sarcasm
    }
