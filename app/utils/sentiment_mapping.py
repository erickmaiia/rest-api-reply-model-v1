SENTIMENT_MAPPING = {
    0: "negative",
    1: "neutral",
    2: "positive"
}

def assign_sentiment_alias(prediction: int) -> str:
    """
    Atribui um rótulo de sentimento baseado na previsão numérica.
    """
    return SENTIMENT_MAPPING.get(prediction, "unknown")
