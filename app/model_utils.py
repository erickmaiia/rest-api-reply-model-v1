import time
import joblib
import os

# Caminho para a pasta ml_models
ML_MODELS_PATH = os.path.join(os.path.dirname(__file__), "ml_models")

# Atualizar os caminhos dos modelos para apontar para a pasta ml_models
MODEL_PATHS = {
    "Naive Bayes": os.path.join(ML_MODELS_PATH, "Naive_Bayes_model.joblib"),
    "SVM": os.path.join(ML_MODELS_PATH, "SVM_model.joblib"),
    "XGBoost": os.path.join(ML_MODELS_PATH, "XGBoost_model.joblib"),
    "LightGBM": os.path.join(ML_MODELS_PATH, "LightGBM_model.joblib"),
    "Multilayer Perceptron": os.path.join(ML_MODELS_PATH, "Multilayer_Perceptron_model.joblib"),
    "Gradient Boosting": os.path.join(ML_MODELS_PATH, "Gradient_Boosting_model.joblib"),
    "Random Forest": os.path.join(ML_MODELS_PATH, "Random_Forest_model.joblib"),
    "AdaBoost": os.path.join(ML_MODELS_PATH, "AdaBoost_model.joblib"),
    "Decision Tree": os.path.join(ML_MODELS_PATH, "Decision_Tree_model.joblib"),
}

# Caminho do vetorizador
VECTORIZER_PATH = os.path.join(ML_MODELS_PATH, "tfidf_vectorizer.pkl")

def assign_sentiment_alias(prediction):
    """Atribui um rótulo de sentimento baseado na previsão numérica."""
    sentiment_mapping = {0: "negative", 1: "neutral", 2: "positive"}
    return sentiment_mapping.get(prediction, "unknown")

def load_text_model(model_choice):
    """
    Carrega o modelo e o vetorizador baseado no modelo escolhido pelo usuário.
    """
    if model_choice in MODEL_PATHS:
        model_path = MODEL_PATHS[model_choice]
        model = joblib.load(model_path)
        vectorizer = joblib.load(VECTORIZER_PATH)
        return model, vectorizer
    
    elif model_choice == "All":
        models = {}
        for model_name, model_path in MODEL_PATHS.items():
            models[model_name] = joblib.load(model_path)
        vectorizer = joblib.load(VECTORIZER_PATH)
        return models, vectorizer
    else:
        raise ValueError("Modelo inválido.")

def predict_text(model, vectorizer, text):
    """
    Função para fazer uma previsão de sentimento baseada no texto e no modelo escolhido.
    """
    text_vector = vectorizer.transform([text])  # Transforma o texto em uma representação numérica
    prediction = model.predict(text_vector)[0]
    sentiment = assign_sentiment_alias(int(prediction))
    return sentiment

def predict_text_all_models(models, vectorizer, text):
    """
    Função para fazer uma previsão de sentimento baseada no texto e em todos os modelos disponíveis,
    incluindo estatísticas como tempo de inferência, confiança na previsão e distribuição das previsões.
    A previsão final é ordenada com base na confiança.
    """
    predictions = {}
    sentiment_distribution = {"positive": 0, "neutral": 0, "negative": 0}
    
    text_vector = vectorizer.transform([text])  # Transforma o texto em uma representação numérica

    for model_name, model in models.items():
        # Medir tempo de inferência
        start_time = time.time()
        prediction = model.predict(text_vector)[0]
        inference_time = time.time() - start_time

        # Confiança na previsão (se disponível)
        try:
            confidence = max(model.predict_proba(text_vector)[0]) * 100  # Em porcentagem
            confidence = float(confidence)  # Converte para float nativo, se necessário
        except AttributeError:
            confidence = None  # Modelo não suporta probabilidade

        # Converte a previsão em sentimento
        sentiment = assign_sentiment_alias(int(prediction))

        # Atualiza a distribuição de sentimentos
        if sentiment in sentiment_distribution:
            sentiment_distribution[sentiment] += 1

        # Adiciona as estatísticas e informações
        predictions[model_name] = {
            "sentiment": sentiment,
            "inference_time_ms": round(inference_time * 1000, 2),
            "confidence": confidence,
        }

    # Ordena as previsões por confiança (do maior para o menor)
    sorted_predictions = dict(sorted(predictions.items(), key=lambda item: item[1]["confidence"] if item[1]["confidence"] is not None else 0, reverse=True))

    # Adiciona a distribuição de sentimentos ao dicionário de predições
    sorted_predictions["sentiment_distribution"] = sentiment_distribution

    return sorted_predictions
