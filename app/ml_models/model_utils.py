import time
from collections import defaultdict
import joblib
import os

# Definir os caminhos dos modelos
import os

MODEL_PATHS = {
    "Naive Bayes": os.path.join(os.path.dirname(__file__), "Naive_Bayes_model.joblib"),
    "SVM": os.path.join(os.path.dirname(__file__), "SVM_model.joblib"),
    "XGBoost": os.path.join(os.path.dirname(__file__), "XGBoost_model.joblib"),
    # "LightGBM": os.path.join(os.path.dirname(__file__), "LightGBM_model.joblib"),
    # "Multilayer Perceptron": os.path.join(os.path.dirname(__file__), "Multilayer_Perceptron_model.joblib"),
    # "Gradient Boosting": os.path.join(os.path.dirname(__file__), "Gradient_Boosting_model.joblib"),
    # "Random Forest": os.path.join(os.path.dirname(__file__), "Random_Forest_model.joblib"),
    # "AdaBoost": os.path.join(os.path.dirname(__file__), "AdaBoost_model.joblib"),
    # "Decision Tree": os.path.join(os.path.dirname(__file__), "Decision_Tree_model.joblib"),
}


# Definir o caminho do vetorizador
VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), "tfidf_vectorizer.pkl")

def assign_sentiment_alias(prediction):
    if prediction == 0:
        return "negative"
    elif prediction == 1:
        return "neutral"
    elif prediction == 2:
        return "positive"
    else:
        return "unknown"

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
            model = joblib.load(model_path)
            models[model_name] = model
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
    incluindo estatísticas como tempo de inferência e confiança na previsão,
    além da distribuição das previsões.
    """
    predictions = {}
    sentiment_distribution = defaultdict(int)  # Contador para a distribuição das previsões
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

        # Adiciona as estatísticas e informações
        predictions[model_name] = {
            "sentiment": sentiment,
            "inference_time_ms": round(inference_time * 1000, 2),
            "confidence": confidence,
        }

        # Incrementa a contagem da distribuição das previsões
        sentiment_distribution[sentiment] += 1

    # Adiciona a distribuição das previsões ao dicionário de resultados
    predictions["sentiment_distribution"] = dict(sentiment_distribution)

    return predictions



