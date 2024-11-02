from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

# Definir os caminhos dos modelos
MODEL_PATHS = {
    "Naive Bayes": os.path.join(os.path.dirname(__file__), "Naive_Bayes_model.joblib"),
    "SVM": os.path.join(os.path.dirname(__file__), "SVM_model.joblib"),
    "XGBoost": os.path.join(os.path.dirname(__file__), "XGBoost_model.joblib")
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
    else:
        raise ValueError("Modelo inválido. Escolha entre 'Naive Bayes', 'SVM' ou 'XGBoost'.")

def predict_text(model, vectorizer, text):
    """
    Função para fazer uma previsão de sentimento baseada no texto e no modelo escolhido.
    """
    text_vector = vectorizer.transform([text])  # Transforma o texto em uma representação numérica
    prediction = model.predict(text_vector)[0]
    sentiment = assign_sentiment_alias(int(prediction))
    return sentiment
