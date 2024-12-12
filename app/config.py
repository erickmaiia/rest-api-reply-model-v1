import os

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
    "BERT": os.path.join(ML_MODELS_PATH + "\\BERT", "BERT.safetensors"),
}


# Caminho do vetorizador
VECTORIZER_PATH = os.path.join(ML_MODELS_PATH, "tfidf_vectorizer.pkl")
BERT_CONFIG_PATH = os.path.join(ML_MODELS_PATH + "\\BERT", "config.json")