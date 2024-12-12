from app.config import MODEL_PATHS, BERT_CONFIG_PATH
from joblib import load
from app.services.bert_service import load_bert

def load_model(model: str):
    if model not in MODEL_PATHS:
        raise ValueError(f"Model {model} not found.")
    
    model_path = MODEL_PATHS[model]


    if model == "BERT":
        model = load_bert(model_path, BERT_CONFIG_PATH)
        return model
    else:
        model = load(model_path)
        
    return model

def load_models():
    models = {}
    for model in MODEL_PATHS:
        models[model] = load_model(model)
    return models