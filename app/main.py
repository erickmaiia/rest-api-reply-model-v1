from fastapi import FastAPI, HTTPException
from app.models import TextPredictRequest, TextPredictResponse, TextPredictResponseAll
from app.model_utils import load_text_model, predict_text, predict_text_all_models
from fastapi.middleware.cors import CORSMiddleware

# Inicializar o FastAPI
app = FastAPI()

# Adiciona middleware CORS para permitir chamadas de qualquer origem
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Text Analysis API"}

def handle_prediction_request(model_name: str, text: str, all_models: bool = False):
    """
    Função auxiliar para centralizar a lógica de carregamento de modelos e previsão,
    tanto para um único modelo quanto para todos os modelos.
    """
    try:
        if all_models:
            models, vectorizer = load_text_model("All")  # Carrega todos os modelos
            predictions = predict_text_all_models(models, vectorizer, text)
            return TextPredictResponseAll(prediction=predictions)
        else:
            model, vectorizer = load_text_model(model_name)  # Carrega o modelo selecionado
            prediction = predict_text(model, vectorizer, text)
            return TextPredictResponse(prediction=str(prediction))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")

@app.post("/predict_text", response_model=TextPredictResponse)
def predict_text_endpoint(request: TextPredictRequest):
    """
    Endpoint para previsão com um único modelo escolhido pelo usuário.
    """
    return handle_prediction_request(request.model, request.text)

@app.post("/predict_text_all_models", response_model=TextPredictResponseAll)
def predict_text_all_models_endpoint(request: TextPredictRequest):
    """
    Endpoint para previsão com todos os modelos disponíveis.
    """
    return handle_prediction_request(request.model, request.text, all_models=True)
