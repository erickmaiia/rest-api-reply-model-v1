from fastapi import FastAPI, HTTPException
from app.models import TextPredictRequest, TextPredictResponse, TextPredictResponseAll
from app.ml_models.model_utils import load_text_model, predict_text, predict_text_all_models
from fastapi.middleware.cors import CORSMiddleware

# Inicializar o FastAPI
app = FastAPI()

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

@app.post("/predict_text", response_model=TextPredictResponse)
def predict_text_endpoint(request: TextPredictRequest):
    try:
        # Carrega o modelo baseado na escolha do usuário
        model, vectorizer = load_text_model(request.model)

        # Usa a função predict_text para fazer a previsão com o modelo e o vetor carregados
        prediction = predict_text(model, vectorizer, request.text)

        # Converte a previsão para string
        prediction_str = str(prediction)

        return TextPredictResponse(prediction=prediction_str)
    except Exception as e:
        # Retorna uma exceção HTTP em caso de erro
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/predict_text_all_models", response_model=TextPredictResponseAll)
def predict_text_endpoint(request: TextPredictRequest):
    try:
        # Carrega o modelo baseado na escolha do usuário
        models, vectorizer = load_text_model(request.model)

        # Usa a função predict_text para fazer a previsão com o modelo e o vetor carregados
        predictions = predict_text_all_models(models, vectorizer, request.text)

        return TextPredictResponseAll(prediction=predictions)
    except Exception as e:
        # Retorna uma exceção HTTP em caso de erro
        raise HTTPException(status_code=500, detail=str(e))