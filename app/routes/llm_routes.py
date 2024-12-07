from fastapi import APIRouter, HTTPException, Depends
from app.schemas.text_analysis import TextAnalysisRequest, TextAnalysisResponse, MultiModelAnalysisResponse, MultiModelAnalysisRequest
from app.schemas.db_schemas import ModelPrediction, MultiModelPrediction
from app.services.model_service import load_model, load_models
from app.services.prediction_service import predict_prompt, predict_prompt_multi_models
from app.utils.preprocess import clean_input

from sqlalchemy.orm import Session
from app.database.db import get_db

router = APIRouter()

@router.post("/model_prediction", response_model=TextAnalysisResponse)
def model_prediction(request: TextAnalysisRequest, db: Session = Depends(get_db)):
    """
    Analisa o texto com base no modelo escolhido e armazena a requisição no banco de dados.
    """
    if not request.prompt or not request.model:
        raise HTTPException(status_code=400, detail="Missing 'prompt' or 'model'")
    
    try:
        # Limpeza do prompt
        cleaned_prompt = clean_input(request.prompt)
        
        # Carregar o modelo específico
        model = load_model(request.model)
        
        # Realizar a previsão
        sentiment = predict_prompt(model, cleaned_prompt)

        # Armazenar a requisição no banco de dados
        db_prediction = ModelPrediction(
            prompt=request.prompt,
            model=request.model,
            sentiment=sentiment
        )
        db.add(db_prediction)
        db.commit()
        db.refresh(db_prediction)

        # Retorno estruturado
        return TextAnalysisResponse(
            prompt=request.prompt, 
            model=request.model, 
            sentiment=sentiment
        )
    
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/multi_model_prediction", response_model=MultiModelAnalysisResponse)
def multi_model_prediction(
    request: MultiModelAnalysisRequest, 
    db: Session = Depends(get_db)
):
    """
    Analisa o texto com base em todos os modelos e retorna os sentimentos de cada um.
    """
    if not request.prompt:
        raise HTTPException(status_code=400, detail="Missing 'prompt'")
    
    try:
        # Limpar o texto de entrada
        cleaned_prompt = clean_input(request.prompt)

        # Carregar todos os modelos
        models = load_models()

        # Obter os sentimentos para todos os modelos
        sentiment_results = predict_prompt_multi_models(models, cleaned_prompt)

        # Salvar os dados no banco de dados
        db_prediction = MultiModelPrediction(
            prompt=request.prompt,
            sentiment_results=sentiment_results
        )
        db.add(db_prediction)
        db.commit()
        db.refresh(db_prediction)

        # Preparar o retorno com todos os sentimentos
        return MultiModelAnalysisResponse(
            prompt=request.prompt,
            sentiment_results=sentiment_results  # Dicionário de resultados dos modelos
        )
    
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))