from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from app.schemas.text_analysis import MultiModelAnalysisRequest, MultiModelAnalysisResponse
from app.services.model_service import load_models
from app.services.prediction_service import predict_prompt_multi_models
from app.utils.preprocess import clean_input
from app.utils.db_operations import save_multi_model_prediction
from app.database.db import get_db

router = APIRouter()

@router.post("/multi_model_prediction", response_model=MultiModelAnalysisResponse)
def multi_model_prediction(request: MultiModelAnalysisRequest, db: Session = Depends(get_db)):
    if not request.prompt:
        raise HTTPException(status_code=400, detail="Missing 'prompt'")
    
    try:
        # Processamento e predição
        cleaned_prompt = clean_input(request.prompt)
        models = load_models()
        sentiment_results = predict_prompt_multi_models(models, cleaned_prompt)

        # Salvar no banco de dados usando função utilitária
        save_multi_model_prediction(db, request.prompt, sentiment_results)

        return MultiModelAnalysisResponse(
            prompt=request.prompt,
            sentiment_results=sentiment_results
        )
    
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))