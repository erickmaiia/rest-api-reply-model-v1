from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from app.schemas.text_analysis import TextAnalysisRequest, TextAnalysisResponse
from app.services.model_service import load_model
from app.services.prediction_service import predict_prompt
from app.utils.preprocess import clean_input
from app.utils.db_operations import save_model_prediction
from app.database.db import get_db

router = APIRouter()

@router.post("/model_prediction", response_model=TextAnalysisResponse)
def model_prediction(request: TextAnalysisRequest, db: Session = Depends(get_db)):
    if not request.prompt or not request.model:
        raise HTTPException(status_code=400, detail="Missing 'prompt' or 'model'")
    
    try:
        # Preprocessamento e carga do modelo
        cleaned_prompt = clean_input(request.prompt)
        model = load_model(request.model)
        sentiment = predict_prompt(model, cleaned_prompt)

        # Salvar no banco de dados usando função utilitária
        save_model_prediction(db, request.prompt, request.model, sentiment)

        return TextAnalysisResponse(
            prompt=request.prompt, 
            model=request.model, 
            sentiment=sentiment
        )
    
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
