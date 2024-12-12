from fastapi import APIRouter, HTTPException, Depends,Request
from sqlalchemy.orm import Session
from app.schemas.text_analysis import TextAnalysisRequest, TextAnalysisResponse
from app.services.model_service import load_model
from app.services.prediction_service import predict_prompt
from app.utils.preprocess import clean_input
from app.database.db_operations import save_model_prediction
from app.database.db_connection import get_db
from app.services.rate_limiter import rate_limit

router = APIRouter()

@router.post("/model_prediction", response_model=TextAnalysisResponse)
@rate_limit(max_calls=5, period=60)
async def model_prediction(
    request: Request,
    body: TextAnalysisRequest, 
    db: Session = Depends(get_db)):
    if not body.prompt or not body.model:
        raise HTTPException(status_code=400, detail="Missing 'prompt' or 'model'")
    
    try:
        # Preprocessamento e carga do modelo
        cleaned_prompt = clean_input(body.prompt)
        model = load_model(body.model)
        sentiment = predict_prompt(model, cleaned_prompt)

        # Salvar no banco de dados usando função utilitária
        save_model_prediction(db, request.prompt, request.model, sentiment)

        return TextAnalysisResponse(
            prompt=body.prompt, 
            model=body.model, 
            sentiment=sentiment
        )
    
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
