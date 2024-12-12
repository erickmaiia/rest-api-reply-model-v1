from sqlalchemy.orm import Session
from app.schemas.db_schemas import ModelPrediction, MultiModelPrediction

def save_model_prediction(db: Session, prompt: str, model: str, sentiment: str) -> ModelPrediction:
    """
    Salva a previsão de modelo no banco de dados.
    
    Args:
        db (Session): Sessão do banco de dados.
        prompt (str): Texto analisado.
        model (str): Modelo utilizado.
        sentiment (str): Sentimento retornado.

    Returns:
        ModelPrediction: Objeto salvo no banco de dados.
    """
    db_prediction = ModelPrediction(
        prompt=prompt,
        model=model,
        sentiment=sentiment
    )
    db.add(db_prediction)
    db.commit()
    db.refresh(db_prediction)
    return db_prediction

def save_multi_model_prediction(db: Session, prompt: str, sentiment_results: dict) -> MultiModelPrediction:
    """
    Salva os resultados de previsão de múltiplos modelos no banco de dados.
    
    Args:
        db (Session): Sessão do banco de dados.
        prompt (str): Texto analisado.
        sentiment_results (dict): Resultados de sentimentos para cada modelo.

    Returns:
        MultiModelPrediction: Objeto salvo no banco de dados.
    """
    db_prediction = MultiModelPrediction(
        prompt=prompt,
        sentiment_results=sentiment_results
    )
    db.add(db_prediction)
    db.commit()
    db.refresh(db_prediction)
    return db_prediction
