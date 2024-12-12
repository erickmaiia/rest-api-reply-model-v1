from sqlalchemy import Column, Integer, String, JSON
from app.database.db_connection import Base

class ModelPrediction(Base):
    __tablename__ = 'model_predictions'
    id = Column(Integer, primary_key=True, index=True)
    prompt = Column(String, index=True)
    model = Column(String)
    sentiment = Column(String)

class MultiModelPrediction(Base):
    __tablename__ = "multi_model_predictions"

    id = Column(Integer, primary_key=True, index=True)
    prompt = Column(String, nullable=False)
    sentiment_results = Column(JSON, nullable=False)  # Armazena os resultados dos modelos como JSON
