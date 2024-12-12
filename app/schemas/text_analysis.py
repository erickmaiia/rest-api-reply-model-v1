from pydantic import BaseModel
from typing import Dict

class TextAnalysisRequest(BaseModel):
    prompt: str
    model: str
    class Config:
        # Impede campos extras na requisição
        extra = "forbid"

class TextAnalysisResponse(BaseModel):
    prompt: str
    model: str
    sentiment: str

class MultiModelAnalysisRequest(BaseModel):
    prompt: str
    class Config:
        # Impede campos extras na requisição
        extra = "forbid"

class MultiModelAnalysisResponse(BaseModel):
    prompt: str
    sentiment_results: Dict[str, str]  # Dicionário com o nome do modelo e seu respectivo sentimento
