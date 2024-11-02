from pydantic import BaseModel

class TextPredictRequest(BaseModel):
    text: str  # Campo para o texto de entrada
    model: str  # Campo para o modelo a ser utilizado

class TextPredictResponse(BaseModel):
    prediction: str  # Campo para a previsão gerada pelo modelo
