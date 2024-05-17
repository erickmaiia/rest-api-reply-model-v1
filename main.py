import joblib
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer
from torch import device, no_grad

app = FastAPI()
model = joblib.load('./models/neural_networks.joblib')
device = device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
tokenizer = AutoTokenizer.from_pretrained("./models/tokenizer")


def assign_sentiment_alias(prediction):
    if prediction == 0:
        return "negative"
    elif prediction == 1:
        return "neutral"
    elif prediction == 2:
        return "positive"
    else:
        return "unknown"


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)

class TextModel(BaseModel):
    text: str

@app.get("/")
def welcome():
    return "Welcome to the sentiment analysis API!"

@app.post("/process_text/")
def process_text(content: TextModel):
    encoded_input = tokenizer(content.text, padding=True, truncation=True, return_tensors="pt")

    encoded_input = {key: tensor.to(device) for key, tensor in encoded_input.items()}

    with no_grad():
        output = model(**encoded_input)
        
    predictions = output.logits.argmax(dim=-1)
    prediction_value = predictions.item()

    return {"sentiment": assign_sentiment_alias(prediction_value)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="localhost", port=8080, reload=True)
