# Sentiment analysis API with FastAPI and a neural network model

This is an API built with [FastAPI](https://fastapi.tiangolo.com/) that utilizes an neural networks model for sentiment analysis on texts. The API receives a text via POST method and returns the sentiment associated with that text.

# About the models

Access this [repository](https://github.com/erickmaiia/financial-sentiment-analysis-ml)

## Routes

### Text Processing

#### `POST /predict_text`

This route receives a JSON object containing the text to be processed and returns the sentiment associated with that text.

#### Parameters

- `text` (string): The text to be analyzed.
- `model` (string): The model that will be used

#### Request Example

```json
{
  "text": "$ESI on lows, down $1.50 to $2.50 BK a real possibility."
  "model": "SVM"
}
```

#### Response Example

```json
{
  "sentiment": "negative"
}
```

#### `POST /predict_text_all_models`

This route receives a JSON object containing the text to be processed and returns all sentiment associated with that text across all models.

#### Parameters

- `text` (string): The text to be analyzed.
- `model` (string): The model that will be used

#### Request Example

```json
{
  "text": "$ESI on lows, down $1.50 to $2.50 BK a real possibility."
  "model": "All"
}
```

#### Response Example

```json
{
    "prediction": {
        "Decision Tree": {
            "sentiment": "positive",
            "inference_time_ms": 0.0,
            "confidence": 100.0
        },
        "Multilayer Perceptron": {
            "sentiment": "positive",
            "inference_time_ms": 0.0,
            "confidence": 99.79286726748342
        },
        "Random Forest": {
            "sentiment": "neutral",
            "inference_time_ms": 4.0,
            "confidence": 72.0
        },
        "Gradient Boosting": {
            "sentiment": "neutral",
            "inference_time_ms": 1.0,
            "confidence": 65.40153464625972
        },
        "Naive Bayes": {
            "sentiment": "neutral",
            "inference_time_ms": 0.0,
            "confidence": 58.39391454926953
        },
        "XGBoost": {
            "sentiment": "neutral",
            "inference_time_ms": 4.0,
            "confidence": 55.5428581237793
        },
        "LightGBM": {
            "sentiment": "positive",
            "inference_time_ms": 79.11,
            "confidence": 54.753866556500896
        },
        "AdaBoost": {
            "sentiment": "neutral",
            "inference_time_ms": 6.0,
            "confidence": 34.21799931861368
        },
        "SVM": {
            "sentiment": "positive",
            "inference_time_ms": 1.0,
            "confidence": null
        },
        "sentiment_distribution": {
            "positive": 4,
            "neutral": 5,
            "negative": 0
        }
    }
}
```

## Production API

The API is hosted on the GCP. You can access it at https://rest-api-reply-model-v1.onrender.com/.

You can visually access [here](https://interface-reply-model.vercel.app/) or click in the image.

[<img src="./image/interface-reply-model.vercel.app.png" alt="img">](https://interface-reply-model.vercel.app/)
