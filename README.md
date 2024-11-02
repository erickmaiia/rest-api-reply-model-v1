# Sentiment analysis API with FastAPI and a neural network model

This is an API built with [FastAPI](https://fastapi.tiangolo.com/) that utilizes an neural networks model for sentiment analysis on texts. The API receives a text via POST method and returns the sentiment associated with that text.

# About model

Access this [repository](https://www.kaggle.com/code/supreethrao/bert-s-a-stock-market-guru-86-22-huggingface)

## Routes

### Text Processing

#### `POST /process_text/`

This route receives a JSON object containing the text to be processed and returns the sentiment associated with that text.

#### Parameters

- `text` (string): The text to be analyzed.

#### Request Example

```json
{
  "text": "$ESI on lows, down $1.50 to $2.50 BK a real possibility."
}
```

#### Response Example

```json
{
  "sentiment": "negative"
}
```

## Production API

The API is hosted on the GCP. You can access it at https://rest-api-dus2eb35vq-uc.a.run.app/process_text/.

You can visually access [here](https://interface-reply-model.vercel.app/) or click in the image.

[<img src="./image/interface-reply-model.vercel.app.png" alt="img">](https://interface-reply-model.vercel.app/)
