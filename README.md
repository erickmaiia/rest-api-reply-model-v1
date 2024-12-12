# Financial Sentiment Analysis API  

An API built with **FastAPI** for text analysis and sentiment detection, focusing on financial-related content.  

## Features  

- **Sentiment Analysis**: Returns sentiment (positive, negative, or neutral) based on the provided financial text.  
- **Machine Learning Models**: Integrates algorithms such as SVM, Naive Bayes, XGBoost, LightGBM, and others.  
- **Text Preprocessing**: Cleans and normalizes text before analysis.  

---  

## Technologies Used  

Key technologies and libraries used in this project include:  

- **FastAPI**: Modern, high-performance backend framework.  
- **Pydantic**: For data validation and serialization.  
- **Scikit-learn**: Tools for machine learning and data mining.  
- **Joblib**: For serialization of trained models.  
- **NLTK**: Tools for natural language processing.  
- **PostgreSQL**: For storing analysis data, now hosted on **Neon**.  

---  

## API Endpoints  

**Interactive Documentation Access**:  

- Swagger UI: [https://rest-api-reply-model-v1.fly.dev/docs](https://rest-api-reply-model-v1.fly.dev/docs)
- ReDoc: [https://rest-api-reply-model-v1.fly.dev/redoc](https://rest-api-reply-model-v1.fly.dev/redoc)

### Endpoints Overview  

#### 1. **`POST /model_prediction`**  
Performs sentiment analysis based on a provided text and specified model.  

#### 2. **`POST /multi_model_prediction`**  
Performs sentiment analysis using multiple models on a provided text.  

- **Available Models**:  
  - `Naive Bayes`  
  - `SVM`  
  - `XGBoost`  
  - `LightGBM`  
  - `Multilayer Perceptron`  
  - `Gradient Boosting`  
  - `Random Forest`  
  - `AdaBoost`  
  - `Decision Tree`  

---  

## About the Models  

Access the repository for more details:  

- Financial Market Sentiment Prediction: [https://github.com/erickmaiia/ml-financial-sentiment-analysis](https://github.com/erickmaiia/ml-financial-sentiment-analysis)  
