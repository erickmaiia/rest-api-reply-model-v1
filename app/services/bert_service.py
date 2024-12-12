
from transformers import AutoModelForSequenceClassification, AutoConfig
from safetensors.torch import load_file
import torch

def load_bert(model_path ,BERT_CONFIG_PATH):
        # Carregar a configuração do modelo
    config = AutoConfig.from_pretrained(BERT_CONFIG_PATH)

    # Criar o modelo
    model = AutoModelForSequenceClassification.from_config(config)

    # Carregar os pesos do modelo com safetensors
    state_dict = load_file(model_path)
    model.load_state_dict(state_dict)

    return model

def bert_predict(prompt, model, tokenizer):
    encoded_input = tokenizer(prompt.lower(), padding=True, truncation=True, return_tensors="pt")
            # Passar os tokens para o modelo
    outputs = model(**encoded_input)

    # Obter as logits (pontuação da classificação)
    logits = outputs.logits

    # Determinar a classe predita (para classificação binária ou multi-classe)
    predicted_class = torch.argmax(logits, dim=-1).item()

    # Mapear a classe predita para um sentimento
    return predicted_class