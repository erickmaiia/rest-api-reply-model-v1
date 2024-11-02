# Use uma imagem base Python
FROM python:3.10-slim

# Definir o diretório de trabalho dentro do container
WORKDIR /app

# Copiar os arquivos de dependências para o container
COPY requirements.txt .

# Instalar as dependências
RUN pip install --no-cache-dir -r requirements.txt

# Instalar as dependências específicas do nltk
RUN python -m nltk.downloader punkt

# Copiar o código da aplicação para o diretório de trabalho
COPY . .

# Expor a porta em que a FastAPI será executada
EXPOSE 8000

# Comando para iniciar a aplicação usando Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
