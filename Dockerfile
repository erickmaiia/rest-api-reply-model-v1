# Use uma imagem base Python
FROM python:3.12.7

# Atualizar o sistema e instalar libgomp
RUN apt-get update && \
    apt-get install -y libgomp1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Definir o diretório de trabalho dentro do container
WORKDIR /app

# Copiar o arquivo de dependências para o container
COPY requirements.txt .

# Instalar as dependências
RUN pip install --no-cache-dir -r requirements.txt

# Instalar pacotes específicos do nltk de uma vez
RUN python -m nltk.downloader punkt punkt_tab

# Copiar o código da aplicação para o diretório de trabalho
COPY . .

# Expor a porta em que a FastAPI será executada
EXPOSE 8000

# Comando para iniciar a aplicação usando Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
