# Use a imagem oficial do Python como base
FROM python:3.12.3

# Defina o diretório de trabalho dentro do contêiner
WORKDIR /app

# Copie o arquivo requirements.txt para o diretório de trabalho
COPY requirements.txt .

# Instale as dependências do projeto
RUN pip install --no-cache-dir -r requirements.txt

# Copie todo o código fonte para o diretório de trabalho
COPY . .

# Exponha a porta em que o servidor FastAPI estará ouvindo
EXPOSE 8080

# Comando para iniciar o servidor FastAPI quando o contêiner for executado
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
