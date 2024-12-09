from fastapi import FastAPI
from app.routes import router as api_router
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Configuração do CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)

# Incluindo as rotas
app.include_router(api_router)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Text Analysis API"}