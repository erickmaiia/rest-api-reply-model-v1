from fastapi import FastAPI, Request
from app.routes import router as api_router
from fastapi.middleware.cors import CORSMiddleware
from app.services.rate_limiter import rate_limit

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
@rate_limit(max_calls=5, period=60)
async def read_root(request: Request):
    return {"message": "Welcome to the Text Analysis API"}