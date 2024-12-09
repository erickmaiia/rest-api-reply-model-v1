from fastapi import APIRouter
from .single_model import router as single_model_router
from .multi_model import router as multi_model_router

router = APIRouter()

router.include_router(single_model_router, tags=["Single Model Prediction"])
router.include_router(multi_model_router, tags=["Multi Model Prediction"])
