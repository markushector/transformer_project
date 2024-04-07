from fastapi import APIRouter

from app.config import Settings


api_router = APIRouter()


@api_router.get('/health')
def health():
    s = Settings()
    return {'version': s.version, 'status': 'running', 'model': s.model_version}


@api_router.get("/")
async def root():

    return {"message": "Hello world!"}


@api_router.get("/generate")
async def generate_message():

    message: str = ""
    return {'message': message}


@api_router.get("/generate/{input}")
async def items(input: str):
    reply = input + ""
    return {"item_id": reply}