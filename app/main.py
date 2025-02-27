from fastapi import FastAPI

from app.config import Settings
from app.enums import Test
from model.transformer import get_model, Transformer

app = FastAPI()


@app.get('/health')
def health():
    s = Settings()
    return {'version': s.version, 'status': 'running', 'model': s.model_version}

@app.get("/")
async def root():
    return {"message": "Hello world!"}


@app.get("/generate")
async def generate_message():
    m: Transformer = get_model()
    message: str = m.generate()
    return {'message': message}


@app.get("/generate/{input}")
async def items(input: str):
    reply = input + ""
    return {"item_id": reply}



