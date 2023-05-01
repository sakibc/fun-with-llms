from __future__ import annotations

from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from model import Model

import os

load_dotenv()

HOSTED_MODEL_TOKEN = os.getenv("HOSTED_MODEL_TOKEN")

ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    ml_models["model"] = Model(model_config={
            "path": "decapoda-research/llama-7b-hf",
            "lora": "tloen/alpaca-lora-7b",
        })
    
    yield

    ml_models.clear()

app = FastAPI(lifespan=lifespan)
http_bearer = HTTPBearer()

class TextGenerationInput(BaseModel):
    prompt: str
    stop: Optional[List[str]]

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(http_bearer)):
    token = credentials.credentials
    if token != HOSTED_MODEL_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.post("/generate-text", dependencies=[Depends(verify_token)])
async def generate_text(text_generation_input: TextGenerationInput):
    generated_text = ml_models["model"].generate(text_generation_input.prompt, text_generation_input.stop)

    return {"generated_text": generated_text}
