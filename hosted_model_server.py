from __future__ import annotations

from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from llm.model import Model
import uvicorn

import os
import argparse
import json

load_dotenv()

models = [name.split(".")[0] for name in os.listdir("models")]

parser = argparse.ArgumentParser()
parser.add_argument(
    "model_name",
    help="Name of the model to use",
    choices=models,
)
parser.add_argument(
    "--size",
    help="Size of the model to use, depends on model chosen",
)
parser.add_argument(
    "--backend",
    help="Backend to use for model",
    choices=["cpu", "cuda", "mps"],
    default="cpu",
)

args = parser.parse_args()

with open(os.path.join("models", f"{args.model_name}.json")) as f:
    model_config = json.load(f)

HOSTED_MODEL_TOKEN = os.getenv("HOSTED_MODEL_TOKEN")

ml_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    ml_models["model"] = Model(
        model_name=args.model_name,
        model_config=model_config,
        size=args.size,
        backend=args.backend,
    )

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
    generated_text = ml_models["model"].generate(
        text_generation_input.prompt, text_generation_input.stop
    )

    return {"generated_text": generated_text}


@app.get("/health", dependencies=[Depends(verify_token)])
async def health():
    return {
        "status": "ok",
        "model_name": args.model_name,
    }


uvicorn.run(app, host="0.0.0.0", port=8000)
