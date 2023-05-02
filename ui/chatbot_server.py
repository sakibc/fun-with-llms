from __future__ import annotations

from fastapi import (
    Cookie,
    Depends,
    FastAPI,
    Query,
    WebSocket,
    WebSocketDisconnect,
    WebSocketException,
    status,
)
from fastapi.responses import HTMLResponse
import uvicorn
import asyncio
from typing import List
import os
from dotenv import load_dotenv
import copy

load_dotenv()

CLIENT_TOKEN = os.getenv("CLIENT_TOKEN")

class ChatbotServer:
    def __init__(self, bot) -> None:
        self.bot = bot

        self.app = FastAPI()

    async def verify_token(self, token: str = Query(...)):
        if token != CLIENT_TOKEN:
            raise WebSocketException(code=status.WS_1008_POLICY_VIOLATION)
        
        return token

    async def serve(self):
        # composition using https://stackoverflow.com/a/74841885
        app: FastAPI = self.app

        @app.websocket("/ws")
        async def websocket_chat(websocket: WebSocket, token: str = Depends(self.verify_token)):
            await websocket.accept()
            bot = copy.deepcopy(self.bot)

            try:
                while True:
                    data = (await websocket.receive_json())["submittedText"]

                    generatingResponse = {
                        "state": "generating",
                    }

                    await websocket.send_json(generatingResponse)

                    response = bot.generate_response(data)

                    serverResponse = {
                        "state": "idle",
                        "generatedText": response,
                    }

                    await websocket.send_json(serverResponse)
            except WebSocketDisconnect:
                pass

        
        config = uvicorn.Config(app, host="127.0.0.1", port=7860)
        server = uvicorn.Server(config)
        await server.serve()

    def run(self):
        asyncio.run(self.serve())
