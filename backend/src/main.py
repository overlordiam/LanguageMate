from fastapi import FastAPI
from api.routers.chatbotInteraction import router

app = FastAPI(title="Language Chatbot API")

app.include_router(router, prefix="/api")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000) 