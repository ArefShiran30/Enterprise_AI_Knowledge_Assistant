from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def home():
    return {"message": "Enterprise AI Assistant is running"}