from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Message(BaseModel):
    message: str


# Zoha ye jo ab FileResponse use kar rahi hai ye ab frontend open kar den gi na k root me message ko jase pehle tha 
# @app.get("/")
# def root():
#     return {
#         "message": "The backend is now running. Thanks to Allah."
#     }
@app.get("/")
def root():
    return FileResponse("index.html")


@app.post("/chat")
def chat(data: Message):
    if data.message == "hello":
        return {
            "message": "Hello, how can I help you?"
        }
    else:
        return {
            "message": "I don't understand your message. Zoha did not add these kind of things in me yet. When she adds them, I'll be able to reply properly — for now please wait."
        }