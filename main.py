from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
from gradio_client import Client, handle_file
import shutil
import uuid
import os

app = FastAPI(
    title="Bayan OmniVoice API",
    version="1.0"
)

client = Client("k2-fsa/OmniVoice")

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


@app.get("/")
def home():
    return {
        "status": "online",
        "engine": "Hugging Face OmniVoice",
        "service": "Bayan TTS"
    }


@app.post("/tts")
async def text_to_voice(
    text: str = Form(...),
    voice: UploadFile = File(None)
):

    request_id = str(uuid.uuid4())

    voice_path = None

    if voice:
        voice_path = f"{OUTPUT_DIR}/{request_id}_voice.wav"

        with open(voice_path, "wb") as f:
            shutil.copyfileobj(
                voice.file,
                f
            )


    result = client.predict(
        text,
        voice_path if voice_path else None,
        api_name="/_clone_fn"
    )


    output_file = f"{OUTPUT_DIR}/{request_id}.wav"

    shutil.copy(
        result,
        output_file
    )


    return FileResponse(
        output_file,
        media_type="audio/wav",
        filename="bayan_voice.wav"
    )
