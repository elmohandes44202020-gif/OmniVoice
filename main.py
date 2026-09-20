from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from gradio_client import Client, handle_file
import os
import uuid
import shutil
import traceback


app = FastAPI(
    title="Bayan OmniVoice API",
    version="1.0"
)


# اتصال بـ Hugging Face Space
client = Client("k2-fsa/OmniVoice")


OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


@app.get("/")
def home():
    return {
        "status": "online",
        "engine": "OmniVoice",
        "app": "Bayan"
    }


@app.post("/tts")
async def tts(
    text: str = Form(...),
    voice: UploadFile = File(None)
):

    try:

        request_id = str(uuid.uuid4())

        voice_path = None


        # حفظ الصوت المرجعي
        if voice:

            voice_path = f"{OUTPUT_DIR}/{request_id}_voice.mp3"

            with open(voice_path, "wb") as f:
                shutil.copyfileobj(
                    voice.file,
                    f
                )


        # استدعاء OmniVoice
        if voice_path:

            result = client.predict(
                text,
                handle_file(voice_path),
                api_name="/predict"
            )

        else:

            result = client.predict(
                text,
                api_name="/predict"
            )


        output = f"{OUTPUT_DIR}/{request_id}.wav"


        shutil.copy(
            result,
            output
        )


        return FileResponse(
            output,
            media_type="audio/wav",
            filename="bayan.wav"
        )


    except Exception as e:

        error = traceback.format_exc()

        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "details": error
            }
        )
