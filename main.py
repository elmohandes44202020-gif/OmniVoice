from fastapi import FastAPI, Form, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from omnivoice import OmniVoice
import torch
import os
import uuid
import shutil
import traceback


app = FastAPI(
    title="Bayan OmniVoice API",
    version="1.0"
)


OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# تحميل النموذج مرة واحدة عند تشغيل السيرفر
model = OmniVoice.from_pretrained(
    "k2-fsa/OmniVoice",
    device_map="cuda:0",
    dtype=torch.float16
)


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
    voice: UploadFile = File(...),
    voice_text: str = Form("")
):

    try:

        uid = str(uuid.uuid4())

        voice_file = f"{OUTPUT_DIR}/{uid}_voice.wav"

        with open(voice_file, "wb") as f:
            shutil.copyfileobj(
                voice.file,
                f
            )


        result = model.generate(
            text=text,
            ref_audio=voice_file,
            ref_text=voice_text
        )


        output = f"{OUTPUT_DIR}/{uid}.wav"


        # حفظ الصوت الناتج
        result.save(output)


        return FileResponse(
            output,
            media_type="audio/wav",
            filename="bayan.wav"
        )


    except Exception as e:

        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "details": traceback.format_exc()
            }
        )
