import os
import io
import numpy as np  # <-- Явный импорт
import torch
import torchaudio
from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.responses import Response

app = FastAPI(title="Silero Voice API")

# Оптимизация для CPU и бесплатных тарифов
torch.set_num_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Глобальные переменные
tts_model = None
stt_model = None
tts_speakers = None

@app.on_event("startup")
async def load_models():
    global tts_model, stt_model, tts_speakers
    try:
        # TTS - загрузка с явным trust_repo
        tts_model, example_text = torch.hub.load(
            repo_or_dir="snakers4/silero-models",
            model="silero_tts",
            language="ru",
            speaker="kseniya",
            trust_repo=True
        )
        tts_model.to("cpu")
        tts_model.eval()
        tts_speakers = example_text.speakers
        
        # STT
        stt_model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-models",
            model="silero_stt",
            trust_repo=True
        )
        stt_model.to("cpu")
        stt_model.eval()
        
        print("✅ Models loaded successfully")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        raise

@app.post("/tts")
def text_to_speech(text: str = Form(...), speaker: str = Form("kseniya")):
    if tts_model is None:
        raise HTTPException(503, "Model not loaded yet")
    if speaker not in tts_speakers:
        raise HTTPException(400, f"Available speakers: {', '.join(tts_speakers)}")
    
    try:
        audio = tts_model.apply_tts(text=text, speaker=speaker, sample_rate=48000)
        buffer = io.BytesIO()
        torchaudio.save(buffer, audio, 48000, format="wav")
        return Response(content=buffer.getvalue(), media_type="audio/wav")
    except Exception as e:
        raise HTTPException(500, f"TTS error: {str(e)}")

@app.post("/stt")
async def speech_to_text(file: UploadFile):
    if stt_model is None:
        raise HTTPException(503, "Model not loaded yet")
    
    try:
        audio_bytes = await file.read()
        buffer = io.BytesIO(audio_bytes)
        waveform, sample_rate = torchaudio.load(buffer)
        
        if sample_rate != 16000:
            transform = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = transform(waveform)
            
        text = stt_model(waveform.squeeze())
        return {"text": text}
    except Exception as e:
        raise HTTPException(500, f"STT error: {str(e)}")

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "tts_loaded": tts_model is not None,
        "stt_loaded": stt_model is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
