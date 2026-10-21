from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import io
import os
import threading
from PIL import Image

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "u2netp.onnx")

# Photos larger than this (longest side, in pixels) are shrunk first
MAX_SIDE = 1200

_session = None
_lock = threading.Lock()


def get_session():
    """Load the small AI model only when first needed."""
    global _session
    if _session is None:
        with _lock:
            if _session is None:
                import onnxruntime as ort

                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 1
                opts.enable_cpu_mem_arena = False   # saves memory
                opts.enable_mem_pattern = False     # saves memory
                _session = ort.InferenceSession(
                    MODEL_PATH,
                    sess_options=opts,
                    providers=["CPUExecutionProvider"],
                )
    return _session


def remove_background(img: Image.Image) -> Image.Image:
    import numpy as np

    session = get_session()
    img = img.convert("RGB")
    w, h = img.size

    # Prepare the image for the u2netp model (320 x 320 input)
    small = img.resize((320, 320), Image.BILINEAR)
    arr = np.asarray(small, dtype=np.float32)
    arr = arr / max(float(arr.max()), 1e-6)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    arr = arr.transpose(2, 0, 1)[None, :, :, :].astype(np.float32)

    # Run the model
    input_name = session.get_inputs()[0].name
    pred = session.run(None, {input_name: arr})[0][0, 0]

    # Turn the model output into a mask, then resize it to the photo size
    pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
    mask = Image.fromarray((pred * 255).astype("uint8")).resize((w, h), Image.LANCZOS)

    result = img.copy()
    result.putalpha(mask)
    return result


@asynccontextmanager
async def lifespan(app: FastAPI):
    # The port is already open here; warm up the model in the background
    threading.Thread(target=get_session, daemon=True).start()
    yield


app = FastAPI(title="Background Removal API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/remove-bg")
def remove_bg(
    file: UploadFile = File(...),
    alpha_matting: bool = Form(False),  # accepted but ignored in this slim version
):
    try:
        img = Image.open(file.file).convert("RGB")

        if max(img.size) > MAX_SIDE:
            img.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)

        result = remove_background(img)

        buf = io.BytesIO()
        result.save(buf, format="PNG")
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Background removal failed: {e}")


if __name__ == "__main__":
    # Only used when running locally with: py -3.11 main.py
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000)
