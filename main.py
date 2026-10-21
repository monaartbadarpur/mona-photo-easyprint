from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import io
import os
import threading
from PIL import Image

# Point rembg directly to the local models directory
os.environ["U2NET_HOME"] = os.path.join(os.path.dirname(__file__), "models")

# Photos larger than this (in pixels, longest side) are shrunk before
# processing, so the free Render server does not run out of memory.
MAX_SIDE = 2000

_session = None
_lock = threading.Lock()


def get_session():
    """Load the AI model only when first needed, not at server startup."""
    global _session
    if _session is None:
        with _lock:
            if _session is None:
                from rembg import new_session  # lazy import (heavy)
                _session = new_session("u2netp")
    return _session


@asynccontextmanager
async def lifespan(app: FastAPI):
    # The port is already open when this runs.
    # Warm up the model in the background so the first request is faster.
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
    alpha_matting: bool = Form(False),
):
    # Plain "def" (not "async def") so the heavy work runs in a worker thread
    # and the server stays responsive.
    from rembg import remove  # lazy import (heavy)

    try:
        img = Image.open(file.file).convert("RGB")

        # Shrink very large photos to protect server memory
        if max(img.size) > MAX_SIDE:
            img.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)

        if alpha_matting:
            result = remove(
                img,
                session=get_session(),
                alpha_matting=True,
                alpha_matting_foreground_threshold=240,
                alpha_matting_background_threshold=10,
                alpha_matting_erode_size=10,
            )
        else:
            result = remove(img, session=get_session())

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
