from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from rembg import remove, new_session
import io
import os

# Model files are stored inside backend/models
os.environ["U2NET_HOME"] = "./models"

app = FastAPI(title="Background Removal API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Fast model loads at start
fast_session = new_session("u2netp")

# HD model loads only the first time HD is used
hd_session = None


def get_hd_session():
    global hd_session
    if hd_session is None:
        hd_session = new_session("u2net")
    return hd_session


@app.get("/")
def root():
    return {"message": "Server running successfully"}


@app.post("/remove-bg")
async def remove_bg(file: UploadFile = File(...), mode: str = Form("fast")):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload an image file.",
        )

    try:
        input_bytes = await file.read()

        if mode == "hd":
            try:
                output_bytes = remove(
                    input_bytes,
                    session=get_hd_session(),
                    alpha_matting=True,
                    alpha_matting_foreground_threshold=240,
                    alpha_matting_background_threshold=10,
                    alpha_matting_erode_size=10,
                )
            except Exception as matting_error:
                print("HD smoothing failed, using normal mode:", matting_error)
                output_bytes = remove(input_bytes, session=fast_session)
        else:
            output_bytes = remove(input_bytes, session=fast_session)

        return StreamingResponse(io.BytesIO(output_bytes), media_type="image/png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


import os
import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)