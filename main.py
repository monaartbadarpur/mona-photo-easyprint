from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from rembg import remove, new_session
from PIL import Image
import io
import os

# Model files are stored inside backend/models
os.environ["U2NET_HOME"] = "./models"

app = FastAPI(title="Background Removal API")

# Updated CORS to allow your deployed frontend on Render + Localhost
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "https://desktop-passport-frontend.onrender.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
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
    # You can keep u2netp for RAM safety, or switch to 'u2net' if you want slightly sharper edges in HD mode 
    # and your server has enough headroom. For 512MB RAM, sticking to u2netp is safest.
        hd_session = new_session("u2netp")
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

        # Optional RAM protection: Resize huge images to max 1500px dimension
        try:
            img = Image.open(io.BytesIO(input_bytes))
            img.verify() # Verify it's a valid image
            img = Image.open(io.BytesIO(input_bytes)) # Reopen after verify
            
            max_dimension = 1500
            if max(img.size) > max_dimension:
                img.thumbnail((max_dimension, max_dimension))
                buffer = io.BytesIO()
                # Keep original format or save as PNG/JPEG for processing
                img_format = img.format if img.format in ["PNG", "JPEG"] else "JPEG"
                img.save(buffer, format=img_format)
                input_bytes = buffer.getvalue()
        except Exception as img_err:
            print("Image preprocessing warning:", img_err)
            # Proceed with raw bytes if PIL pre-check fails

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


import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)