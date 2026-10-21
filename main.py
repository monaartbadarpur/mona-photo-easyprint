from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import os

# Point rembg directly to the local models directory
os.environ["U2NET_HOME"] = os.path.join(os.path.dirname(__file__), "models")

from rembg import remove, new_session
from PIL import Image
import io

app = FastAPI(title="Background Removal API")

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
]

# Initialize the lightweight model session from local folder
print("Initializing rembg session from local models...")
try:
    fast_session = new_session("u2netp")
    print("Session initialized successfully!")
except Exception as e:
    print("Error initializing session:", e)
    fast_session = None

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

        # RAM protection: Resize huge images to max 1200px dimension
        try:
            img = Image.open(io.BytesIO(input_bytes))
            img.verify()
            img = Image.open(io.BytesIO(input_bytes))
            
            max_dimension = 1200
            if max(img.size) > max_dimension:
                img.thumbnail((max_dimension, max_dimension))
                buffer = io.BytesIO()
                img_format = img.format if img.format in ["PNG", "JPEG"] else "JPEG"
                img.save(buffer, format=img_format)
                input_bytes = buffer.getvalue()
        except Exception as img_err:
            print("Image preprocessing warning:", img_err)

        if fast_session:
            output_bytes = remove(input_bytes, session=fast_session)
        else:
            output_bytes = remove(input_bytes, model_name="u2netp")

        return StreamingResponse(io.BytesIO(output_bytes), media_type="image/png")

    except Exception as e:
        print("Background removal execution error:", str(e))
        raise HTTPException(status_code=500, detail=str(e))


import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)