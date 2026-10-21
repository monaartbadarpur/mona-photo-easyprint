from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import io
import os

# Point rembg directly to the local models directory
os.environ["U2NET_HOME"] = os.path.join(os.path.dirname(__file__), "models")

from rembg import remove, new_session
from PIL import Image

app = FastAPI(title="Background Removal API")

# Allow all origins (or explicitly include trailing/non-trailing URLs)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins to eliminate CORS blocks
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)