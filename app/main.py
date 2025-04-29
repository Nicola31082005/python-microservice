# python-microservice/app/main.py
import os
import sys
import base64
import uuid
import logging
import shutil
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Union

# Import the verification logic from the local verification module
try:
    # Use relative import within the package
    from .verification import verify_identity, DEEPFACE_AVAILABLE, DEEPFACE_IMPORT_ERROR
except ImportError as e:
    logging.critical(f"Failed to import verification module: {e}")
    # Fallback for potential path issues during development/debugging
    try:
        sys.path.append(os.path.dirname(__file__)) # Add current dir
        from verification import verify_identity, DEEPFACE_AVAILABLE, DEEPFACE_IMPORT_ERROR
    except ImportError as e_inner:
        logging.critical(f"Failed to import verification module (fallback attempt): {e_inner}")
        DEEPFACE_AVAILABLE = False
        DEEPFACE_IMPORT_ERROR = f"Failed to import verification: {e_inner}"


# --- Configuration ---
# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Determine the base directory of the app for consistent path handling
APP_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_DIR = os.path.join(APP_BASE_DIR, "temp_images")

# --- Temporary File Handling ---
if not os.path.exists(TEMP_DIR):
    try:
        os.makedirs(TEMP_DIR)
        logging.info(f"Created temp directory: {TEMP_DIR}")
    except OSError as e:
        logging.error(f"Could not create temp directory {TEMP_DIR}: {e}")
        # Exit or handle appropriately if temp dir is essential
        sys.exit(f"Error: Cannot create temporary directory {TEMP_DIR}")

# Initialize FastAPI app
app = FastAPI(title="Face Verification Microservice")

# --- CORS Configuration ---
# Read allowed origins from environment variable, split by comma
# Provide a default value that includes common local development origins
allowed_origins_str = os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")
origins = [origin.strip() for origin in allowed_origins_str.split(',') if origin.strip()]

# Log the origins being used (good for debugging deployment)
logging.info(f"Configuring CORS for origins: {origins}")

# Ensure there's at least one origin if the env var was empty/invalid
if not origins:
    logging.warning("ALLOWED_ORIGINS environment variable resulted in an empty list. Falling back to default localhost.")
    origins = ["http://localhost:3000", "http://127.0.0.1:3000"] # Sensible default fallback

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, # Use the dynamic list read from env var
    allow_credentials=True,
    allow_methods=["POST", "GET"], # Limit methods if possible
    allow_headers=["Content-Type"], # Limit headers if possible
)

def save_base64_temp(base64_string: str, prefix: str = "") -> Union[str, None]:
    """Saves a base64 string to a temporary image file."""
    try:
        # Remove data URI prefix if present
        if "," in base64_string:
            header, encoded = base64_string.split(",", 1)
        else:
            encoded = base64_string

        # Pad base64 string if needed
        encoded += '=' * (-len(encoded) % 4)

        image_data = base64.b64decode(encoded)
        # Use a consistent image format if possible, or try to detect
        temp_filename = f"{prefix}{uuid.uuid4()}.jpg" # Assuming jpeg
        temp_filepath = os.path.join(TEMP_DIR, temp_filename)

        with open(temp_filepath, "wb") as f:
            f.write(image_data)
        logging.info(f"Saved temp file: {temp_filepath}")
        return temp_filepath
    except base64.binascii.Error as b64_err:
        logging.error(f"Base64 decoding error ({prefix}): {b64_err}")
        return None
    except Exception as e:
        logging.error(f"Error saving base64 temp file ({prefix}): {e}")
        return None

def cleanup_file(filepath: Union[str, None]):
    """Safely deletes a file."""
    if filepath and os.path.exists(filepath):
        try:
            os.remove(filepath)
            logging.info(f"Cleaned up temp file: {filepath}")
        except Exception as e:
            logging.warning(f"Failed to delete temp file {filepath}: {e}")

# --- Pydantic Models ---
class VerificationRequest(BaseModel):
    idImage: str = Field(..., description="Base64 encoded ID image string")
    selfieImage: str = Field(..., description="Base64 encoded Selfie image string")

# --- API Endpoints ---
@app.post("/verify")
async def handle_verification(request_body: VerificationRequest):
    """
    Handles the face verification request.
    """
    logging.info("Received /verify request")
    id_image_path = None
    selfie_image_path = None

    if not DEEPFACE_AVAILABLE:
         logging.error(f"DeepFace library unavailable: {DEEPFACE_IMPORT_ERROR}")
         raise HTTPException(
             status_code=503, # Service Unavailable
             detail={"success": False, "match": False, "error": "Service Unavailable", "details": "Face verification component is not available."}
         )

    try:
        logging.info("Saving temporary images...")
        id_image_path = save_base64_temp(request_body.idImage, "id-")
        selfie_image_path = save_base64_temp(request_body.selfieImage, "selfie-")

        if not id_image_path or not selfie_image_path:
             logging.error("Failed to save one or both temporary images.")
             # Clean up the one that might have been saved
             cleanup_file(id_image_path)
             cleanup_file(selfie_image_path)
             raise HTTPException(
                 status_code=400, # Bad Request (invalid base64 likely)
                 detail={"success": False, "match": False, "error": "Invalid Image Data", "details": "Could not decode or save one or both base64 image strings."}
             )

        logging.info("Temporary images saved. Calling verification logic...")
        result = verify_identity(id_image_path, selfie_image_path)
        logging.info(f"Verification result: {result}")

        status_code = 200 if result.get("success") else 400
        # Distinguish between verification logic errors (400) and unexpected server errors (500)
        if not result.get("success") and "error" in result:
            if result["error"] not in ["ValueError during verification", "FileNotFoundError"]:
                 status_code = 500 # Internal Server Error

        return JSONResponse(content=result, status_code=status_code)

    except HTTPException as http_exc:
        # Re-raise HTTPExceptions raised earlier (e.g., 503, 400)
        raise http_exc
    except Exception as e:
        logging.error(f"Unexpected error in /verify endpoint: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={"success": False, "match": False, "error": "Internal Server Error", "details": "An unexpected error occurred during processing."}
        )

    finally:
        logging.info("Cleaning up temporary files...")
        cleanup_file(id_image_path)
        cleanup_file(selfie_image_path)
        logging.info("Cleanup finished for request.")

@app.get("/")
def read_root():
    """Root endpoint for health check."""
    return {"message": "Face Verification Microservice is running"}

@app.get("/health")
def health_check():
    """Detailed health check including DeepFace availability."""
    return {
        "status": "ok",
        "deepface_available": DEEPFACE_AVAILABLE,
        "deepface_import_error": DEEPFACE_IMPORT_ERROR if not DEEPFACE_AVAILABLE else None
    }

# Note: Uvicorn is used to run the app, e.g.:
# cd python-microservice
# uvicorn app.main:app --reload --host 0.0.0.0 --port 8000