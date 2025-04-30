# python-microservice/app/verification.py
import os
import logging # Use logging for better debug/error info

# --- IMPORTANT: Configure DeepFace/TensorFlow Logging ---
# Suppress excessive TensorFlow logs BEFORE importing DeepFace/TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # 0 = all, 1 = info, 2 = warning, 3 = error
logging.getLogger('tensorflow').setLevel(logging.ERROR)
# Suppress obnoxious PIL logs if they appear
logging.getLogger('PIL').setLevel(logging.WARNING)

# Now import DeepFace safely
DEEPFACE_IMPORT_ERROR = None # Initialize to None
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError as e:
    # Handle cases where DeepFace might not be installed
    DEEPFACE_AVAILABLE = False
    DEEPFACE_IMPORT_ERROR = str(e) # Assign the error string if import fails

# --- Verification Logic ---
def verify_identity(img1_path, img2_path):
    """
    Verifies if two images contain the same face using DeepFace.

    Args:
        img1_path (str): Path to the first image file (e.g., ID card).
        img2_path (str): Path to the second image file (e.g., selfie).

    Returns:
        dict: A dictionary containing the verification results or an error.
    """
    if not DEEPFACE_AVAILABLE:
        return {"success": False, "error": f"DeepFace library failed to import: {DEEPFACE_IMPORT_ERROR}", "details": "Ensure DeepFace and its dependencies (Tensorflow, etc.) are installed correctly in the Python environment."}

    try:
        # --- Perform Face Verification ---
        # Common models: 'VGG-Face', 'Facenet', 'Facenet512', 'ArcFace', 'Dlib', 'SFace'
        # Common backends: 'opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface', 'mediapipe'
        # enforce_detection=True: Raises an error if no face is found.
        # enforce_detection=False: Returns a result indicating no face found. Choose based on UX.
        result = DeepFace.verify(
            img1_path=img1_path,
            img2_path=img2_path,
            model_name='VGG-Face', # Keeping VGG-Face as it's a good balance of speed/accuracy
            detector_backend='opencv', # Changed from 'mtcnn' to 'opencv' for faster detection
            distance_metric='cosine', # 'cosine' or 'euclidean_l2' usually work well
            enforce_detection=False, # Changed to False to avoid errors when face detection is difficult
            align=True # Usually good to keep True for better accuracy
        )

        # --- Process Result ---
        similarity = (1 - result.get('distance', 1.0)) * 100 # Calculate similarity % (approx)
        is_match = result.get('verified', False)

        # Make thresholds slightly more lenient
        confidence = "low"
        if similarity >= 70: # Reduced from 75 to 70
             confidence = "high"
        elif similarity >= 55: # Reduced from 60 to 55
             confidence = "medium"

        return {
            "success": True,
            "match": is_match,
            "similarity": round(similarity, 2),
            "distance": round(result.get('distance', 1.0), 4),
            "threshold": round(result.get('threshold', 0.0), 4),
            "confidence": confidence,
            "model": result.get('model', 'N/A'),
            "detector_backend": result.get('detector_backend', 'N/A'),
            "message": "Face verification successful." if is_match else "Faces do not appear to match."
         }

    except ValueError as ve:
        # Specific error from DeepFace (e.g., face could not be detected in one/both images)
        err_str = str(ve).lower()
        details = "Face detection failed."
        if "face could not be detected" in err_str:
             details = "Could not detect a face in one or both images. Please ensure the face is clear and unobstructed."
        elif "more than one face" in err_str:
             details = "Multiple faces detected in one or both images. Please ensure only one face is present."

        return {"success": False, "error": "ValueError during verification", "details": details, "match": False}
    except FileNotFoundError as fnf:
         return {"success": False, "error": "FileNotFoundError", "details": f"Image file not found: {str(fnf)}", "match": False}
    except Exception as e:
        # Catch any other exceptions (e.g., invalid image format, library issues)
        logging.error(f"Unexpected error during face verification: {e}", exc_info=True)
        return {"success": False, "error": "An unexpected error occurred during verification.", "details": str(e), "match": False}

# Note: The __main__ block from the original script has been removed as it's not needed here.