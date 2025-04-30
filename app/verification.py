# python-microservice/app/verification.py
import os
import logging # Use logging for better debug/error info

# --- TensorFlow Import and Configuration ---
# Suppress excessive TensorFlow logs BEFORE importing
logging.info("Configuring TF log level in verification.py...")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('PIL').setLevel(logging.WARNING)
logging.info("TF log level configured.")

TF_AVAILABLE = False
TF_IMPORT_ERROR = None
tf = None
try:
    import tensorflow as tf
    logging.info(f"TensorFlow version {tf.__version__} imported successfully.")
    TF_AVAILABLE = True
    # Explicitly configure TF to use CPU ONLY upon import
    try:
        physical_devices_gpu = tf.config.list_physical_devices('GPU')
        logging.info(f"Physical GPUs detected by TF: {physical_devices_gpu}")
        if physical_devices_gpu:
            # This shouldn't happen with tensorflow-cpu, but as a safeguard:
            tf.config.set_visible_devices([], 'GPU')
            logging.warning("TensorFlow detected GPUs, but explicitly set visible GPUs to none.")
        else:
            logging.info("No physical GPUs detected by TensorFlow, CPU will be used.")
        physical_devices_cpu = tf.config.list_physical_devices('CPU')
        logging.info(f"Physical CPUs detected by TF: {physical_devices_cpu}")
    except Exception as e:
        logging.error(f"Error configuring TensorFlow devices: {e}")
except ImportError as e:
    logging.error(f"TensorFlow import failed: {e}", exc_info=True)
    TF_IMPORT_ERROR = str(e)

# --- DeepFace Import ---
DEEPFACE_IMPORT_ERROR = None
DEEPFACE_AVAILABLE = False
DeepFace = None
if TF_AVAILABLE:
    logging.info("Attempting to import DeepFace...")
    try:
        from deepface import DeepFace
        DEEPFACE_AVAILABLE = True
        logging.info("DeepFace imported successfully.")
    except ImportError as e:
        logging.error(f"DeepFace import failed: {e}", exc_info=True)
        DEEPFACE_IMPORT_ERROR = str(e)
else:
    logging.warning("Skipping DeepFace import because TensorFlow is not available.")
    DEEPFACE_IMPORT_ERROR = f"TensorFlow failed to import: {TF_IMPORT_ERROR}"


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
    logging.info(f"Entered verify_identity. TF available: {TF_AVAILABLE}, DeepFace available: {DEEPFACE_AVAILABLE}")
    if not TF_AVAILABLE:
        logging.error(f"Attempted verification, but TensorFlow not available. Import error: {TF_IMPORT_ERROR}")
        return {"success": False, "error": f"TensorFlow library failed to import: {TF_IMPORT_ERROR}", "details": "Ensure TensorFlow (tensorflow-cpu) is installed correctly."}
    if not DEEPFACE_AVAILABLE:
        logging.error(f"Attempted verification, but DeepFace not available. Import error: {DEEPFACE_IMPORT_ERROR}")
        return {"success": False, "error": f"DeepFace library failed to import: {DEEPFACE_IMPORT_ERROR}", "details": "Ensure DeepFace is installed correctly."}

    # --- Explicitly configure TF devices AGAIN just before the call (extra safety) ---
    try:
        physical_devices_gpu = tf.config.list_physical_devices('GPU')
        logging.info(f"[verify_identity] Physical GPUs detected by TF: {physical_devices_gpu}")
        if physical_devices_gpu:
            tf.config.set_visible_devices([], 'GPU')
            logging.warning("[verify_identity] Explicitly set visible GPUs to none before DeepFace call.")
        else:
            logging.info("[verify_identity] No physical GPUs detected before DeepFace call.")
    except Exception as e:
        logging.error(f"[verify_identity] Error configuring TensorFlow devices before DeepFace call: {e}")
        # Decide if we should proceed or return an error
        # return {"success": False, "error": "Internal Configuration Error", "details": "Failed to configure TensorFlow devices.", "match": False}

    logging.info(f"Attempting DeepFace.verify on {img1_path} and {img2_path} using detector_backend='ssd'")
    try:
        # --- Perform Face Verification --- Switch detector backend
        result = DeepFace.verify(
            img1_path=img1_path,
            img2_path=img2_path,
            model_name='VGG-Face',
            detector_backend='ssd', # Changed from 'opencv' to 'ssd'
            distance_metric='cosine',
            enforce_detection=False,
            align=True
        )
        logging.info(f"DeepFace.verify call completed. Raw result: {result}")

        # --- Process Result ---
        similarity = (1 - result.get('distance', 1.0)) * 100 # Calculate similarity % (approx)
        is_match = result.get('verified', False)

        # Make thresholds slightly more lenient
        confidence = "low"
        if similarity >= 70:
             confidence = "high"
        elif similarity >= 55:
             confidence = "medium"

        return {
            "success": True,
            "match": is_match,
            "similarity": round(similarity, 2),
            "distance": round(result.get('distance', 1.0), 4),
            "threshold": round(result.get('threshold', 0.0), 4),
            "confidence": confidence,
            "model": result.get('model', 'N/A'),
            "detector_backend": result.get('detector_backend', 'N/A'), # Should now report 'ssd'
            "message": "Face verification successful." if is_match else "Faces do not appear to match."
         }

    except ValueError as ve:
        logging.warning(f"ValueError during DeepFace.verify: {ve}")
        err_str = str(ve).lower()
        details = "Face detection failed."
        if "face could not be detected" in err_str:
             details = "Could not detect a face in one or both images. Please ensure the face is clear and unobstructed."
        elif "more than one face" in err_str:
             details = "Multiple faces detected in one or both images. Please ensure only one face is present."

        return {"success": False, "error": "ValueError during verification", "details": details, "match": False}
    except FileNotFoundError as fnf:
         logging.error(f"FileNotFoundError during DeepFace.verify: {fnf}")
         return {"success": False, "error": "FileNotFoundError", "details": f"Image file not found: {str(fnf)}", "match": False}
    except Exception as e:
        logging.error(f"Unexpected error during face verification in verify_identity: {e}", exc_info=True)
        # Check if the error message contains CUDA/GPU references
        error_detail = str(e)
        if "cuda" in error_detail.lower() or "gpu" in error_detail.lower():
            logging.error("Error message contains CUDA/GPU references, indicating potential GPU initialization issue.")
            error_detail = "Internal error related to device configuration. Please check service logs."
        return {"success": False, "error": "An unexpected error occurred during verification.", "details": error_detail, "match": False}

# Note: The __main__ block from the original script has been removed as it's not needed here.