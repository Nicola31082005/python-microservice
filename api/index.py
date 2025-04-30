# api/index.py
import sys
import os

# Add the parent directory (containing the 'app' module) to the Python path
# This allows us to import 'app.main'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the FastAPI app instance from your main application file
# Vercel will run this 'app' object
from app.main import app

# Optional: Add a simple root endpoint specific to the API route if needed
# Note: Endpoints defined in app.main will also be available
# @app.get("/api") # You can uncomment this if you want a specific /api route
# def handle_api_root():
#     return {"message": "Python backend entrypoint"}

# The 'app' variable is what Vercel expects to find and serve 