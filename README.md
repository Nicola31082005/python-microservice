# Face Verification Microservice

This Python microservice provides an API endpoint to verify if two images contain the same face using the DeepFace library.

## Setup

1.  **Clone the repository (if applicable).**
2.  **Navigate to the microservice directory:**
    ```bash
    cd python-microservice
    ```
3.  **Create a virtual environment:**
    ```bash
    python -m venv .venv
    # or use python3 if needed
    # python3 -m venv .venv
    ```
4.  **Activate the virtual environment:**
    - Windows (Command Prompt): `.venv\Scripts\activate`
    - Windows (PowerShell): `.venv\Scripts\Activate.ps1`
    - macOS/Linux: `source .venv/bin/activate`
5.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    _Note: DeepFace installation might take some time as it downloads models and dependencies (like TensorFlow)._

## Running the Service

### Local Development

For local development, run the FastAPI application using Uvicorn with auto-reload:

```bash
# Make sure you are in the python-microservice directory
# and the virtual environment is activated.

uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

- Using `--host 127.0.0.1` is often sufficient for local testing if only accessing from the same machine.
- The `--reload` flag automatically restarts the server when code changes.

### Production / Deployment (e.g., Render)

For production environments like Render, use Gunicorn to manage Uvicorn workers. Render will typically provide the port via the `PORT` environment variable.

**Render Start Command:**

Set this command in your Render service settings:

```bash
gunicorn -w 4 -k uvicorn.workers.UvicornWorker app.main:app --bind 0.0.0.0:$PORT
```

- `gunicorn`: The production WSGI server.
- `-w 4`: Number of worker processes. Adjust based on your Render plan (e.g., 2-4 workers is common).
- `-k uvicorn.workers.UvicornWorker`: Tells Gunicorn to use Uvicorn to handle asynchronous code.
- `app.main:app`: Path to your FastAPI application instance.
- `--bind 0.0.0.0:$PORT`: Binds to all network interfaces on the port specified by Render (`$PORT`).

**Do not use** `--reload` in production.

## API Endpoints

- **`GET /`**: Basic message indicating the service is running.
- **`GET /health`**: Health check endpoint showing service status and DeepFace availability.
- **`POST /verify`**:

  - Accepts a JSON body with two base64-encoded image strings:
    ```json
    {
      "idImage": "data:image/jpeg;base64,/9j/...",
      "selfieImage": "data:image/png;base64,iVBORw0..."
    }
    ```
  - Returns a JSON response with verification results:

    ```json
    // Example Success (Match)
    {
        "success": true,
        "match": true,
        "similarity": 78.54,
        "distance": 0.2146,
        "threshold": 0.40,
        "confidence": "high",
        "model": "VGG-Face",
        "detector_backend": "opencv",
        "message": "Face verification successful."
    }

    // Example Failure (No Match)
    {
        "success": true,
        "match": false,
        // ... other fields
        "message": "Faces do not appear to match."
    }

    // Example Error (e.g., face not detected)
    {
        "success": false,
        "error": "ValueError during verification",
        "details": "Could not detect a face in one or both images...",
        "match": false
    }
    ```

## Environment Variables

This service uses environment variables for configuration, especially important for deployment:

- **`ALLOWED_ORIGINS`**: (Required for Production) A comma-separated string of URLs allowed to make requests to this API (CORS).
  - **Example Render Value:** `https://your-nextjs-app.onrender.com,https://www.your-custom-domain.com`
  - **Local Development:** If not set, it defaults to `http://localhost:3000,http://127.0.0.1:3000`.
- **`PORT`**: (Provided by Render) The port the application should bind to. You don't set this manually in Render; use `$PORT` in the start command.

Set `ALLOWED_ORIGINS` in the Environment section of your Render service settings.
