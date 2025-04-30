# Use an official Python runtime as a parent image matching your local version
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Install libgl1 for DeepFace dependencies (graphics)
RUN apt-get update && apt-get install -y --no-install-recommends libgl1

# Install libglib2.0-0 for DeepFace dependencies
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0

# Copy the application code into the container
COPY app /app/app
COPY api /app/api
COPY requirements.txt /app/

# Install any dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port that FastAPI will run on
EXPOSE 8000

# Set environment variables (optional, but good practice)
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Command to run the FastAPI application using Uvicorn, pointing to the new entrypoint
CMD ["uvicorn", "api.index:app", "--host", "0.0.0.0", "--port", "8000"]