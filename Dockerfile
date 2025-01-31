# Use the official Python image as the base image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /root/workspace/services/generative-ai-buddy-rev

# Copy the requirements file into the container
COPY dependencies/requirements.txt .
COPY .env.dev .

ENV PYTHONPATH=/root/workspace/services/generative-ai-buddy-rev

# Install the dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Command to run the FastAPI application
CMD ["python", "scripts/run_API.py"]