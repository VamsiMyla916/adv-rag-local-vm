# Dockerfile

# 1. Start from an official Python 3.12 image.
FROM python:3.12-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy the requirements file first and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy the rest of your application code (main.py, data.txt)
COPY . .

# 5. Expose the port that FastAPI will run on
EXPOSE 8000

# 6. The command to run your application when the container starts
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]