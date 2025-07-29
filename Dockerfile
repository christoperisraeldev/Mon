# Use official lightweight Python 3.11 image
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements.txt first for caching
COPY requirements.txt ./

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy rest of the application code
COPY . .

# Expose port 8080 (Cloud Run expects this)
EXPOSE 8080

# Start the Flask app with gunicorn on port 8080
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8080"]
