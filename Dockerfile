# Use official Python image
FROM python:3.11.3

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project
COPY . .

RUN ls -lh

# Expose the port
EXPOSE 5000

# Run the Flask app
CMD ["python", "app.py"]
