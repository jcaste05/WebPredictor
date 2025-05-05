FROM python:3.10-slim

# System dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . /app

# Install python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port where the container will listen
EXPOSE 8080

ENV PYTHONPATH=/app

RUN pytest

# Run the web application
CMD ["python", "./web/main.py"]
