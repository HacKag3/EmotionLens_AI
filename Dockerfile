FROM python:3.10-slim

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y && rm -rf /var/lib/apt/lists/*

# Crea ed usa la directory di lavoro specificata
WORKDIR /app

# Copia delle dipendenze del progetto nel container
COPY requirements.txt .
# Intallazione delle dipendenze
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copia di tutto il progetto
COPY . .

ENV PYTHONPATH=/app
CMD ["python", "-u", "src.fer2013.py"]