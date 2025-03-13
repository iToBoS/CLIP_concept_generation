FROM python:3.10

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt 

COPY . /app

CMD ["python", "inference.py", "data"]  