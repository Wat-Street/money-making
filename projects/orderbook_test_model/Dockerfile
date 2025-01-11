FROM python:3.9-slim

WORKDIR ./

COPY model.py .
COPY requirements.txt .

RUN pip install -r requirements.txt
CMD ["sleep", "infinity"]
