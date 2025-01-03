FROM python:3.9-slim

WORKDIR ./app

COPY app/app.py /app/app.py

RUN pip install streamlit

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.enableCORS=false"]


