FROM python:3.13
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
COPY langchain_setup.py .
EXPOSE 8000
ENV PORT 8000            
CMD ["gunicorn", "--workers=4", "--bind=0.0.0.0:8000", "app:app"]

