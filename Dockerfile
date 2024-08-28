FROM python:3.11
WORKDIR /app
COPY requirements.txt /app/
RUN pip install -r requirements.txt
COPY . /app
EXPOSE 7000 8080
CMD ["mkdir","iris_model"]
CMD ["python", "index.py"]