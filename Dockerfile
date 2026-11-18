# Reproducible analysis image. Not yet built or tested: build it after the lock file is regenerated with torch.
FROM python:3.12-slim
WORKDIR /study
COPY requirements-lock.txt .
RUN pip install --no-cache-dir -r requirements-lock.txt
COPY . .
RUN pip install --no-cache-dir --no-deps -e .
CMD ["python", "scripts/check_environment.py"]
