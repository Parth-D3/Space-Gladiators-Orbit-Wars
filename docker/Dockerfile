FROM python:3.13-slim

RUN pip install --no-cache-dir \
    torch==2.11.0 \
    numpy==2.4.4 \
    kaggle-environments==1.30.1 \
    wandb==0.25.1

WORKDIR /workspace
COPY . .
ENTRYPOINT ["python", "train_ppo.py"]
