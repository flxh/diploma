FROM python:3.7-slim-buster

WORKDIR /app

COPY requirements.txt .
COPY src ./src
COPY loadprofiles_1min.mat .
COPY ihm-daten_20252.csv .
COPY eval_episode.pkl .

RUN python -m pip install -r requirements.txt

ENV PYTHONPATH "${PYTHONPATH}:/app/src"

WORKDIR /app/src/ml

ENTRYPOINT ["python", "ppo_training.py"]