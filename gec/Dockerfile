# CUDA와 CUDNN이 포함된 베이스 이미지 사용
FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04 as base

# Python 3.10.5 slim 이미지를 사용하여 requirements.txt 생성
FROM python:3.11.3-slim as requirements_exporter
WORKDIR /app
RUN pip install --no-cache-dir poetry
COPY pyproject.toml poetry.lock ./
RUN poetry export -f requirements.txt --without-hashes --output requirements.txt
RUN poetry export -f requirements.txt --without-hashes --only test --output requirements-test.txt

# AWS CLI 이미지를 사용하여 모델 다운로드
FROM amazon/aws-cli:2.7.0 as aws_downloader
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ARG S3_MODEL_BUCKET
ARG S3_MODEL_ID
ARG S3_MODEL_KEY
RUN aws configure set aws_access_key_id ${S3_MODEL_ID} && \
    aws configure set aws_secret_access_key ${S3_MODEL_KEY}
RUN mkdir -p /model
RUN aws s3 cp s3://${S3_MODEL_BUCKET}/gec_coedit/ /model --recursive

# Python 3.10.5 slim 이미지를 기반으로 설정
FROM python:3.10.5-slim as python-base
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    libpcre3-dev \
    locales \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=aws_downloader /model /model
COPY --from=requirements_exporter /app/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    python -m spacy download en_core_web_sm && \
    python -m spacy link --force en_core_web_sm en
RUN python -c "import stanza; stanza.download('en')"
RUN python -c "import stanza; nlp = stanza.Pipeline('en')"

COPY resources resources
COPY src src

# 테스트용 이미지
FROM python-base as test
COPY --from=requirements_exporter /app/requirements-test.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-test.txt
COPY tests tests
COPY pyproject.toml ./

COPY . .

CMD ["pytest", "--env-file", ".env", "tests"]

# 최종 프로덕션 이미지
FROM python-base as prod

RUN pip install transformers==4.27.4
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
EXPOSE 80

HEALTHCHECK CMD curl -f http://localhost/v1/gec/docs || exit 1

ENTRYPOINT ["uvicorn", "--host", "0.0.0.0", "--port", "80", "src.api:app"]
# ENTRYPOINT ["gunicorn", "src.api:app", "-c", "src/gunicorn.conf.py"]
