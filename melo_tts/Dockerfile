# 기본 이미지 설정
# AWS 설정
FROM amazon/aws-cli:2.7.0 as aws_downloader
# 환경변수로부터 받은 값을 사용하여 설정
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ARG S3_MODEL_BUCKET
ARG S3_MODEL_ID
ARG S3_MODEL_KEY
RUN aws configure set aws_access_key_id ${S3_MODEL_ID} && \
    aws configure set aws_secret_access_key ${S3_MODEL_KEY}
RUN mkdir -p /model
RUN aws s3 cp s3://${S3_MODEL_BUCKET}/tts/melo_tts_model/model/ /model --recursive
# RUN aws s3 cp s3://${S3_MODEL_BUCKET}/nlp/chunking/ /app/model --recursive


# Python 환경 설정
FROM python:3.9-slim as base
WORKDIR /app
COPY --from=aws_downloader /model /model
COPY . .

RUN pip install --upgrade pip
RUN pip install unidic
RUN python -m unidic download
RUN pip install -r requirements.txt
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install botocore


HEALTHCHECK CMD curl -f http://localhost/v1/tts/en-us/docs || exit 1
     
ENV MODEL_FOLDER_PATH=/model
ENV PYTHONPATH "${PYTHONPATH}:/app/src"
ENTRYPOINT ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "80"]

