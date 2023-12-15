# syntax=docker/dockerfile:1
FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:2.13.0-cpu-py310-ubuntu20.04-sagemaker
WORKDIR /src
COPY . .
RUN pip install -r requirements.txt
CMD ["python", "run.py"]