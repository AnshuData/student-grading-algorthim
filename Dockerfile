#FROM rust:1.67 as builder
#WORKDIR /usr/src/myapp

#COPY . .
#RUN cargo install --path .

FROM python:3.9.16-slim-bullseye
ENV PYTHONPATH "${PYTHONPATH}:/app/"
WORKDIR /app
RUN apt-get update

RUN apt-get install -y \
    git \
    libenchant-2-dev
    #build-essential \
    #curl \
    #ffmpeg

COPY ./requirements.txt .

# install the packages from the requirements.txt in the container
RUN pip install -r requirements.txt
COPY . .


# expose the port that uvicorn will run the app on
ENV PORT=80
EXPOSE 80

CMD ["python","gram_check.py"]
