FROM nvidia/cuda:12.3.2-base-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
    git \
    python3-pip \
    python3-dev \
    python3-opencv \
    libglib2.0-0 \
    make 

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip
RUN pip install .

COPY Makefile /app

CMD ["make", "run_train"]

