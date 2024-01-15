# Start from the latest Ubuntu image
FROM ubuntu:latest

# Set non-interactive mode for apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Install common software and build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libncurses5-dev \
    libncursesw5-dev \
    libreadline-dev \
    libsqlite3-dev \
    libgdbm-dev \
    libdb5.3-dev \
    libbz2-dev \
    libexpat1-dev \
    liblzma-dev \
    libffi7 \
    libffi-dev \
    wget \
    curl

# Download and install Python 3.6 from source
WORKDIR /tmp
RUN wget https://www.python.org/ftp/python/3.6.15/Python-3.6.15.tgz \
    && tar xvf Python-3.6.15.tgz \
    && cd Python-3.6.15 \
    && ./configure --enable-optimizations \
    && make -j$(nproc) \
    && make altinstall
RUN wget https://www.python.org/ftp/python/3.7.12/Python-3.7.12.tgz \
    && tar xvf Python-3.7.12.tgz \
    && cd Python-3.7.12 \
    && ./configure --enable-optimizations \
    && make -j$(nproc) \
    && make altinstall

RUN wget https://www.python.org/ftp/python/3.8.12/Python-3.8.12.tgz \
    && tar xvf Python-3.8.12.tgz \
    && cd Python-3.8.12 \
    && ./configure --enable-optimizations \
    && make -j$(nproc) \
    && make altinstall
RUN apt-get install -y \
    python3.10 python3.10-venv python3.10-dev
    

# Install pip for Python 3.10
RUN curl https://bootstrap.pypa.io/get-pip.py | python3.10

# Copy and install requirements.txt using Python 3.10
COPY requirements.txt /tmp/
RUN pip3.10 install --no-cache-dir -r /tmp/requirements.txt

# Default command
CMD ["bash"]
