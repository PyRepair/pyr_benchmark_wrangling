# Start from the latest Ubuntu image
FROM ubuntu:22.04 as lite

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
    git \
    curl \
    ca-certificates \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Use multi-stage builds to compile Python and remove the build dependencies afterwards


# Download and install Python 3.6 from source
WORKDIR /tmp
# Download and install Python versions from source
RUN wget https://www.python.org/ftp/python/3.6.15/Python-3.6.15.tgz \
    && tar xvf Python-3.6.15.tgz \
    && cd Python-3.6.15 \
    && ./configure --enable-optimizations \
    && make -j$(nproc) \
    && make altinstall \
    && cd .. && rm -rf Python-3.6.15* \
    # Repeat the process for other Python versions
    && wget https://www.python.org/ftp/python/3.7.12/Python-3.7.12.tgz \
    && tar xvf Python-3.7.12.tgz \
    && cd Python-3.7.12 \
    && ./configure --enable-optimizations \
    && make -j$(nproc) \
    && make altinstall \
    && cd .. && rm -rf Python-3.7.12* \
    && wget https://www.python.org/ftp/python/3.8.12/Python-3.8.12.tgz \
    && tar xvf Python-3.8.12.tgz \
    && cd Python-3.8.12 \
    && ./configure --enable-optimizations \
    && make -j$(nproc) \
    && make altinstall \
    && cd .. && rm -rf Python-3.8.12*


RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3.10-dev \
    && rm -rf /var/lib/apt/lists/*

# Install pip for Python 3.10
RUN wget -O get-pip.py 'https://bootstrap.pypa.io/get-pip.py' \
    && python3.10 get-pip.py \
    && rm get-pip.py


# Copy and install requirements.txt using Python 3.10
COPY . /tmp/
RUN pip install .



RUN bgp update_bug_records && bgp clone 

From lite as full

COPY --from=lite /usr/ /usr/
COPY --from=lite /root/.abw/ /root/.abw/
COPY --from=lite /tmp/ /tmp/


# This will build all one time buildable envs
RUN bgp prep --repo_list ansible,black,cookiecutter,fastapi,httpie,keras,luigi,matplotlib,PySnooper,sanic,scrapy,thefuck,tornado,tqdm,youtube-dl \
    && bgp prep --bugids pandas:1,pandas:7


# Default command
CMD ["bash"]
