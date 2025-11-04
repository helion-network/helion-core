FROM nvcr.io/nvidia/cuda:11.8.0-devel-ubuntu22.04
LABEL maintainer="bigscience-workshop"
LABEL repository="helion"

WORKDIR /home
# Set en_US.UTF-8 locale by default
RUN echo "LC_ALL=en_US.UTF-8" >> /etc/environment

# Install packages
RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential \
  wget \
  git \
  && apt-get clean autoclean && rm -rf /var/lib/apt/lists/{apt,dpkg,cache,log} /tmp/* /var/tmp/*

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O install_miniconda.sh && \
  bash install_miniconda.sh -b -p /opt/conda && rm install_miniconda.sh
ENV PATH="/opt/conda/bin:${PATH}"

RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
  && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

RUN conda install -y python~=3.10.12 pip && \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu118 torch==2.1.2 && \
    pip install --no-cache-dir bitsandbytes==0.41.3 && \
    conda clean --all && rm -rf ~/.cache/pip

VOLUME /cache
ENV PETALS_CACHE=/cache

COPY . helion/
RUN pip install --no-cache-dir -e helion

WORKDIR /home/helion/
CMD bash
