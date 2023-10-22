FROM ubuntu:20.04

# Packages versions
ENV CUDA_VERSION=10.2.89 \ 
    CUDA_PKG_VERSION=10-2=10.2.89-1 \
    NCCL_VERSION=2.5.6 \
    CUDNN_VERSION=7.6.5.32

# BASE
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends gnupg2 curl ca-certificates wget software-properties-common && \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub | apt-key add - && \
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda.list && \
    echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get purge --autoremove -y curl && \
    rm -rf /var/lib/apt/lists/*

# RUNTIME CUDA
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin &&\
    mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600 &&\
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub &&\
    add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /" &&\    
    apt-get update -y && \
    apt-get install -y --no-install-recommends cuda-cudart-${CUDA_PKG_VERSION} cuda-compat-10-2 \
                                               cuda-libraries-${CUDA_PKG_VERSION} cuda-nvtx-${CUDA_PKG_VERSION} libcublas10=10.2.2.89-1 \
                                               libnccl2=$NCCL_VERSION-1+cuda10.2 && \
    ln -s cuda-10.2 /usr/local/cuda && \
    apt-mark hold libnccl2 && \
    rm -rf /var/lib/apt/lists/*

# RUNTIME CUDNN7
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends libcudnn7=${CUDNN_VERSION}-1+cuda10.2 && \
    apt-mark hold libcudnn7 && \
    rm -rf /var/lib/apt/lists/* &&\
    echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
    echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf



ENV PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    NVIDIA_REQUIRE_CUDA="cuda>=10.2 brand=tesla,driver>=384,driver<385 brand=tesla,driver>=396,driver<397 brand=tesla,driver>=410,driver<411 brand=tesla,driver>=418,driver<419 brand=tesla,driver>=439,driver<441"

RUN add-apt-repository universe && apt-get update -y && apt-get install -y python3-pip python-is-python3 git libsndfile1 ffmpeg && rm -rf /var/lib/apt/lists/* 


ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 

RUN conda create -n my_env python=3.8 pip &&  echo "conda activate my_env" >> ~/.bashrc 
RUN conda install python=3.8 -y && conda clean -afy
RUN conda install --yes pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=10.2 -c pytorch && conda clean -afy
RUN pip install transformers==4.17.0 datasets==1.18.4 accelerate jiwer librosa jupyterlab  && rm -rf ~/.cache/pip 
# RUN pip install torch==1.12.1+cu102 torchvision==0.13.1+cu102 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu102 && rm -rf ~/.cache/pip
# # RUN pip install transformers==4.17.0 datasets==1.18.4 accelerate jiwer librosa jupyterlab  && rm -rf ~/.cache/pip 





CMD ["bash"]
