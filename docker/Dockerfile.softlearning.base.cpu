# Base softlearning container that contains all softlearning requirements,
# but not the actual softlearning repo. Could be used for example when developing
# softlearning, in which case you would mount softlearning repo in to the container
# as a volume, and thus be able to modify code on the host, yet run things inside
# the container. You are encouraged to use docker-compose (docker-compose.dev.yml),
# which should allow you to setup your environment with a single one command.

ARG UBUNTU_VERSION=18.04

FROM ubuntu:${UBUNTU_VERSION} as base

ARG UBUNTU_VERSION

ARG MJKEY

SHELL ["/bin/bash", "-c"]

# MAINTAINER Kristian Hartikainen <kristian.hartikainen@gmail.com>

ENV DEBIAN_FRONTEND="noninteractive"
# See http://bugs.python.org/issue19846
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    git mercurial subversion

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    /bin/bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> /etc/bash.bashrc

RUN apt-get install -y curl grep sed dpkg && \
    TINI_VERSION=`curl https://github.com/krallin/tini/releases/latest | grep -o "/v.*\"" | sed 's:^..\(.*\).$:\1:'` && \
    curl -L "https://github.com/krallin/tini/releases/download/v${TINI_VERSION}/tini_${TINI_VERSION}.deb" > tini.deb && \
    dpkg -i tini.deb && \
    rm tini.deb && \
    apt-get clean \
    && rm -rf /var/lib/apt/lists/*


RUN conda update -y --name base conda \
    && conda clean --all -y


# ========== Softlearning dependencies ==========
RUN apt-get update -y \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        gnupg2 \
        make \
        cmake \
        ffmpeg \
        swig \
        libz-dev \
        unzip \
        zlib1g-dev \
        libglfw3 \
        libglfw3-dev \
        libxrandr2 \
        libxinerama-dev \
        libxi6 \
        libxcursor-dev \
        libgl1-mesa-dev \
        libgl1-mesa-glx \
        libglew-dev \
        libosmesa6-dev \
        lsb-release \
        ack-grep \
        patchelf \
        vim \
        emacs \
        wget \
        xpra \
        xserver-xorg-dev \
        xvfb \
    && export CLOUD_SDK_REPO="cloud-sdk-$(lsb_release -c -s)" \
    && echo "deb http://packages.cloud.google.com/apt $CLOUD_SDK_REPO main" \
            | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list \
    && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg \
            | apt-key add - \
    && apt-get update -y \
    && apt-get install -y google-cloud-sdk \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


# ========= MuJoCo ===============
COPY ./scripts/install_mujoco.py /tmp/

RUN /tmp/install_mujoco.py --mujoco-path=/root/.mujoco --versions 1.50 2.00 \
    && ln -s /root/.mujoco/mujoco200_linux /root/.mujoco/mujoco200 \
    && rm /tmp/install_mujoco.py

ENV LD_LIBRARY_PATH /root/.mujoco/mjpro150/bin:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH /root/.mujoco/mujoco200/bin:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH /root/.mujoco/mujoco200_linux/bin:${LD_LIBRARY_PATH}

# This is a hack required to make mujocopy to compile in gpu mode
RUN mkdir -p /usr/lib/nvidia-000
ENV LD_LIBRARY_PATH ${LD_LIBRARY_PATH}:/usr/lib/nvidia-000


# ========== Conda Environment ==========
COPY ./environment.yml /tmp/
COPY ./requirements.txt /tmp/

# NOTE: Don't separate the Mujoco key echo and remove commands into separate
# run commands! Otherwise your key will be readable by anyone who has access
# To the container. We need the key in order to compile mujoco_py.
RUN echo "${MJKEY}" > ~/.mujoco/mjkey.txt \
    && conda env update -f /tmp/environment.yml \
    && conda clean --all -y \
    && rm ~/.mujoco/mjkey.txt

RUN echo "conda activate softlearning" >> ~/.bashrc \
    && echo "cd ~/softlearning" >> ~/.bashrc


# =========== Container Entrypoint =============
COPY ./docker/entrypoint.sh /entrypoint.sh
ENTRYPOINT ["/usr/bin/tini", "--", "/entrypoint.sh"]
