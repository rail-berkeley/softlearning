FROM continuumio/anaconda3

MAINTAINER Kristian Hartikainen <kristian.hartikainen@gmail.com>


# ========== Special Deps ==========
RUN apt update \
    && apt upgrade -y \
    && apt -y install \
        build-essential \
        curl \
        git \
        make \
        cmake \
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
        ack-grep \
        vim \
        emacs \
        wget \
        xpra \
        xserver-xorg-dev \
        xvfb \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*


# ========== Add codebase stub ==========
WORKDIR /root/softlearning

COPY ./environment.yml /tmp/
COPY ./requirements.txt /tmp/
RUN conda update -y --name base conda \
    && conda env update -f /tmp/environment.yml \
    && rm /tmp/requirements.txt \
    && rm /tmp/environment.yml

ENV PYTHONPATH /root/softlearning:$PYTHONPATH
ENV PATH /opt/conda/envs/softlearning/bin:$PATH
RUN echo "source activate softlearning" >> /root/.bashrc
ENV BASH_ENV /root/.bashrc


# ========= rllab ===============
# We need to clone rllab repo in order to use the
# `rllab.sandbox.rocky.tf` functions.

ENV RLLAB_PATH=/root/rllab \
    RLLAB_VERSION=b3a28992eca103cab3cb58363dd7a4bb07f250a0

RUN git clone https://github.com/rll/rllab.git ${RLLAB_PATH} \
    && cd ${RLLAB_PATH} \
    && git checkout ${RLLAB_VERSION} \
    && mkdir ${RLLAB_PATH}/vendor/mujoco \
    && python -m rllab.config

ENV PYTHONPATH ${RLLAB_PATH}:${PYTHONPATH}


# ========= MuJoCo ===============
ENV MUJOCO_VERSION=150 \
    MUJOCO_PATH=/root/.mujoco

RUN MUJOCO_ZIP="mjpro${MUJOCO_VERSION}_linux.zip" \
    && mkdir -p ${MUJOCO_PATH} \
    && wget -P ${MUJOCO_PATH} https://www.roboti.us/download/${MUJOCO_ZIP} \
    && unzip ${MUJOCO_PATH}/${MUJOCO_ZIP} -d ${MUJOCO_PATH} \
    && cp ${MUJOCO_PATH}/mjpro${MUJOCO_VERSION}/bin/libmujoco${MUJOCO_VERSION}.so ${RLLAB_PATH}/vendor/mujoco/ \
    && cp ${MUJOCO_PATH}/mjpro${MUJOCO_VERSION}/bin/libglfw.so.3 ${RLLAB_PATH}/vendor/mujoco/ \
    && rm ${MUJOCO_PATH}/${MUJOCO_ZIP}

ENV LD_LIBRARY_PATH /root/.mujoco/mjpro${MUJOCO_VERSION}/bin:${LD_LIBRARY_PATH}


# We need to add /root/code/rllab to PYTHONPATH in order for rllab to find its stuff
ENV  PYTHONPATH /root/code/rllab:${PYTHONPATH}

COPY ./docker/entrypoint.sh /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
