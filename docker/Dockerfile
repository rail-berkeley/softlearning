FROM continuumio/anaconda3

MAINTAINER Kristian Hartikainen <kristian.hartikainen@gmail.com>

# ========== Special Deps ==========
RUN export DEBIAN_FRONTEND=noninteractive \
    && apt update \
    && apt upgrade -yq \
    && apt -yq install \
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
RUN conda env update -f /tmp/environment.yml \
    && rm /tmp/requirements.txt \
    && rm /tmp/environment.yml

ENV PYTHONPATH /root/softlearning:$PYTHONPATH
ENV PATH /opt/conda/envs/softlearning/bin:$PATH
RUN echo "source activate softlearning" >> /root/.bashrc
ENV BASH_ENV /root/.bashrc

# ========= MuJoCo ===============
ARG MJKEY

# Mujoco for gym and mujoco_py
ENV MUJOCO_VERSION=150 \
    MUJOCO_PATH=/root/.mujoco

RUN MUJOCO_ZIP="mjpro${MUJOCO_VERSION}_linux.zip" \
    && mkdir -p ${MUJOCO_PATH} \
    && wget -P ${MUJOCO_PATH} https://www.roboti.us/download/${MUJOCO_ZIP} \
    && unzip ${MUJOCO_PATH}/${MUJOCO_ZIP} -d ${MUJOCO_PATH} \
    && rm ${MUJOCO_PATH}/${MUJOCO_ZIP}

ENV LD_LIBRARY_PATH /root/.mujoco/mjpro${MUJOCO_VERSION}/bin:${LD_LIBRARY_PATH}

# Trigger mujoco_py compilation using the MJKEY provided. Delete MJKEY afterwards.
RUN echo "${MJKEY}" > /root/.mujoco/mjkey.txt \
    && bash -c "python -c 'import mujoco_py'" \
    && rm /root/.mujoco/mjkey.txt


COPY ./docker/entrypoint.sh /entrypoint.sh

ENV DISPLAY=:0

ENTRYPOINT ["/entrypoint.sh"]
