# WIP

# Dockerfile that clones the softlearning repo into the softlearning base
# image. Should be used for running stuff on the cloud, e.g. with ray.

# Base container to clone the softlearning-private repo
FROM ubuntu:18.04 as git_cloner
# Note that the SSH_PRIVATE_KEY arg is NOT saved on the final container

# add credentials on build
ARG SSH_PRIVATE_KEY

# install git
RUN apt-get update \
    && apt-get install -y git \
    && mkdir /root/.ssh/ \
    && echo "${SSH_PRIVATE_KEY}" > /root/.ssh/id_rsa \
    && chmod 0600 /root/.ssh/id_rsa \
    && touch /root/.ssh/known_hosts \
    && ssh-keyscan github.com >> /root/.ssh/known_hosts \
    && git clone git@github.com:rail-berkeley/softlearning.git /root/softlearning \
    && rm -vf /root/.ssh/id_rsa

# Base container to clone the sac_envs repo
FROM ubuntu:18.04 as sac_envs_cloner
# Note that the SSH_PRIVATE_KEY arg is NOT saved on the final container

# add credentials on build
ARG SSH_PRIVATE_KEY

# install git
RUN apt-get update \
    && apt-get install -y git \
    && mkdir /root/.ssh/ \
    && echo "${SSH_PRIVATE_KEY}" > /root/.ssh/id_rsa \
    && chmod 0600 /root/.ssh/id_rsa \
    && touch /root/.ssh/known_hosts \
    && ssh-keyscan github.com >> /root/.ssh/known_hosts \
    && git clone git@github.com:vikashplus/sac_envs.git /root/sac_envs \
    && rm -vf /root/.ssh/id_rsa

FROM softlearning-dev

# ========== Add codebase stub ==========
COPY --from=softlearning_cloner /root/softlearning /root/softlearning
COPY --from=sac_envs_cloner /root/sac_envs /root/sac_envs
WORKDIR /root/softlearning
