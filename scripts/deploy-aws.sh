#!/bin/bash

declare -r SCRIPT_DIRECTORY="$(dirname $(realpath ${BASH_SOURCE[0]}))"
declare -r PROJECT_ROOT="$(dirname ${SCRIPT_DIRECTORY})"

cd "${PROJECT_ROOT}" \
    && . ./.env \
    && . ./config/locals

if [ -z "${AWS_ECR_REGISTRY_URL}" ]; then
    echo "AWS_ECR_REGISTRY_URL variable in 'config/locals' is empty or unset." \
         " Fill in the values in 'config/locals' and rerun this file."
    exit 1
fi

declare -r IMAGE_NAME="softlearning"
declare -r IMAGE_TAG="${SOFTLEARNING_DEV_TAG}"
declare -r TARGET_REGISTRY="${AWS_ECR_REGISTRY_URL}"

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

build_docker_image() {

    echo "Building Docker image."

    docker-compose \
        -f ./docker/docker-compose.dev.cpu.yml \
        build \
        --build-arg MJKEY="$(cat ~/.mujoco/mjkey.txt)"

    echo "Build successful."

}

push_image_to_aws_ecr() {

    SOURCE_IMAGE="${IMAGE_NAME}:${IMAGE_TAG}"
    TARGET_IMAGE="${TARGET_REGISTRY}/${SOURCE_IMAGE}"

    echo "${SOURCE_IMAGE}"
    echo "${TARGET_IMAGE}"

    $(aws ecr get-login --no-include-email)

    docker tag "${SOURCE_IMAGE}" "${TARGET_IMAGE}"
    docker push "${TARGET_IMAGE}"

}

main() {

    build_docker_image
    push_image_to_aws_ecr

}

main
