cloud-build-local \
    --config=./docker/cloudbuild.yaml \
    --dryrun=false \
    --push \
    --write-workspace=/tmp/workspace \
    --substitutions=REPO_NAME="softlearning",BRANCH_NAME="$(git rev-parse --abbrev-ref HEAD)",COMMIT_SHA="$(git rev-parse HEAD)",SHORT_SHA="$(git rev-parse --short HEAD)" \
    .
