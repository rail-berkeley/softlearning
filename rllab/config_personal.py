
import os.path as osp
import os

BUCKET = "tuomas"
#BUCKET = "hrtang0"

USE_GPU = False

USE_TF = True

AWS_REGION_NAME = "us-west-1"

if USE_GPU:
    DOCKER_IMAGE = "dementrock/rllab3-shared-gpu-cuda80"
else:
    DOCKER_IMAGE = "dementrock/rllab3-shared"

DOCKER_LOG_DIR = "/tmp/expt"

AWS_S3_PATH = "s3://tuomas/rllab/experiments"
#AWS_S3_PATH = "s3://hrtang0/rllab/experiments"

AWS_CODE_SYNC_S3_PATH = "s3://tuomas/rllab/code"
#AWS_CODE_SYNC_S3_PATH = "s3://hrtang0/rllab/code"

ALL_REGION_AWS_IMAGE_IDS = {
    "ap-northeast-1": "ami-c42689a5",
    "ap-northeast-2": "ami-865b8fe8",
    "ap-south-1": "ami-ea9feb85",
    "ap-southeast-1": "ami-c74aeaa4",
    "ap-southeast-2": "ami-0792ae64",
    "eu-central-1": "ami-f652a999",
    "eu-west-1": "ami-8c0a5dff",
    "sa-east-1": "ami-3f2cb053",
    "us-east-1": "ami-de5171c9",
    "us-east-2": "ami-e0481285",
    "us-west-1": "ami-efb5ff8f",
    "us-west-2": "ami-53903033",
}

AWS_IMAGE_ID = ALL_REGION_AWS_IMAGE_IDS[AWS_REGION_NAME]

if USE_GPU:
    AWS_INSTANCE_TYPE = "g2.2xlarge"
else:
    AWS_INSTANCE_TYPE = "c4.2xlarge"

ALL_REGION_AWS_KEY_NAMES = {
    "us-east-2": "rllab-us-east-2",
    "us-east-1": "rllab-us-east-1",
    "us-west-1": "rllab-us-west-1",
    "us-west-2": "rllab-us-west-2"
}

AWS_KEY_NAME = ALL_REGION_AWS_KEY_NAMES[AWS_REGION_NAME]

AWS_SPOT = True

AWS_SPOT_PRICE = '0.5'

AWS_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY", None)

AWS_ACCESS_SECRET = os.environ.get("AWS_ACCESS_SECRET", None)

AWS_IAM_INSTANCE_PROFILE_NAME = "rllab"

AWS_SECURITY_GROUPS = ["rllab-sg"]

ALL_REGION_AWS_SECURITY_GROUP_IDS = {
    "us-east-2": [
        "sg-b3d90cda"
    ],
    "us-east-1": [
        "sg-3d366447"
    ],
    "us-west-1": [
        "sg-70a29714"
    ],
    "us-west-2": [
        "sg-9f7fd7e6"
    ]
}

AWS_SECURITY_GROUP_IDS = ALL_REGION_AWS_SECURITY_GROUP_IDS[AWS_REGION_NAME]

FAST_CODE_SYNC_IGNORES = [
    ".git",
    "data",
    "src",
    ".idea",
    ".pods",
    "tests",
    "examples",
    "docs",
    ".idea",
    ".DS_Store",
    ".ipynb_checkpoints",
    "blackbox",
    "blackbox.zip",
    "*.pyc",
    "*.ipynb",
    "scratch-notebooks",
    "conopt_root",
    "private/key_pairs",
]

FAST_CODE_SYNC = True

# TF
TF_USE_GPU = False
TF_GPU_ALLOW_GROWTH = False
TF_GPU_MEM_FRAC = 0.9
TF_LOG_DEVICE_PLACEMENT = False

# MUJOCO
MUJOCO_KEY_PATH = osp.join(osp.dirname(__file__), "../vendor/mujoco")
