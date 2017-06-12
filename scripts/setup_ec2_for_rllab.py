import boto3
import re
import sys
import json
import botocore
import os
from rllab.misc import console
from rllab import config
from string import Template
from collections import OrderedDict

ACCESS_KEY = os.environ["AWS_ACCESS_KEY"]
ACCESS_SECRET = os.environ["AWS_ACCESS_SECRET"]
S3_BUCKET_NAME = os.environ["RLLAB_S3_BUCKET"]
PREFIX = os.environ.get("RLLAB_PREFIX", "")

SECURITY_GROUP_NAME = PREFIX + "rllab-sg"
INSTANCE_PROFILE_NAME = PREFIX + "rllab"
INSTANCE_ROLE_NAME = PREFIX + "rllab"

ALL_REGION_AWS_SECURITY_GROUP_IDS = {}
ALL_REGION_AWS_KEY_NAMES = {}

ALL_SUBNET_INFO = {}

REGIONS = [
   "ap-northeast-1",
   "ap-northeast-2",
   "ap-south-1",
   "ap-southeast-1",
   "ap-southeast-2",
   "eu-central-1",
   "eu-west-1",
   "sa-east-1",
   "us-east-1",
   "us-east-2",
   "us-west-1",
   "us-west-2",
]

CONFIG_TEMPLATE = Template("""
import os.path as osp
import os

USE_GPU = False

USE_TF = True

AWS_REGION_NAME = "us-west-1"

if USE_GPU:
    DOCKER_IMAGE = "dementrock/rllab3-shared-gpu-cuda80"
else:
    DOCKER_IMAGE = "dementrock/rllab3-shared"

DOCKER_LOG_DIR = "/tmp/expt"

AWS_S3_PATH = "s3://$s3_bucket_name/rllab/experiments"

AWS_CODE_SYNC_S3_PATH = "s3://$s3_bucket_name/rllab/code"

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

ALL_SUBNET_INFO = $all_subnet_info

INSTANCE_TYPE_INFO = {  #this prices are orientative.
    "c4.large": dict(price=0.105, vCPU=2),
    "c4.xlarge": dict(price=0.209, vCPU=4),
    "c4.2xlarge": dict(price=0.419, vCPU=8),
    "c4.4xlarge": dict(price=0.838, vCPU=16),
    "c4.8xlarge": dict(price=1.00, vCPU=36),
    "m4.large": dict(price=0.1, vCPU=2),
    "m4.xlarge": dict(price=0.5, vCPU=4),
    "m4.2xlarge": dict(price=0.5, vCPU=8),
    "m4.4xlarge": dict(price=0.8, vCPU=16),
    "m4.10xlarge": dict(price=2.394, vCPU=40),
    "m4.16xlarge": dict(price=1.5, vCPU=64),
    "g2.2xlarge": dict(price=0.65, vCPU=8),
}

AWS_IMAGE_ID = ALL_REGION_AWS_IMAGE_IDS[AWS_REGION_NAME]

if USE_GPU:
    AWS_INSTANCE_TYPE = "g2.2xlarge"
else:
    AWS_INSTANCE_TYPE = "c4.2xlarge"

ALL_REGION_AWS_KEY_NAMES = $all_region_aws_key_names

AWS_KEY_NAME = ALL_REGION_AWS_KEY_NAMES[AWS_REGION_NAME]

AWS_SPOT = True

AWS_SPOT_PRICE = '0.5'

AWS_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY", None)

AWS_ACCESS_SECRET = os.environ.get("AWS_ACCESS_SECRET", None)

AWS_IAM_INSTANCE_PROFILE_NAME = "$instance_profile_name"

AWS_SECURITY_GROUPS = ["$security_group_name"]

ALL_REGION_AWS_SECURITY_GROUP_IDS = $all_region_aws_security_group_ids

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

""")


def setup_iam():
    iam_client = boto3.client(
        "iam",
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=ACCESS_SECRET,
    )
    iam = boto3.resource('iam', aws_access_key_id=ACCESS_KEY, aws_secret_access_key=ACCESS_SECRET)

    # delete existing role if it exists
    try:
        existing_role = iam.Role(INSTANCE_ROLE_NAME)
        existing_role.load()
        # if role exists, delete and recreate
        response = query_yes_no(
            "There is an existing role named %s. Proceed to delete everything rllab-related and recreate?" %
            INSTANCE_ROLE_NAME,
            default="no", allow_skip=True)
        if response == "skip":
            return
        elif not response:
            sys.exit()
        else:
            pass
        print("Listing instance profiles...")
        inst_profiles = existing_role.instance_profiles.all()
        for prof in inst_profiles:
            for role in prof.roles:
                print("Removing role %s from instance profile %s" % (role.name, prof.name))
                prof.remove_role(RoleName=role.name)
            print("Deleting instance profile %s" % prof.name)
            prof.delete()
        for policy in existing_role.policies.all():
            print("Deleting inline policy %s" % policy.name)
            policy.delete()
        for policy in existing_role.attached_policies.all():
            print("Detaching policy %s" % policy.arn)
            existing_role.detach_policy(PolicyArn=policy.arn)
        print("Deleting role")
        existing_role.delete()
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchEntity':
            pass
        else:
            raise e

    print("Creating role %s " % INSTANCE_ROLE_NAME)
    iam_client.create_role(
        Path='/',
        RoleName=INSTANCE_ROLE_NAME,
        AssumeRolePolicyDocument=json.dumps({'Version': '2012-10-17', 'Statement': [
            {'Action': 'sts:AssumeRole', 'Effect': 'Allow', 'Principal': {'Service': 'ec2.amazonaws.com'}}]})
    )

    role = iam.Role(INSTANCE_ROLE_NAME)
    print("Attaching policies")
    role.attach_policy(PolicyArn='arn:aws:iam::aws:policy/AmazonS3FullAccess')
    role.attach_policy(PolicyArn='arn:aws:iam::aws:policy/ResourceGroupsandTagEditorFullAccess')

    print("Creating inline policies")
    iam_client.put_role_policy(
        RoleName=role.name,
        PolicyName='CreateTags',
        PolicyDocument=json.dumps({
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": ["ec2:CreateTags"],
                    "Resource": ["*"]
                }
            ]
        })
    )
    iam_client.put_role_policy(
        RoleName=role.name,
        PolicyName='TerminateInstances',
        PolicyDocument=json.dumps({
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Sid": "Stmt1458019101000",
                    "Effect": "Allow",
                    "Action": [
                        "ec2:TerminateInstances"
                    ],
                    "Resource": [
                        "*"
                    ]
                }
            ]
        })
    )

    print("Creating instance profile %s" % INSTANCE_PROFILE_NAME)
    iam_client.create_instance_profile(
        InstanceProfileName=INSTANCE_PROFILE_NAME,
        Path='/'
    )
    print("Adding role %s to instance profile %s" % (INSTANCE_ROLE_NAME, INSTANCE_PROFILE_NAME))
    iam_client.add_role_to_instance_profile(
        InstanceProfileName=INSTANCE_PROFILE_NAME,
        RoleName=INSTANCE_ROLE_NAME
    )


def setup_s3():
    print("Creating S3 bucket at s3://%s" % S3_BUCKET_NAME)
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=ACCESS_SECRET,
    )
    try:
        s3_client.create_bucket(
            ACL='private',
            Bucket=S3_BUCKET_NAME,
        )
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == 'BucketAlreadyExists':
            raise ValueError("Bucket %s already exists. Please reconfigure S3_BUCKET_NAME" % S3_BUCKET_NAME) from e
        elif e.response['Error']['Code'] == 'BucketAlreadyOwnedByYou':
            print("Bucket already created by you")
        else:
            raise e
    print("S3 bucket created")


def setup_ec2():
    for region in REGIONS:
        print("Setting up region %s" % region)

        ec2 = boto3.resource(
            "ec2",
            region_name=region,
            aws_access_key_id=ACCESS_KEY,
            aws_secret_access_key=ACCESS_SECRET,
        )
        ec2_client = boto3.client(
            "ec2",
            region_name=region,
            aws_access_key_id=ACCESS_KEY,
            aws_secret_access_key=ACCESS_SECRET,
        )
        existing_vpcs = list(ec2.vpcs.all())
        assert len(existing_vpcs) >= 1
        vpc = existing_vpcs[0]
        print("Creating security group in VPC %s" % str(vpc.id))
        try:
            security_group = vpc.create_security_group(
                GroupName=SECURITY_GROUP_NAME, Description='Security group for rllab'
            )
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == 'InvalidGroup.Duplicate':
                sgs = list(vpc.security_groups.filter(GroupNames=[SECURITY_GROUP_NAME]))
                security_group = sgs[0]
            else:
                raise e

        ALL_REGION_AWS_SECURITY_GROUP_IDS[region] = [security_group.id]

        ec2_client.create_tags(Resources=[security_group.id], Tags=[{'Key': 'Name', 'Value': SECURITY_GROUP_NAME}])
        try:
            security_group.authorize_ingress(FromPort=22, ToPort=22, IpProtocol='tcp', CidrIp='0.0.0.0/0')
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == 'InvalidPermission.Duplicate':
                pass
            else:
                raise e
        print("Security group created with id %s" % str(security_group.id))

        key_name = PREFIX + ('rllab-%s' % region)
        try:
            print("Trying to create key pair with name %s" % key_name)
            key_pair = ec2_client.create_key_pair(KeyName=key_name)
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == 'InvalidKeyPair.Duplicate':
                if not query_yes_no("Key pair with name %s exists. Proceed to delete and recreate?" % key_name, "no"):
                    sys.exit()
                print("Deleting existing key pair with name %s" % key_name)
                ec2_client.delete_key_pair(KeyName=key_name)
                print("Recreating key pair with name %s" % key_name)
                key_pair = ec2_client.create_key_pair(KeyName=key_name)
            else:
                raise e

        key_pair_folder_path = os.path.join(config.PROJECT_PATH, "private", "key_pairs")
        file_name = os.path.join(key_pair_folder_path, "%s.pem" % key_name)

        print("Saving keypair file")
        console.mkdir_p(key_pair_folder_path)
        with os.fdopen(os.open(file_name, os.O_WRONLY | os.O_CREAT, 0o600), 'w') as handle:
            handle.write(key_pair['KeyMaterial'] + '\n')

        # adding pem file to ssh
        # os.system("ssh-add %s" % file_name)

        ALL_REGION_AWS_KEY_NAMES[region] = key_name
        print(ALL_REGION_AWS_KEY_NAMES)
        print(ALL_REGION_AWS_SECURITY_GROUP_IDS)

    subnets_info = get_subnets_info(REGIONS)  # this could be done at the same time than the above, keep it here for now
    for key, value in subnets_info.items():
        ALL_SUBNET_INFO[key] = value


def get_subnets_info(regions):
    clients = []
    for region in regions:
        client = boto3.client(
            "ec2",
            region_name=region,
            aws_access_key_id=config.AWS_ACCESS_KEY,
            aws_secret_access_key=config.AWS_ACCESS_SECRET,
        )
        client.region = region
        clients.append(client)
    subnet_info = OrderedDict()
    for client in clients:
        # first find the group
        security_group = client.describe_security_groups()['SecurityGroups'][0]['GroupId']
        subnets = client.describe_subnets()['Subnets']
        for subnet in subnets:
            subnet_info[subnet['AvailabilityZone']] = dict(SubnetID=subnet['SubnetId'], Groups=security_group)
    return subnet_info


def write_config():
    print("Writing config file...")
    content = CONFIG_TEMPLATE.substitute(
        all_region_aws_key_names=json.dumps(ALL_REGION_AWS_KEY_NAMES, indent=4),
        all_subnet_info=json.dumps(ALL_SUBNET_INFO, indent=4),  # CF
        all_region_aws_security_group_ids=json.dumps(ALL_REGION_AWS_SECURITY_GROUP_IDS, indent=4),
        s3_bucket_name=S3_BUCKET_NAME,
        security_group_name=SECURITY_GROUP_NAME,
        instance_profile_name=INSTANCE_PROFILE_NAME,
        instance_role_name=INSTANCE_ROLE_NAME,
    )

    config_personal_file = os.path.join(config.PROJECT_PATH, "rllab/config_personal.py")
    if os.path.exists(config_personal_file):
        if not query_yes_no("rllab/config_personal.py exists. Override?", "no"):
            sys.exit()
    with open(config_personal_file, "wb") as f:
        f.write(content.encode("utf-8"))


def setup():
    print("Using prefix: %s" % PREFIX)
    setup_s3()
    setup_iam()
    setup_ec2()
    write_config()


def query_yes_no(question, default="yes", allow_skip=False):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if allow_skip:
        valid["skip"] = "skip"
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)
    if allow_skip:
        prompt += " or skip"
    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")


if __name__ == "__main__":
    setup()
    # setup_ec2()
