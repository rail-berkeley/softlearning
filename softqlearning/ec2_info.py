from rllab import config
instance_info = {
    "c4.large": dict(price=0.105, vCPU=2),
    "c4.xlarge": dict(price=0.209, vCPU=4),
    "c4.2xlarge": dict(price=0.419, vCPU=8),
    "c4.4xlarge": dict(price=0.838, vCPU=16),
    "m4.10xlarge": dict(price=2.394,vCPU=40),
    "c4.8xlarge": dict(price=1.675,vCPU=36),
    "g2.2xlarge": dict(price=0.65, vCPU=8),
    "p2.xlarge": dict(price=0.9, vCPU=4),
}

all_subnet_info = {
    'tuomas': {
        "us-west-1b": dict(
            SubnetID="subnet-3d90f159", Groups=["sg-7571c912"]),
        "us-west-2a": dict(
            SubnetID="subnet-39f52370", Groups=["sg-9985fae1"]),
    },
}
subnet_info = all_subnet_info[config.BUCKET]
