import requests


def instance_preempted():
    try:
        response = requests.get(
            "http://metadata/computeMetadata/v1/instance/preempted",
            headers={'Metadata-Flavor': 'Google'}
        )
        preempted = (response.status_code == 200
                     and response.text != 'FALSE')
    except Exception:
        preempted = False

    return preempted
