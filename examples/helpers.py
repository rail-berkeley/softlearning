import argparse


def str2bool(value):
    if value.lower() in ('yes', 'y', 'true', 't', '1'):
        return True
    elif value.lower() in ('no', 'n', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError(
            'Expected boolean value. Got {value}, {type(value)}'
            ''.format(value))
