import datetime


def datetimestamp(divider='-', datetime_divider='T'):
    now = datetime.datetime.now()
    return now.strftime(
        '%Y{d}%m{d}%dT%H{d}%M{d}%S'
        ''.format(d=divider, dtd=datetime_divider))


def datestamp(divider='-'):
    return datetime.date.today().isoformat().replace('-', divider)


def timestamp(divider='-'):
    now = datetime.datetime.now()
    time_now = datetime.datetime.time(now)
    return time_now.strftime(
        '%H{d}%M{d}%S'.format(d=divider))
