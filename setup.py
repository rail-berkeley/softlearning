from distutils.core import setup
from setuptools import find_packages


NAME = 'softlearning'
VERSION = '0.0.1'
DESCRIPTION = (
    "Softlearning is a deep reinforcement learning toolbox for training"
    " maximum entropy policies in continuous domains.")


setup(
    name=NAME,
    packages=find_packages(
        exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    version=VERSION,
    description=DESCRIPTION,
    long_description=open('./README.md').read(),
    author='Kristian Hartikainen',
    author_email='kristian.hartikainen@gmail.com',
    url='https://github.com/rail-berkeley/softlearning',
    keywords=(
        'softlearning',
        'soft-actor-critic',
        'sac',
        'soft-q-learning',
        'sql',
        'machine-learning',
        'reinforcement-learning',
        'deep-learning',
        'python',
    ),
    entry_points={
        'console_scripts': (
            'softlearning=softlearning.scripts.console_scripts:main',
        )
    },
    requires=(),
    zip_safe=True,
    license='MIT'
)
