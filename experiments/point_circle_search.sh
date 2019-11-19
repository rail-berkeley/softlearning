#!/usr/bin/env bash
source activate softlearning
cd ~/softlearning/
# TODO put in directory and .out file named by date
nohup softlearning run_example_local examples.development.unconstrained_hyper_param_search \
    --universe=gym \
    --domain=PointCircle \
    --task=v0 \
    --exp-name=sac_point_circle_unconstrained_hyper_param_search \
    --num-samples=1 \
    --trial-cpus=1  \
    --checkpoint-frequency=1000 > sac_point_circle_unconstrained_search.out
