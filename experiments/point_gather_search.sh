#!/usr/bin/env bash
source activate softlearning
cd ~/softlearning/
# TODO put in directory named by date
nohup softlearning run_example_local examples.development.unconstrained_hyper_param_search \
    --universe=gym \
    --domain=PointGather \
    --task=v0 \
    --exp-name=sac_point_gather_unconstrained_hyper_param_search \
    --num-samples=1 \
    --trial-cpus=1  \
    --checkpoint-frequency=1000 > sac_point_gather_unconstrained_grid_search.out
