#!/usr/bin/env bash
source activate softlearning
cd ..
nohup softlearning run_example_local examples.development.unconstrained_grid_search \
    --universe=gym \
    --domain=PointGather \
    --task=v0 \
    --exp-name=sac_point_gather_unconstrained_search \
    --num-samples=1 \
    --trial-cpus=2  \
    --checkpoint-frequency=1000 > sac_point_gather_unconstrained.out
