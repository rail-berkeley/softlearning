#!/usr/bin/env bash
source activate softlearning
cd ~/softlearning/

name=sac_point_gather_unconstrained_search
out_file="${name}_`date +"%d-%b-%Y"`.out"

nohup softlearning run_example_local examples.development.unconstrained_hyper_param_search \
    --universe=gym \
    --domain=PointGather \
    --task=v0 \
    --exp-name=${name} \
    --num-samples=1 \
    --trial-cpus=1  \
    --checkpoint-frequency=1000 > ${out_file}
