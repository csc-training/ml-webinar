#!/bin/bash
rm -rf ~/.local/lib/python3.*
rm ~/ml-webinar/examples/slurm*out
cd ~/ml-webinar
git checkout -- examples/run.sh
git status
env | grep SBATCH_
