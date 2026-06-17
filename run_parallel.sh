#!/usr/bin/env bash
#SBATCH --job-name=tvnets_par
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=tvnets_par_%j.log

set -euo pipefail

N=${SLURM_CPUS_PER_TASK:-16}

for i in $(seq "$N"); do
    julia /home/skooiker/TVNets/YieldFactorModels.jl/test.jl > run_$i.log 2>&1 &
done
wait

echo "All $N runs complete."
