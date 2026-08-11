#!/bin/bash -l
#SBATCH -A liyanxie
#SBATCH -p a100-4
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH -J wm_atom16_pipe
#SBATCH -o /users/2/deng0379/watermarkability_lora_msi/logs/atom16_pipeline_%j.out
#SBATCH -e /users/2/deng0379/watermarkability_lora_msi/logs/atom16_pipeline_%j.err

set -euo pipefail

ROOT=${WM_LORA_ROOT:-$HOME/watermarkability_lora_msi}
cd "$ROOT"
mkdir -p logs

echo "[pipeline] building the OPT atomic-16 partition"
bash run_opt_atomic16_partition.slurm

echo "[pipeline] partition passed; starting the matched ECC experiment"
bash run_opt_atomic16_ecc_experiment.slurm

echo "[pipeline] complete"
