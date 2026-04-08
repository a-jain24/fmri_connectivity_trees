#!/bin/bash
#SBATCH --job-name=fmri_grid                              # job name
#SBATCH --partition=GPU                                   # GPU partition
#SBATCH --nodes=1                                         # one node per task
#SBATCH --ntasks=1                                        # one task per node
#SBATCH --cpus-per-task=4                                 # CPU cores for data loading
#SBATCH --gres=gpu:1                                      # one GPU per task
#SBATCH --mem=64G                                         # memory (64 GB — N=400 x G x noise grid)
#SBATCH --time=3-00:00:00                                 # max walltime (3 days)
#SBATCH --array=0-7                                       # 8 connectivity specs (indices 0-7)
#SBATCH --output=jobs/logs/grid_%A_%a.out                 # stdout per task
#SBATCH --error=jobs/logs/grid_%A_%a.err                  # stderr per task
#SBATCH --mail-user=dhvani.jain@utsouthwestern.edu        # email notifications
#SBATCH --mail-type=END,FAIL                              # notify on end or failure

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
BASE_DIR=/project/greencenter/Lin_lab/s229618/fmri_connectivity_trees
SCRIPT_DIR=${BASE_DIR}/code/simulations

eval "$(conda shell.bash hook)"
conda activate dynamric
cd "${SCRIPT_DIR}"
mkdir -p jobs/logs

# Default to 0 when run outside SLURM (e.g. bash grid_job.sh locally)
SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

# ---------------------------------------------------------------------------
# Map SLURM array index → connectivity spec  (indices 0-7)
#   0   : tree
#   1   : hierarchical, branching = 2
#   2   : hierarchical, branching = 3
#   3-7 : Erdős–Rényi with densities 0.1, 0.3, 0.5, 0.7, 0.9
# ---------------------------------------------------------------------------
case "${SLURM_ARRAY_TASK_ID}" in
    0) CONN_SPEC="tree";            CONN_TAG="tree"     ;;
    1) CONN_SPEC="hierarchical:2";  CONN_TAG="hier_b2"  ;;
    2) CONN_SPEC="hierarchical:3";  CONN_TAG="hier_b3"  ;;
    3) CONN_SPEC="erdos_renyi:0.1"; CONN_TAG="er_0p10"  ;;
    4) CONN_SPEC="erdos_renyi:0.3"; CONN_TAG="er_0p30"  ;;
    5) CONN_SPEC="erdos_renyi:0.5"; CONN_TAG="er_0p50"  ;;
    6) CONN_SPEC="erdos_renyi:0.7"; CONN_TAG="er_0p70"  ;;
    7) CONN_SPEC="erdos_renyi:0.9"; CONN_TAG="er_0p90"  ;;
    *) echo "Unexpected SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"; exit 1 ;;
esac

# ---------------------------------------------------------------------------
# Grid parameters — shared across all tasks
#   N values   : 10 levels matching sweep_job.sh
#   G values   : 3 global coupling strengths
#   Noise (D)  : 5 individuals per cell, spanning 4 orders of magnitude
# ---------------------------------------------------------------------------
N_VALUES="5 10 15 20 25 50 75 100 200 400"
G_VALUES="0.1 0.3 0.5"
NOISE_LEVELS="1e-5 5e-5 2e-4 8e-4 3e-3"

SEED=$(( 2000 + SLURM_ARRAY_TASK_ID ))   # unique but reproducible seed per task
RUN_ID="grid_${CONN_TAG}"

echo "========================================================"
echo "  Array task  : ${SLURM_ARRAY_TASK_ID}"
echo "  Connectivity: ${CONN_SPEC}  [${CONN_TAG}]"
echo "  N values    : ${N_VALUES}"
echo "  G values    : ${G_VALUES}"
echo "  Noise levels: ${NOISE_LEVELS}"
echo "  Run ID      : ${RUN_ID}"
echo "  Seed        : ${SEED}"
echo "========================================================"

python grid_sim.py \
    --N-values ${N_VALUES} \
    --G-values ${G_VALUES} \
    --connectivity "${CONN_SPEC}" \
    --noise-levels ${NOISE_LEVELS} \
    --simlen 600000 \
    --edge-mode random \
    --device cuda \
    --run-id "${RUN_ID}" \
    --seed "${SEED}"

echo "Done — task ${SLURM_ARRAY_TASK_ID} finished at $(date)"
