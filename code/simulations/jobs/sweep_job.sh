#!/bin/bash
#SBATCH --job-name=fmri_sweep                             # job name
#SBATCH --partition=GPU                                   # GPU partition
#SBATCH --nodes=1                                         # one node per task
#SBATCH --ntasks=1                                        # one task per node
#SBATCH --cpus-per-task=4                                 # CPU cores for data loading
#SBATCH --gres=gpu:1                                      # one GPU per task
#SBATCH --mem=32G                                         # memory (32 GB covers N=400)
#SBATCH --time=2-00:00:00                                 # max walltime (2 days)
#SBATCH --array=0-48                                      # 7 connectivity x 7 coupling = 49 jobs
#SBATCH --output=jobs/logs/sweep_%A_%a.out                # stdout per task
#SBATCH --error=jobs/logs/sweep_%A_%a.err                 # stderr per task
#SBATCH --mail-user=dhvani.jain@utsouthwestern.edu           # email notifications
#SBATCH --mail-type=END,FAIL                              # notify on end or failure

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
BASE_DIR=/project/greencenter/Lin_lab/s229618/fmri_connectivity_trees
SCRIPT_DIR=${BASE_DIR}/code/simulations
OUTPUT_ROOT=${SCRIPT_DIR}/output/sweep

conda activate dynamric
cd "${SCRIPT_DIR}"
mkdir -p jobs/logs

# ---------------------------------------------------------------------------
# Map SLURM array index → connectivity config (0-6) and coupling config (0-6)
# ---------------------------------------------------------------------------
CONN_IDX=$(( SLURM_ARRAY_TASK_ID / 7 ))
COUP_IDX=$(( SLURM_ARRAY_TASK_ID % 7 ))

# ---------------------------------------------------------------------------
# Connectivity configs  (indices 0-6)
#   0-4 : Erdős–Rényi with densities 0.1, 0.3, 0.5, 0.7, 0.9
#   5   : hierarchical, branching = 2
#   6   : hierarchical, branching = 3
# ---------------------------------------------------------------------------
DENSITIES=(0.1 0.3 0.5 0.7 0.9)

if [ "${CONN_IDX}" -lt 5 ]; then
    CONNECTIVITY="erdos_renyi"
    DENSITY="${DENSITIES[$CONN_IDX]}"
    CONN_EXTRA="--density ${DENSITY}"
    CONN_TAG="er_$(echo ${DENSITY} | tr '.' '_')"
elif [ "${CONN_IDX}" -eq 5 ]; then
    CONNECTIVITY="hierarchical"
    CONN_EXTRA="--branching 2"
    CONN_TAG="hier_b2"
else
    CONNECTIVITY="hierarchical"
    CONN_EXTRA="--branching 3"
    CONN_TAG="hier_b3"
fi

# ---------------------------------------------------------------------------
# Coupling configs  (indices 0-6)
#   0 : uniform — linear
#   1 : uniform — quadrature
#   2 : uniform — rectified
#   3 : uniform — squared
#   4 : uniform — pac
#   5 : random  — all 5 types
#   6 : random  — linear, quadrature, squared only
# ---------------------------------------------------------------------------
COUP_TYPES=(linear quadrature rectified squared pac)

if [ "${COUP_IDX}" -lt 5 ]; then
    COUP_TYPE="${COUP_TYPES[$COUP_IDX]}"
    COUP_EXTRA="--edge-mode uniform --coupling-type ${COUP_TYPE}"
    COUP_TAG="uniform_${COUP_TYPE}"
elif [ "${COUP_IDX}" -eq 5 ]; then
    COUP_EXTRA="--edge-mode random"
    COUP_TAG="random_all"
else
    COUP_EXTRA="--edge-mode random --coupling-types linear quadrature squared"
    COUP_TAG="random_lqs"
fi

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
OUTDIR="${OUTPUT_ROOT}/${CONN_TAG}/${COUP_TAG}"
SEED=$(( 1000 + SLURM_ARRAY_TASK_ID ))   # unique but reproducible seed per task

echo "========================================================"
echo "  Array task : ${SLURM_ARRAY_TASK_ID}"
echo "  Connectivity: ${CONNECTIVITY} ${CONN_EXTRA}  [${CONN_TAG}]"
echo "  Coupling    : ${COUP_EXTRA}  [${COUP_TAG}]"
echo "  Output      : ${OUTDIR}"
echo "  Seed        : ${SEED}"
echo "========================================================"

python sweep_sim.py \
    --N-values 5 10 15 20 25 50 75 100 200 400 \
    --trials 10 \
    --simlen 600000 \
    --connectivity "${CONNECTIVITY}" \
    ${CONN_EXTRA} \
    ${COUP_EXTRA} \
    --no-delays \
    --device cuda \
    --outdir "${OUTDIR}" \
    --seed "${SEED}"

echo "Done — task ${SLURM_ARRAY_TASK_ID} finished at $(date)"
