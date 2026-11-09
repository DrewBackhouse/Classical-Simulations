#!/bin/bash
#SBATCH --job-name=QRTEBD
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out

# Runs Neuron_Tenpy_QRTEBD.py once. Submit from this folder (logs/ must exist for the --output above, so create it first):
#   mkdir -p logs && sbatch Bash_Script.sh --init 00000 --N 8 --max_bond 64 --timestep 0.05
# Everything after Bash_Script.sh is passed straight to the Python script (see Command_Line_Parameters), left out it uses the script's defaults
# Override resources on the command line, before the script name: sbatch --mem 4G --time 1-00:00:00 Bash_Script.sh ...

cd "$SLURM_SUBMIT_DIR"
if [ ! -x TeNPyvenv/bin/python ]; then
    echo "TeNPyvenv not found in $PWD, run ./setup_venv.sh on the login node first"
    exit 1
fi
mkdir -p Output logs # np.savez writes into Output/ and fails if it is missing (it is excluded from the rsync)

# One process, so limit BLAS threads to the allocated cores (1 unless submitted with --cpus-per-task)
NT=${SLURM_CPUS_PER_TASK:-1}
export OMP_NUM_THREADS=$NT OPENBLAS_NUM_THREADS=$NT MKL_NUM_THREADS=$NT
export MPLBACKEND=Agg # No display on the compute nodes, plt.show() becomes a no-op

source TeNPyvenv/bin/activate # Built from the system /usr/bin/python3 by setup_venv.sh, no module needed

echo "Job $SLURM_JOB_ID on $(hostname), threads=$NT, args: $*"
START=$(date +%s)
python Neuron_Tenpy_QRTEBD.py "$@"
STATUS=$?
echo "Job $SLURM_JOB_ID ($*) exit $STATUS, done in $(( $(date +%s) - START ))s"
exit $STATUS
