#!/bin/bash
# Builds TeNPyvenv in this folder from requirements.txt. Run once on the CREATE login node before the first sbatch
# The venv has to be built on the cluster: one made on a Mac holds macOS binaries and links to the Mac's Python, so it won't run on the Linux nodes
# To rebuild from scratch: rm -rf TeNPyvenv && ./setup_venv.sh

set -e
cd "$(dirname "$0")"

# CREATE has no Python module, the venv is built from the system Python (3.12 on AlmaLinux 10), the same one the compute nodes use
/usr/bin/python3 -m venv TeNPyvenv
TeNPyvenv/bin/python -m pip install --upgrade pip
TeNPyvenv/bin/python -m pip install -r requirements.txt
TeNPyvenv/bin/python -c "import tenpy, numpy, scipy, matplotlib; print(f'TeNPyvenv ready: tenpy {tenpy.__version__}, numpy {numpy.__version__}, scipy {scipy.__version__}, matplotlib {matplotlib.__version__}')"
