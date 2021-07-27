#!/bin/bash

# -> to make it executable: chmod +x runme.sh or chmod 755 runme.sh

STARTUP=true

module purge > /dev/null 2>&1

module load julia
module load cuda/11.2
module load openmpi/gcc83-316-c112

export JULIA_MPI_BINARY=system
export JULIA_CUDA_MEMORY_POOL=binned
export JULIA_CUDA_USE_BINARYBUILDER=false

export IGG_CUDAAWARE_MPI=1
export JULIA_NUM_THREADS=4

# Only the first time
if [ "$STARTUP" = true ]; then

    julia --project -e 'using Pkg; pkg"activate ."; pkg"resolve"; pkg"instantiate"'

fi

# Every time
julia --project -e 'using Pkg; pkg"instantiate"; pkg"build MPI"'
julia --project -e 'using Pkg; pkg"precompile"'
