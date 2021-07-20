#!/bin/bash

# -> to make it executable: chmod +x runme.sh or chmod 755 runme.sh

RESOL=(64 128 256 512 1024 2048 4096 8192 16384)

declare -a RUN=( "diffusion_2D_damp_perf_gpu" "diffusion_2D_damp_perf_xpu2" "diffusion_2D_damp_perf_xpu" "diffusion_2D_damp_xpu" "diffusion_2D_damp_perf_loop_fun" "diffusion_2D_damp_perf_loop" "diffusion_2D_damp_perf" )

USE_GPU=true
DO_VIZ=false
DO_SAVE=true

# rm output/*.txt

if [ ! -d ./output ]; then
  mkdir -p ./output;
fi

# Read the array values with space
for name in "${RUN[@]}"; do

    if [ "$DO_SAVE" = "true" ]; then
        FILE=./output/out_$name.txt
        if [ -f "$FILE" ]; then
            echo "Systematic results (file $FILE) already exists. Remove to continue."
            exit 0
        else 
            echo "Launching systematics (saving results to $FILE)."
        fi
    fi

    for i in "${RESOL[@]}"; do

        USE_GPU=$USE_GPU DO_VIZ=$DO_VIZ DO_SAVE=$DO_SAVE NX=$i NY=$i PS_THREAD_BOUND_CHECK=0 julia --project -O3 --check-bounds=no "$name".jl
    
    done

done
