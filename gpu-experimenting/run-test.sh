module load mpi/openmpi-x86_64  
export JULIA_DEPOT_PATH=`pwd`/../.julia
export JULIA_MPI_PATH=/usr/lib64/openmpi
export OMPI_MCA_mca_base_component_show_load_errors=0
export CUDA_VISIBLE_DEVICES=0,1,2,3

mpirun --output-filename out  -np 12 ~/julia-1.5.3/bin/julia --project=. gpu-experimenting/p1.jl

