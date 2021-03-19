# module load mpi/openmpi-x86_64  
# module load ~/spack/mod/tcl/linux-centos8-skylake_avx512/openmpi-3.1.5-gcc-8.2.1-2jbaslt
export JULIA_DEPOT_PATH=`pwd`/../.julia
export JULIA_MPI_PATH=/home/cnh/spack/buildtree/linux-centos8-skylake_avx512/gcc-8.2.1/openmpi-3.1.5-2jbaslttjdic5kmi4mpoivc73cdtbh65
export JULIA_MPI_PATH=/home/cnh/spack/buildtree/linux-centos8-skylake_avx512/gcc-8.2.1/openmpi-3.1.5-zwmnmt2nbbacg35d5wzt53ef6bv4a3tm
export OMPI_MCA_mca_base_component_show_load_errors=0

mpirun --output-filename out  -np 12 ./gpu-experimenting/multigpurunner.sh ~/julia-1.5.3/bin/julia --project=. gpu-experimenting/p1.jl

