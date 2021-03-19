using MPI
using PencilArrays
using LinearAlgebra: transpose!
using CUDA

MPI.Init()
comm = MPI.COMM_WORLD       # we assume MPI.Comm_size(comm) == 12
rank = MPI.Comm_rank(comm)  # rank of local process, in 0:11

# Define MPI Cartesian topology: distribute processes on a 3×4 grid.
topology = MPITopology(comm, (3, 4))

# Let's decompose 3D arrays along dimensions (2, 3).
# This corresponds to the "x-pencil" configuration in the figure.
# This configuration is described by a Pencil object.
dims_global = (42, 31, 29)  # global dimensions of the array
decomp_dims = (2, 3)
pen_x = Pencil(topology, dims_global, decomp_dims)
s_local = size_local(pen_x)
# We can now allocate distributed arrays in the x-pencil configuration.
Ax = PencilArray{Float64}(undef, pen_x)
AxC = PencilArray(pen_x, CuArray{Float64}(undef, s_local ))
fill!(Ax,  rank * π)  # each process locally fills its part of the array
fill!(AxC, rank * π)  # each process locally fills its part of the array

MPI.Barrier(comm)

if rank == 0
 println( "Ax:  ", size(Ax)          )    # size of local part
 println( "Ax:  ", size_global(Ax)   )    # total size of the array = (42, 31, 29)
 println( "AxC: ", size(AxC)         )    # size of local part
 println( "AxC: ", size_global(AxC)  )    # total size of the array = (42, 31, 29)
end

# Create another pencil configuration, decomposing along dimensions (1, 3).
# We could use the same constructor as before, but it's recommended to reuse the
# previous Pencil instead.
pen_y = Pencil(pen_x, decomp_dims=(1, 3), permute = Permutation(2, 1, 3))
# pen_y = Pencil(pen_x, decomp_dims=(1, 3) )

# Now transpose from the x-pencil to the y-pencil configuration, redistributing
# the data initially in Ax.
Ay = PencilArray{Float64}(undef, pen_y)
s_local = size_local(pen_y, MemoryOrder() )
AyC = PencilArray(pen_y, CuArray{Float64}(undef, s_local ))

MPI.Barrier(comm)

if rank == 0
 println( "Ay:  ", size(Ay)          )    # size of local part
 println( "Ay:  ", size_global(Ay)   )    # total size of the array = (42, 31, 29)
 println( "AyC: ", size(AyC)         )    # size of local part
 println( "AyC: ", size_global(AyC)  )    # total size of the array = (42, 31, 29)
end

# Create another pencil configuration, decomposing along dimensions (1, 3).

transpose!(Ay, Ax)
transpose!(AyC, AxC)

localmaxAx=maximum(@. abs(AxC - Ax) )
localmaxAy=maximum(@. abs(AyC - Ay) )

maxAx=MPI.Allreduce( localmaxAx, +, comm )
maxAy=MPI.Allreduce( localmaxAy, +, comm )

MPI.Barrier(comm)

if rank == 0
 println("maxAx: ", maxAx)
 println("maxAy: ", maxAy)
end

# Test gather on transposed arrays
AyG=gather(Ay)
Ay.=AyC
if rank == 0
 println("typeof(Ay): ",typeof(Ay))
 println("typeof(AyC): ",typeof(AyC))
end
AyCG=gather(AyC)

if rank == 0
 println("typeof(AyC): ",typeof(AyC))
 println("typeof(AyCG): ",typeof(AyCG))
 println("size(AyC): ",size(AyC))
 println("size(AyCG): ",size(AyCG))
 println("maximum(AyCG): ",maximum(AyCG))
end

## if rank == 0
##  println( maximum(@. abs(AyG - AyCG) ) )
## end

