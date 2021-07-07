using Test

# H_ref = H[1:27:end]
tests_1d = Dict("diffusion_1D_expl.jl"=> [1.687700671616363e-11, 2.026554839900115e-7, 0.00026304660891236103,
                                          0.03690782595065184, 0.5597777471603071,
                                          0.9177496036501366, 0.16264630103928718, 0.003115843137683058,
                                          6.452357026295877e-6, 1.4443505264260712e-9],
                "diffusion_1D_impl.jl"=> [1.687700671616363e-11, 2.026554839900115e-7, 0.00026304660891236103,
                                          0.03690782595065184, 0.5597777471603071, 0.9177496036501366,
                                          0.16264630103928718, 0.003115843137683058, 6.452357026295877e-6,
                                          1.4443505264260712e-9],
                "diffusion_1D_damp.jl"=>[1.687700671616363e-11, 2.026554839900115e-7, 0.00026304660891236103,
                                         0.03690782595065184, 0.5597777471603071, 0.9177496036501366,
                                         0.16264630103928718, 0.003115843137683058, 6.452357026295877e-6,
                                         1.4443505264260712e-9]
                )

# H_ref = repr(H[1:33:end, 30:33:end].data)
tests_2d = Dict("iceflow.jl"=> [0.0 0.0 0.0 0.0 0.0;
                                2107.470713773525 2300.5901089255344 1839.9641691700776 1968.1291004159364 668.7717653362265;
                                1602.810338109451 2392.6458316876924 2480.2918550322306 0.0 0.0],
                 "iceflow_xpu.jl"=> [0.0 0.0 0.0 0.0 0.0;
                                     2107.470713773525 2300.5901089255344 1839.9641691700776 1968.1291004159364 668.7717653362265;
                                     1602.810338109451 2392.6458316876924 2480.2918550322306 0.0 0.0],
                 "iceflow_xpu_evo.jl"=> [0.0 0.0 0.0 0.0 0.0;
                                         0.0 0.0 0.0 0.0 0.0;
                                         0.0 251.15273565047443 0.0 0.0 0.0]
                 )

@testset "helpers.jl" begin
    include(joinpath(@__DIR__, "../scripts/helpers.jl"))
    # TODO: maybe add actual tests...
end


## Test the scripts
# note: this cannot be run in a test-set as the include acts in global scope
# note: all XPU tests relying on ParallelStencil.jl must have same dims and backend to succeed (else requires a reset)

do_visu, do_save = false, false # disable plotting & saving

for (fl, H_ref) in tests_1d
    lx = 10.0        # domain size
    nx = 256         # numerical grid resolution
    dx = lx/nx       # grid size

    println("Runninng $fl")
    xc, H = include(joinpath(@__DIR__, "../scripts", fl))
    @test xc == LinRange(dx/2, lx-dx/2, nx)
    @test H[1:27:end] ≈ H_ref
end

ref_data = load_data()
for (fl, H_ref) in tests_2d
    println("Runninng $fl")
    include(joinpath(@__DIR__, "../scripts", fl))
    @test size(ref_data[1]) == size(H)
    @test dims(ref_data[1]) == dims(H)
    @test H_ref ≈ H[1:33:end, 30:33:end].data
end
