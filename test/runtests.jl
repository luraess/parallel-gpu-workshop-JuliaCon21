using Test

# H[1:33:end,30:33:end]
tests_2d = Dict("diffusion_2D_expl.jl"=> [1.4341623497148877e-14 2.021424041084888e-11 4.8022139161215363e-14;
                                          2.394095924752031e-6 0.003374444698589263 8.016498807587078e-6;
                                          0.0006736119165607539 0.594473121056432 0.002255555227680872;
                                          3.1945000985042113e-7 0.00045025860455410284 1.069660825438506e-6],
                "diffusion_2D_impl.jl"=> [1.4341623497148877e-14 2.021424041084888e-11 4.8022139161215363e-14;
                                          2.3940959247520396e-6 0.003374444691354641 8.016498807586814e-6;
                                          0.0006736119165451054 0.6115608556980482 0.0022555552261044605;
                                          3.1945000985042113e-7 0.0004502586045507441 1.069660825438506e-6],
                "diffusion_2D_damp.jl"=> [1.4341623497148877e-14 2.021424041084888e-11 4.8022139161215363e-14;
                                          2.394095924752039e-6 0.0033744446790461404 8.016498807586816e-6;
                                          0.0006736119165187302 0.6106677330399702 0.0022555552234409136;
                                          3.1945000985042113e-7 0.00045025860454511377 1.0696608254385066e-6]
                )

## Test the scripts
# note: this cannot be run in a test-set as the include acts in global scope

do_visu, save_fig = false, false # disable plotting & saving

for (fl, H_ref) in tests_2d
    lx, ly = 10.0, 10.0         # domain size
    nx, ny = 128, 128           # numerical grid resolution
    dx, dy = lx/nx, ly/ny       # grid size

    println("Runninng $fl")
    xco, yco, Hco = include(joinpath(@__DIR__, "../scripts", fl))
    @test xco == LinRange(dx/2, lx-dx/2, nx)
    @test yco == LinRange(dy/2, ly-dy/2, ny)
    @test Hco[1:33:end,30:33:end] ≈ H_ref
end
