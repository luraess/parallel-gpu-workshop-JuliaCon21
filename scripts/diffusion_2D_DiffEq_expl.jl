# Solve 2D nonlinear diffusion using OrdinaryDiffEq.jl (from the DiffEq.jl / SciML.ai universe)
using Plots, Printf, LinearAlgebra, OrdinaryDiffEq

# enable plotting by default
if !@isdefined do_visu; do_visu = true end

# finite-difference support functions
@views av_xi(A) = 0.5*(A[1:end-1,2:end-1].+A[2:end,2:end-1]) # average in x-direction
@views av_yi(A) = 0.5*(A[2:end-1,1:end-1].+A[2:end-1,2:end]) # average in y-direction
@views   inn(A) = A[2:end-1,2:end-1] # computational domain

function diffusion_2D_obj!(du, u, p, t)
    H = u
    dHdt = du
    npow, dx, dy = p.npow, p.dx, p.dy
    # TODO: make this allocation-free
    qHx    = -av_xi(H).^npow.*diff(H[:,2:end-1], dims=1)/dx  # flux
    qHy    = -av_yi(H).^npow.*diff(H[2:end-1,:], dims=2)/dy  # flux
    dHdt[2:end-1,2:end-1]  .= -diff(qHx, dims=1)/dx .- diff(qHy, dims=2)/dy     # rate of change
    dHdt[1:end,1] .= 0; dHdt[1:end,end] .= 0; dHdt[1,1:end] .= 0; dHdt[end,1:end] .= 0 # sets the BC as H[1]=H[end]=0
    return nothing
end

function diffusion_2D_DiffEq_expl(;Solver=Tsit5, do_visu=true, save_fig=false)
    # Physics
    lx, ly = 10.0, 10.0   # domain size
    npow   = 3            # power-law exponent
    ttot   = 1.0          # total simulation time
    # Numerics
    nx, ny = 128, 128     # numerical grid resolution
    # Derived numerics
    dx, dy = lx/nx, ly/ny # grid size
    xc, yc = LinRange(dx/2, lx-dx/2, nx), LinRange(dy/2, ly-dy/2, ny)
    # Initial condition
    H0      = exp.(.-(xc.-lx/2).^2 .-(yc.-ly/2)'.^2)

    prob = ODEProblem(diffusion_2D_obj!, H0, (0.0, ttot), (npow=npow, dx=dx, dy=dy))
    @time sol = solve(prob, Solver())

    # Visualize
    if do_visu
        fontsize = 12
        opts = (aspect_ratio=1, yaxis=font(fontsize, "Courier"), xaxis=font(fontsize, "Courier"),
                ticks=nothing, framestyle=:box, titlefontsize=fontsize, titlefont="Courier", colorbar_title="",
                xlabel="Lx", ylabel="Ly", xlims=(xc[1],xc[end]), ylims=(yc[1],yc[end]), clims=(0.,1.))
        display(heatmap(xc, yc, sol.u[end]'; c=:davos, title="DiffEq with solver $Solver", opts...))
        if save_fig savefig("diff2D_expl.png") end
    end
    return xc, yc, sol.u[end]
end

sol = diffusion_2D_DiffEq_expl(; do_visu=do_visu);
