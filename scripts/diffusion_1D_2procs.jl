using Plots

# enable plotting by default
if !@isdefined do_visu; do_visu = true end

@views function diffusion_1D_2procs(; do_visu=true)
    # Physics
    Hl  = 10.0   # left  H
    Hr  = 1.0    # right H
    λ   = 1.0    # diffusion coeff
    nt  = 200    # number of time steps
    # Numerics
    nx  = 32     # number of local grid points
    dx  = 1.0    # cell size
    # Derived numerics
    dt  = dx^2/λ/2.1
    # Initial condition
    HL  = Hl*ones(nx)
    HR  = Hr*ones(nx)
    H   = [HL[1:end-1]; HR[2:end]]
    Hg  = copy(H)
    # Time loop
    for it = 1:nt
        # Compute physics locally
        HL[2:end-1] .= HL[2:end-1] + dt*λ*diff(diff(HL)/dx)/dx
        HR[2:end-1] .= HR[2:end-1] + dt*λ*diff(diff(HR)/dx)/dx
        # Update boundaries (MPI)
        HL[end] = HR[2]
        HR[1]   = HL[end-1]
        # Global picture
        H .= [HL[1:end-1]; HR[2:end]]
        # Compute physics globally (check)
        Hg[2:end-1] .= Hg[2:end-1] + dt*λ*diff(diff(Hg)/dx)/dx
        # Visualise
        if do_visu
            fontsize = 12
            plot(Hg, legend=false, linewidth=0, markershape=:circle, markersize=5, yaxis=font(fontsize, "Courier"), xaxis=font(fontsize, "Courier"), titlefontsize=fontsize, titlefont="Courier")
            display(plot!(H, legend=false, linewidth=3, framestyle=:box, xlabel="Lx", ylabel="H", title="diffusion (it=$(it))"))
        end
    end
    return
end

diffusion_1D_2procs(; do_visu=do_visu)
