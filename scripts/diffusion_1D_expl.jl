using Plots, Printf, LinearAlgebra

# enable plotting by default
if !@isdefined do_visu; do_visu = true end

@views function diffusion_1D(; do_visu=true)
    # Physics
    lx     = 10.0        # domain size
    D      = 1.0         # diffusion coefficient
    ttot   = 0.6         # total simulation time
    # Numerics
    nx     = 256         # numerical grid resolution
    # Derived numerics
    dx     = lx/nx       # grid size
    dt     = dx^2/D/2.1  # time step (obeys CFL condition)
    xc     = LinRange(dx/2, lx-dx/2, nx)
    # Array allocation
    qH     = zeros(nx-1) # on staggered grid
    dHdt   = zeros(nx-2) # normal grid, without boundary points
    # Initial condition
    H0     = exp.(-(xc.-lx/2).^2)
    H      = copy(H0)
    t = 0.0; it = 0
    # Physical time loop
    while t<ttot
        qH         .= -D*diff(H)/dx         # flux
        dHdt       .=  -diff(qH)/dx         # rate of change
        H[2:end-1] .= H[2:end-1] .+ dt*dHdt # update rule, sets the BC as H[1]=H[end]=0
        t += dt; it += 1
    end
    # Analytic solution
    Hana = 1/sqrt(4*(ttot+1/4)) * exp.(-(xc.-lx/2).^2 /(4*(ttot+1/4)))
    @printf("Total time = %1.2f, time steps = %d, error vs analytic = %1.2e \n", round(ttot, sigdigits=2), it, norm(H-Hana))
    # Visualize
    if do_visu
        plot(xc, H0, linewidth=3); display(plot!(xc, H, legend=false, framestyle=:box, linewidth=3, xlabel="lx", ylabel="H", title="explicit diffusion (nt=$it)"))
    end
    return xc, H0
end

diffusion_1D(; do_visu=do_visu);
