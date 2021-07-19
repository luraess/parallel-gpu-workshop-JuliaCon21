# Linear 1D diffusion with n fake mpi processes
using Plots

# enable plotting by default
if !@isdefined do_visu; do_visu = true end

@views function diffusion_1D_nprocs(; do_visu=true)
    # Physics
    lx  = 10.0
    λ   = 1.0
    nt  = 200
    # Numerics
    np  = 4             # number of procs
    nx  = 32            # local number of grid points
    # Derived numerics
    nxg = (nx-2)*np+2   # global number of grid points
    dxg = lx/nxg        # dx for global grid
    dt  = dxg^2/λ/2.1
    # Array allocation
    x   = zeros(nx,np)  # local coord array
    H   = zeros(nx,np)  # local H array
    xt  = zeros(nxg)    # global coord array
    Ht  = zeros(nxg)    # global initial H array
    Hg  = zeros(nxg)    # global H array
    # Initial condition
    for ip = 1:np
        i1 = 1 + (ip-1)*(nx-2)
        for ix = 1:nx
            x[ix,ip] = ( (ip-1)*(nx-2) + (ix-0.5) )*dxg - 0.5*lx
            H[ix,ip] = exp(-x[ix,ip]^2)
        end
        xt[i1:i1+nx-2] .= x[1:end-1,ip]; if (ip==np) xt[i1+nx-1] = x[end,ip] end
        Ht[i1:i1+nx-2] .= H[1:end-1,ip]; if (ip==np) Ht[i1+nx-1] = H[end,ip] end
    end
    # Time loop
    for it = 1:nt
        for ip = 1:np # compute physics locally
            H[2:end-1,ip] .= H[2:end-1,ip] .+ dt*λ*diff(diff(H[:,ip])/dxg)/dxg
        end
        for ip = 1:np-1 # update boundaries
            H[end,ip  ] = H[    2,ip+1]
            H[  1,ip+1] = H[end-1,ip  ]
        end
        for ip = 1:np # global picture
            i1 = 1 + (ip-1)*(nx-2)
            Hg[i1:i1+nx-2] .= H[1:end-1,ip]
        end
        # Visualise
        if do_visu
            fontsize = 12
            plot(xt, Ht, legend=false, linewidth=1, markershape=:circle, markersize=3, yaxis=font(fontsize, "Courier"), xaxis=font(fontsize, "Courier"), titlefontsize=fontsize, titlefont="Courier")
            display(plot!(xt, Hg, legend=false, linewidth=3, framestyle=:box, xlabel="Lx", ylabel="H", title="diffusion (it=$(it))"))
        end
    end
end

diffusion_1D_nprocs(; do_visu=do_visu)
