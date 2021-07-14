const USE_GPU = false
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2)
else
    @init_parallel_stencil(Threads, Float64, 2)
end
using ImplicitGlobalGrid, Plots, Printf, Statistics
import MPI

@parallel function compute_flux!(qHx::Data.Array, qHy::Data.Array, H::Data.Array, λ::Data.Number, dx::Data.Number, dy::Data.Number)
    @all(qHx) = -λ*@d_xi(H)/dx
    @all(qHy) = -λ*@d_yi(H)/dy
    return
end

@parallel function update_H!(H::Data.Array, qHx::Data.Array, qHy::Data.Array, dt::Data.Number, dx::Data.Number, dy::Data.Number)
    @inn(H) = @inn(H) - dt*(@d_xa(qHx)/dx + @d_ya(qHy)/dy)
    return
end

@views function diffusion_2D_multixpu()
    do_visu = true
    # Physics
    lx, ly  = 10.0, 10.0
    λ       = 1.0
    nt      = 200
    # Numerics
    nx, ny  = 32, 32
    nout    = 10
    # Derived numerics
    me, dims = init_global_grid(nx, ny, 1) # MPI initialisation
    @static if USE_GPU select_device() end                       # select one GPU per MPI local rank (if >1 GPU per node)
    dx, dy  = lx/nx_g(), ly/ny_g()                                # cell sizes
    dt      = min(dx^2,dy^2)/λ/4.1
    # Array allocation
    qHx     = @zeros(nx-1,ny-2)
    qHy     = @zeros(nx-2,ny-1)
    H       = @zeros(nx  ,ny  )
    # Initial condition
    H      .= Data.Array([exp(-(x_g(ix,dx,H)+dx/2 -0.5*lx)^2 -(y_g(iy,dy,H)+dy/2 -0.5*ly)^2) for ix=1:size(H,1), iy=1:size(H,2)])
    # Preparation of visualisation
    if do_visu
        if me==0
            ENV["GKSwstype"]="nul"; if isdir("viz2D_xpu_out")==false mkdir("viz2D_xpu_out") end; loadpath = "./viz2D_xpu_out/"; anim = Animation(loadpath,String[])
            println("Animation directory: $(anim.dir)")
        end
        nx_v, ny_v = (nx-2)*dims[1], (ny-2)*dims[2]
        if (nx_v*ny_v*sizeof(Data.Number) > 0.8*Sys.free_memory()) error("Not enough memory for visualization.") end
        H_v   = zeros(nx_v, ny_v) # global array for visu
        H_inn = zeros(nx-2, ny-2) # no halo local array for visu
        Xi_g, Yi_g = LinRange(dx+dx/2, lx-dx-dx/2, nx_v), LinRange(dy+dy/2, ly-dy-dy/2, ny_v) # inner points only
    end
    t_tic = 0.0
    # Time loop
    for it = 1:nt
        if (it==11) t_tic = Base.time() end
        @parallel compute_flux!(qHx, qHy, H, λ, dx, dy)
        @hide_communication (8, 4) begin # communication/computation overlap
            @parallel update_H!(H, qHx, qHy, dt, dx, dy)
            update_halo!(H)
        end
        # Visualise
        if it % nout == 0 && do_visu
            H_inn .= H[2:end-1,2:end-1]; gather!(H_inn, H_v)
            if (me==0)
            fontsize = 12
            opts = (aspect_ratio=1, yaxis=font(fontsize, "Courier"), xaxis=font(fontsize, "Courier"),
                ticks=nothing, framestyle=:box, titlefontsize=fontsize, titlefont="Courier", 
                xlabel="Lx", ylabel="Ly", xlims=(Xi_g[1], Xi_g[end]), ylims=(Yi_g[1], Yi_g[end]), clims=(0.,1.))
            heatmap(Xi_g, Yi_g, H_v'; c=:davos, title="diffusion 2D MPI (it=$it)", opts...); frame(anim)
            end
        end
    end
    t_toc = Base.time()-t_tic
    if (me==0) @printf("Time = %1.4e s, T_eff = %1.2f GB/s \n", t_toc, round((2/1e9*nx*ny*sizeof(lx))/(t_toc/(nt-10)), sigdigits=2)) end
    if (do_visu && me==0) gif(anim, "diffusion_2D_multixpu.gif", fps = 5)  end
    finalize_global_grid()
    return
end

diffusion_2D_multixpu()
