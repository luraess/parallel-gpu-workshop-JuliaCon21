# 2D nonlinear diffusion multi-XPU implicit solver with acceleration
const USE_GPU = false
const do_visu = true
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2)
else
    @init_parallel_stencil(Threads, Float64, 2)
end
using ImplicitGlobalGrid, Plots, Printf, LinearAlgebra
import MPI

# global reduction
norm_g(A) = (sum2_l = sum(A.^2); sqrt(MPI.Allreduce(sum2_l, MPI.SUM, MPI.COMM_WORLD)))
# macros to avoid array allocation
macro qHx(ix,iy)  esc(:( -(0.5*(H[$ix,$iy+1]+H[$ix+1,$iy+1]))*(0.5*(H[$ix,$iy+1]+H[$ix+1,$iy+1]))*(0.5*(H[$ix,$iy+1]+H[$ix+1,$iy+1])) * (H[$ix+1,$iy+1]-H[$ix,$iy+1])*_dx )) end
macro qHy(ix,iy)  esc(:( -(0.5*(H[$ix+1,$iy]+H[$ix+1,$iy+1]))*(0.5*(H[$ix+1,$iy]+H[$ix+1,$iy+1]))*(0.5*(H[$ix+1,$iy]+H[$ix+1,$iy+1])) * (H[$ix+1,$iy+1]-H[$ix+1,$iy])*_dy )) end
macro dtau(ix,iy) esc(:(  (1.0/(min_dxy2 / (H[$ix+1,$iy+1]*H[$ix+1,$iy+1]*H[$ix+1,$iy+1]) / 4.1) + _dt)^-1  )) end

@parallel_indices (ix,iy) function compute_update!(H2, dHdtau, H, Hold, _dt, damp, min_dxy2, _dx, _dy)
    if (ix<=size(dHdtau,1) && iy<=size(dHdtau,2))
        dHdtau[ix,iy] = -(H[ix+1, iy+1] - Hold[ix+1, iy+1])*_dt + 
                         (-(@qHx(ix+1,iy)-@qHx(ix,iy))*_dx -(@qHy(ix,iy+1)-@qHy(ix,iy))*_dy) +
                         damp*dHdtau[ix,iy]                        # damped rate of change
        H2[ix+1,iy+1] = H[ix+1,iy+1] + @dtau(ix,iy)*dHdtau[ix,iy]  # update rule, sets the BC as H[1]=H[end]=0
    end
    return
end

@parallel_indices (ix,iy) function compute_residual!(ResH, H, Hold, _dt, _dx, _dy)
    if (ix<=size(ResH,1) && iy<=size(ResH,2))
        ResH[ix,iy] = -(H[ix+1, iy+1] - Hold[ix+1, iy+1])*_dt + 
                       (-(@qHx(ix+1,iy)-@qHx(ix,iy))*_dx -(@qHy(ix,iy+1)-@qHy(ix,iy))*_dy)
    end
    return
end

@parallel_indices (ix,iy) function assign!(Hold, H)
    if (ix<=size(H,1) && iy<=size(H,2)) Hold[ix,iy] = H[ix,iy] end
    return
end 

@views function diffusion_2D_damp_perf_multixpu()
    # Physics
    lx, ly = 10.0, 10.0   # domain size
    ttot   = 1.0          # total simulation time
    dt     = 0.2          # physical time step
    # Numerics
    # nx, ny = 32*16*16, 8*64*16 # number of grid points
    nx, ny = 32, 32       # number of grid points
    nout   = 100          # check error every nout
    tol    = 1e-6         # tolerance
    itMax  = 1e5          # max number of iterations
    # Derived numerics
    me, dims = init_global_grid(nx, ny, 1)  # Initialization of MPI and more...
    @static if USE_GPU select_device() end  # select one GPU per MPI local rank (if >1 GPU per node)
    dx, dy = lx/nx_g(), ly/ny_g()           # grid size
    damp   = 1-35/nx_g()                    # damping (this is a tuning parameter, dependent on e.g. grid resolution)
    # Array allocation
    ResH   = @zeros(nx-2, ny-2) # normal grid, without boundary points
    dHdtau = @zeros(nx-2, ny-2) # normal grid, without boundary points
    H      = @zeros(nx  , ny  )
    # Initial condition
    H     .= Data.Array([exp(-(x_g(ix,dx,H)+dx/2 -lx/2)^2 -(y_g(iy,dy,H)+dy/2 -ly/2)^2) for ix=1:size(H,1), iy=1:size(H,2)])
    Hold   = copy(H)
    H2     = copy(H)
    _dx, _dy, _dt = 1.0/dx, 1.0/dy, 1.0/dt
    min_dxy2 = min(dx,dy)^2
    len_ResH_g = ((nx-2-2)*dims[1]+2)*((ny-2-2)*dims[2]+2)
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
    t = 0.0; it = 0; ittot = 0; t_tic = 0.0; niter = 0
    # Physical time loop
    while t<ttot
        iter = 0; err = 2*tol
        # Picard-type iteration
        while err>tol && iter<itMax
            if (it==1 && iter==0) t_tic = Base.time(); niter = 0 end
            @hide_communication (8, 4) begin # communication/computation overlap
                @parallel compute_update!(H2, dHdtau, H, Hold, _dt, damp, min_dxy2, _dx, _dy)
                H, H2 = H2, H
                update_halo!(H)
            end
            if iter % nout == 0
                @parallel compute_residual!(ResH, H, Hold, _dt, _dx, _dy)
                err = norm_g(ResH)/len_ResH_g
            end
            iter += 1; niter += 1
        end
        ittot += iter; it += 1; t += dt
        @parallel assign!(Hold, H)
        # Visualize
        if do_visu
            H_inn .= H[2:end-1,2:end-1]; gather!(H_inn, H_v)
            if (me==0)
                fontsize = 12
                opts = (aspect_ratio=1, yaxis=font(fontsize, "Courier"), xaxis=font(fontsize, "Courier"),
                        ticks=nothing, framestyle=:box, titlefontsize=fontsize, titlefont="Courier", colorbar_title="",
                        xlabel="Lx", ylabel="Ly", xlims=(Xi_g[1], Xi_g[end]), ylims=(Yi_g[1], Yi_g[end]), clims=(0.,1.))
                heatmap(Xi_g, Yi_g, Array(H_v)'; c=:davos, title="diffusion MPI (nt=$it)", opts...); frame(anim)
            end
        end
    end
    t_toc = Base.time() - t_tic
    A_eff = (2*2+1)/1e9*nx_g()*ny_g()*sizeof(Data.Number) # Effective main memory access per iteration [GB]
    t_it  = t_toc/niter                           # Execution time per iteration [s]
    T_eff = A_eff/t_it                            # Effective memory throughput [GB/s]
    if (me==0) @printf("Time = %1.3f sec, T_eff = %1.2f GB/s (niter = %d)\n", t_toc, round(T_eff, sigdigits=2), niter) end
    if (do_visu && me==0) gif(anim, "diffusion_2D_multixpu.gif", fps = 5)  end
    finalize_global_grid()
    return
end

diffusion_2D_damp_perf_multixpu()
