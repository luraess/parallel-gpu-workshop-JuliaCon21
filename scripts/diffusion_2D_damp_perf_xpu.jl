const USE_GPU = true
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2)
else
    @init_parallel_stencil(Threads, Float64, 2)
end
using Plots, Printf, LinearAlgebra

# enable plotting by default
if !@isdefined do_visu; do_visu = false end

# macros to avoid array allocation
macro qHx(ix,iy)  esc(:( -(0.5*(H[$ix,$iy+1]+H[$ix+1,$iy+1]))*(0.5*(H[$ix,$iy+1]+H[$ix+1,$iy+1]))*(0.5*(H[$ix,$iy+1]+H[$ix+1,$iy+1])) * (H[$ix+1,$iy+1]-H[$ix,$iy+1])*_dx )) end
macro qHy(ix,iy)  esc(:( -(0.5*(H[$ix+1,$iy]+H[$ix+1,$iy+1]))*(0.5*(H[$ix+1,$iy]+H[$ix+1,$iy+1]))*(0.5*(H[$ix+1,$iy]+H[$ix+1,$iy+1])) * (H[$ix+1,$iy+1]-H[$ix+1,$iy])*_dy )) end
macro dtau(ix,iy) esc(:(  (1.0/(min_dxy2 / (H[$ix+1,$iy+1]*H[$ix+1,$iy+1]*H[$ix+1,$iy+1]) / 4.1) + _dt)^-1  )) end

# @parallel_indices (ix,iy) function compute_update!(H2::Data.Array, dHdtau::Data.Array, H, Hold::Data.Array, _dt::Data.Number, damp::Data.Number, min_dxy2::Data.Number, _dx::Data.Number, _dy::Data.Number)
@parallel_indices (ix,iy) function compute_update!(H2, dHdtau, H, Hold, _dt, damp, min_dxy2, _dx, _dy)
    if (ix<=size(dHdtau,1) && iy<=size(dHdtau,2)) dHdtau[ix,iy] = -(H[ix+1, iy+1] - Hold[ix+1, iy+1])*_dt + 
                                                                   (-(@qHx(ix+1,iy)-@qHx(ix,iy))*_dx -(@qHy(ix,iy+1)-@qHy(ix,iy))*_dy) +
                                                                   damp*dHdtau[ix,iy] end                       # damped rate of change
    if (ix<=size(dHdtau,1) && iy<=size(dHdtau,2)) H2[ix+1,iy+1] = H[ix+1,iy+1] + @dtau(ix,iy)*dHdtau[ix,iy] end # update rule, sets the BC as H[1]=H[end]=0
    return
end

# @parallel_indices (ix,iy) function compute_residual!(ResH::Data.Array, H::Data.Array, Hold::Data.Array, _dt::Data.Number, _dx::Data.Number, _dy::Data.Number)
@parallel_indices (ix,iy) function compute_residual!(ResH, H, Hold, _dt, _dx, _dy)
    if (ix<=size(ResH,1) && iy<=size(ResH,2)) ResH[ix,iy] = -(H[ix+1, iy+1] - Hold[ix+1, iy+1])*_dt + 
                                                             (-(@qHx(ix+1,iy)-@qHx(ix,iy))*_dx -(@qHy(ix,iy+1)-@qHy(ix,iy))*_dy) end
    return
end

# @parallel_indices (ix,iy) function assign!(Hold::Data.Array, H::Data.Array)
@parallel_indices (ix,iy) function assign!(Hold, H)
    if (ix<=size(H,1) && iy<=size(H,2)) Hold[ix,iy] = H[ix,iy] end
    return
end 

@views function diffusion_2D_damp_perf_xpu(; do_visu=true, save_fig=false)
    # Physics
    lx, ly = 10.0, 10.0   # domain size
    ttot   = 1.0          # total simulation time
    dt     = 0.2          # physical time step
    # Numerics
    nx, ny = 16*32*16, 16*32*16 # numerical grid resolution
    # nx, ny = 512, 512 # numerical grid resolution
    nout   = 100          # check error every nout
    tol    = 1e-6         # tolerance
    itMax  = 1e5          # max number of iterations
    damp   = 1-35/nx      # damping (this is a tuning parameter, dependent on e.g. grid resolution)
    # Derived numerics
    dx, dy = lx/nx, ly/ny # grid size
    xc, yc = LinRange(dx/2, lx-dx/2, nx), LinRange(dy/2, ly-dy/2, ny)
    # Array allocation
    ResH   = @zeros(nx-2, ny-2) # normal grid, without boundary points
    dHdtau = @zeros(nx-2, ny-2) # normal grid, without boundary points
    # Initial condition
    H      = Data.Array(exp.(.-(xc.-lx/2).^2 .-(yc.-ly/2)'.^2))
    Hold   = copy(H)
    H2     = copy(H)
    _dx, _dy, _dt = 1.0/dx, 1.0/dy, 1.0/dt
    min_dxy2  = min(dx,dy)^2
    t = 0.0; it = 0; ittot = 0; t_tic = 0.0; niter = 0
    # Physical time loop
    while t<ttot
        iter = 0; err = 2*tol
        # Picard-type iteration
        while err>tol && iter<itMax
            if (it==1 && iter==0) t_tic = Base.time(); niter = 0 end
            @parallel compute_update!(H2, dHdtau, H, Hold, _dt, damp, min_dxy2, _dx, _dy)
            H, H2 = H2, H
            if iter % nout == 0
                @parallel compute_residual!(ResH, H, Hold, _dt, _dx, _dy)
                err = norm(ResH)/length(ResH)
            end
            iter += 1; niter += 1
        end
        ittot += iter; it += 1; t += dt
        @parallel assign!(Hold, H)
    end
    t_toc = Base.time() - t_tic
    A_eff = (2*2+1)/1e9*nx*ny*sizeof(Float64)  # Effective main memory access per iteration [GB]
    t_it  = t_toc/niter                        # Execution time per iteration [s]
    T_eff = A_eff/t_it                         # Effective memory throughput [GB/s]
    @printf("Time = %1.3f sec, T_eff = %1.2f GB/s (niter = %d)\n", t_toc, round(T_eff, sigdigits=2), niter)
    # Visualize
    if do_visu
        fontsize = 12
        opts = (aspect_ratio=1, yaxis=font(fontsize, "Courier"), xaxis=font(fontsize, "Courier"),
                ticks=nothing, framestyle=:box, titlefontsize=fontsize, titlefont="Courier", colorbar_title="",
                xlabel="Lx", ylabel="Ly", xlims=(xc[1],xc[end]), ylims=(yc[1],yc[end]), clims=(0.,1.))
        display(heatmap(xc, yc, Array(H)'; c=:davos, title="damped diffusion (nt=$it, iters=$ittot)", opts...))
        if save_fig savefig("diff2D_damp.png") end
    end
    return
end

diffusion_2D_damp_perf_xpu(; do_visu=do_visu)
