using Plots, MAT

nprocs = (2, 2) # nprocs (x, y) dim

@views function vizme2D_mpi(nprocs)
    H  = []
    ip = 1
    for ipx = 1:nprocs[1]
        for ipy = 1:nprocs[2]
            file = matopen("H_$(ip-1).mat"); H_loc = read(file, "H"); close(file)
            nx_i, ny_i = size(H_loc,1)-2, size(H_loc,2)-2
            ix1, iy1   = 1+(ipx-1)*nx_i, 1+(ipy-1)*ny_i
            if (ip==1)  H = zeros(nprocs[1]*nx_i, nprocs[2]*ny_i)  end
            H[ix1:ix1+nx_i-1,iy1:iy1+ny_i-1] .= H_loc[2:end-1,2:end-1]
            ip += 1
        end
    end
    fontsize = 12
    opts = (aspect_ratio=1, yaxis=font(fontsize, "Courier"), xaxis=font(fontsize, "Courier"),
        ticks=nothing, framestyle=:box, titlefontsize=fontsize, titlefont="Courier", 
        xlabel="Lx", ylabel="Ly", xlims=(1, size(H,1)), ylims=(1, size(H,2)) )
    display(heatmap(H'; c=:davos, title="diffusion 2D MPI", opts...))
    return
end

vizme2D_mpi(nprocs)
