using Plots, MAT

nprocs = 4

@views function vizme1D_mpi(nprocs)
    H = []
    for ip = 1:nprocs
        file = matopen("H_$(ip-1).mat"); H_loc = read(file, "H"); close(file)
        nx_i = length(H_loc)-2
        i1   = 1 + (ip-1)*nx_i
        if (ip==1)  H = zeros(nprocs*nx_i)  end
        H[i1:i1+nx_i-1] .= H_loc[2:end-1]
    end
    fontsize = 12
    display(plot(H, legend=false, framestyle=:box, linewidth=3, xlims=(1, length(H)), ylims=(0, 1), xlabel="nx", title="diffusion 1D MPI", yaxis=font(fontsize, "Courier"), xaxis=font(fontsize, "Courier"), titlefontsize=fontsize, titlefont="Courier"))
    return
end

vizme1D_mpi(nprocs)
