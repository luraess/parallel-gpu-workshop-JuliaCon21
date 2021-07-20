using DelimitedFiles, Plots, Printf

plot_fig = 2 # 1: CPU results, 2: GPU results, 3: iter scaling
save_fig = false

fontsize = 11
opts = (yaxis=font(fontsize, "Courier"), xaxis=font(fontsize, "Courier"),
        framestyle=:box, titlefontsize=fontsize, titlefont="Courier", ylabel="T_eff [GB/s]",
        linewidth=:3.0, foreground_color_legend = nothing, legendfontsize=11, legend=:topright,
        size=(600,400) )

if plot_fig==1

    perf_i5 = readdlm("output/out_diffusion_2D_damp_perf.txt")
    perf_loop_i5 = readdlm("output/out_diffusion_2D_damp_perf_loop.txt")
    perf_loop_fun_i5 = readdlm("output/out_diffusion_2D_damp_perf_loop_fun.txt")

    x = perf_i5[[1,4,5,6,7],1]
    ticks = collect(x)
    ticklabels = [ @sprintf("%.f",x) for x in ticks ]

    p1 = plot(perf_i5[:,1], perf_i5[:,4]; xlims=(-50,4200), ylims=(-0.5, 12), label="CPU broadcast", opts...)
    plot!(perf_loop_i5[:,1], perf_loop_i5[:,4]; label="CPU loop", opts...)
    plot!(xticks=(ticks,ticklabels),xrotation=0)
    display(plot!(perf_loop_fun_i5[:,1], perf_loop_fun_i5[:,4]; label="CPU loop function", opts...))
    if save_fig savefig("perf_cpu.png") end

elseif plot_fig==2

    perf_gpu_v100 = readdlm("output/out_diffusion_2D_damp_perf_gpu.txt")
    perf_xpu_v100 = readdlm("output/out_diffusion_2D_damp_perf_xpu.txt")
    xpu_v100 = readdlm("output/out_diffusion_2D_damp_xpu.txt")
    perf_xpu2_v100 = readdlm("output/out_diffusion_2D_damp_perf_xpu2.txt")

    x = perf_gpu_v100[[1,6,7,8,9],1]
    ticks = collect(x)
    ticklabels = [ @sprintf("%.f",x) for x in ticks ]

    p2 = plot(perf_gpu_v100[:,1], perf_gpu_v100[:,4]; xlims=(-50, 17000), ylims=(-10, 1150), label="GPU", xlabel="number of grid points (nx)", opts...)
    plot!(perf_xpu2_v100[:,1], perf_xpu2_v100[:,4]; label="XPU parallel_indices 2", markershape=:circle, markersize=5, linealpha=0, opts...)
    plot!(perf_xpu_v100[:,1], perf_xpu_v100[:,4]; label="XPU parallel_indices", opts...)
    plot!(xticks=(ticks,ticklabels),xrotation=0)
    display(plot!(xpu_v100[:,1], xpu_v100[:,4]; label="XPU parallel", opts...))
    if save_fig savefig("perf_gpu.png") end

elseif plot_fig==3

    scaling_iters = readdlm("output/out_diffusion_2D_damp_perf_gpu_iters.txt")

    x = scaling_iters[[2,6,7,8,9],1]
    ticks = collect(x)
    ticklabels = [ @sprintf("%.f",x) for x in ticks ]

    p3 = plot(scaling_iters[2:end,1], scaling_iters[2:end,5]./scaling_iters[2:end,1]; xlims=(-50, 17000), ylims=(-0.2, 5), label=:false, markershape=:circle, markersize=5, xlabel="number of grid points (nx)", opts...)
    plot!(xticks=(ticks,ticklabels),xrotation=0, ylabel="niter/nx")
    display(p3)
    if save_fig savefig("iter_scale.png") end

end
