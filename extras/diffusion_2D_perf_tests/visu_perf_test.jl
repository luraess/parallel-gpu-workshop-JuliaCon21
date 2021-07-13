using DelimitedFiles, Plots, Printf

perf_i5 = readdlm("out_diffusion_2D_damp_perf.txt")
perf_loop_i5 = readdlm("out_diffusion_2D_damp_perf_loop.txt")
perf_loop_fun_i5 = readdlm("out_diffusion_2D_damp_perf_loop_fun.txt")
perf_gpu_v100 = readdlm("out_diffusion_2D_damp_perf_gpu.txt")
perf_xpu_v100 = readdlm("out_diffusion_2D_damp_perf_xpu.txt")
xpu_v100 = readdlm("out_diffusion_2D_damp_xpu.txt")

plot_fig = 2
save_fig = false

fontsize = 11
    opts = (yaxis=font(fontsize, "Courier"), xaxis=font(fontsize, "Courier"),
            framestyle=:box, titlefontsize=fontsize, titlefont="Courier", ylabel="T_eff [GB/s]",
            linewidth=:3.0, foreground_color_legend = nothing, legendfontsize=11, legend=:topright,
            size=(600,400) )

if plot_fig==1

x = perf_i5[[1,4,5,6,7],1]
ticks = collect(x)
ticklabels = [ @sprintf("%.f",x) for x in ticks ]

p1 = plot(perf_i5[:,1], perf_i5[:,4]; xlims=(-50,4200), ylims=(-0.5, 12), label="CPU broadcast", opts...)
plot!(perf_loop_i5[:,1], perf_loop_i5[:,4]; label="CPU loop", opts...)
plot!(xticks=(ticks,ticklabels),xrotation=0)
display(plot!(perf_loop_fun_i5[:,1], perf_loop_fun_i5[:,4]; label="CPU loop function", opts...))
if save_fig savefig("perf_cpu.png") end

else

x = perf_gpu_v100[[1,6,7,8,9],1]
ticks = collect(x)
ticklabels = [ @sprintf("%.f",x) for x in ticks ]

p2 = plot(perf_gpu_v100[:,1], perf_gpu_v100[:,4]; xlims=(-50, 17000), ylims=(-10, 1100), label="GPU", xlabel="grid resolution (nx)", opts...)
plot!(perf_xpu_v100[:,1], perf_xpu_v100[:,4]; label="XPU parallel_indices", opts...)
plot!(xticks=(ticks,ticklabels),xrotation=0)
display(plot!(xpu_v100[:,1], xpu_v100[:,4]; label="XPU parallel", opts...))
if save_fig savefig("perf_gpu.png") end

end
