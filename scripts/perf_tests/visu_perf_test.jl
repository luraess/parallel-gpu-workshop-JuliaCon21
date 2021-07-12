using DelimitedFiles, Plots

perf_i5 = readdlm("out_diffusion_2D_damp_perf_i5.txt")
perf_loop_i5 = readdlm("out_diffusion_2D_damp_perf_loop_i5.txt")
perf_loop_fun_i5 = readdlm("out_diffusion_2D_damp_perf_loop_fun_i5.txt")
perf_gpu_v100 = readdlm("out_diffusion_2D_damp_perf_gpu_v100.txt")
perf_xpu_v100 = readdlm("out_diffusion_2D_damp_perf_xpu_v100.txt")
xpu_v100 = readdlm("out_diffusion_2D_damp_xpu_v100.txt")

plot_fig = 1
save_fig = true

fontsize = 12
    opts = (yaxis=font(fontsize, "Courier"), xaxis=font(fontsize, "Courier"),
            framestyle=:box, titlefontsize=fontsize, titlefont="Courier",
            linewidth=:3.0, foreground_color_legend = nothing, legendfontsize=11, legend=:topright, 
            xlims=(-50,4200))

if plot_fig==1

p1 = plot(perf_i5[:,1], perf_i5[:,4]; label="CPU broadcast", ylabel="T_eff [GB/s]", ylims=(-1, 32), opts...)
plot!(perf_loop_i5[:,1], perf_loop_i5[:,4]; label="CPU loop", opts...)
display(plot!(perf_loop_fun_i5[:,1], perf_loop_fun_i5[:,4]; label="CPU loop function", opts...))
if save_fig savefig("perf_cpu.png") end
else

p2 = plot(perf_gpu_v100[:,1], perf_gpu_v100[:,4]; label="GPU", xlabel="grid resolution (nx)", ylabel="T_eff [GB/s]", ylims=(-10, 1300), opts...)
plot!(perf_xpu_v100[:,1], perf_xpu_v100[:,4]; label="XPU parallel_indices", opts...)
display(plot!(xpu_v100[:,1], xpu_v100[:,4]; label="XPU parallel", opts...))
if save_fig savefig("perf_gpu.png") end
end
