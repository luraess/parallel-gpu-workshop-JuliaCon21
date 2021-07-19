# Compare the 2D explicit and implicit nonlinear diffusion solver results
using Plots

include("diffusion_2D_expl.jl")
include("diffusion_2D_damp.jl")

save_fig=false

@views function compare_expl_impl(; save_fig=false)
    # run the codes
    do_visu = false
    xc, yc, H_expl = diffusion_2D_expl(; do_visu=do_visu)
    xc, yc, H_damp = diffusion_2D_damp(; do_visu=do_visu)
    # extract ∆
    ∆_expl_impl = 100.0*(H_expl .- H_damp)./H_damp
    ∆max = maximum(abs.(∆_expl_impl))
    # visualise
    fontsize = 12
    opts = (aspect_ratio=1, yaxis=font(fontsize, "Courier"), xaxis=font(fontsize, "Courier"),
            ticks=nothing, framestyle=:box, titlefontsize=fontsize, titlefont="Courier", colorbar_title="",
            xlabel="Lx", ylabel="Ly", xlims=(xc[1],xc[end]), ylims=(yc[1],yc[end]), clims=(-∆max, ∆max))
    display(heatmap(xc, yc, ∆_expl_impl; seriescolor=:balance, title="∆(expl vs impl) in %", opts...))
    if save_fig savefig("diff2D_expl_impl.png") end
    return
end

compare_expl_impl(; save_fig=save_fig)
