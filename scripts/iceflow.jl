# Shallow ice approximation (SIA) implicit solver for Greenland (steady state)

using JLD, Plots, Printf, LinearAlgebra

# enable plotting & saving by default
if !@isdefined do_visu; do_visu = true end
if !@isdefined do_save; do_save = true end

# finite difference stencil operation support functions
@views av(A)    = 0.25*(A[1:end-1,1:end-1].+A[2:end,1:end-1].+A[1:end-1,2:end].+A[2:end,2:end]) # average
@views av_xa(A) = 0.5.*(A[1:end-1,:].+A[2:end,:]) # average x-dir
@views av_ya(A) = 0.5.*(A[:,1:end-1].+A[:,2:end]) # average y-dir
@views inn(A)   = A[2:end-1,2:end-1] # inner points

@views function iceflow(dx, dy, Zbed, Hice, Mask, grad_b, z_ELA, b_max)
    println("Initialising ice flow model ... ")
    # physics
    s2y      = 3600*24*365.25  # seconds to years
    rho_i    = 910.0           # ice density
    g        = 9.81            # gravity acceleration
    npow     = 3.0             # Glen's power law exponent
    a0       = 1.5e-24         # Glen's law enhancement term
    # numerics
    @assert (dx>0 && dy>0) "dx and dy need to be positive"
    nx, ny   = size(Zbed,1), size(Zbed,2) # numerical grid resolution
    @assert (nx, ny) == size(Zbed) == size(Hice) == size(Mask) "Sizes don't match"
    itMax    = 1e5             # number of iteration (max)
    nout     = 200             # error check frequency
    tolnl    = 1e-6            # nonlinear tolerance
    epsi     = 1e-4            # small number
    damp     = 0.85            # convergence accelerator (this is a tuning parameter, dependent on e.g. grid resolution)
    dtausc   = 1.0/3.0         # iterative dtau scaling
    # derived physics
    a        = 2.0*a0/(npow+2)*(rho_i*g)^npow*s2y
    # derived numerics
    cfl      = max(dx^2,dy^2)/4.1
    # array initialization
    Err      = zeros(nx  , ny  )
    dSdx     = zeros(nx-1, ny  )
    dSdy     = zeros(nx  , ny-1)
    gradS    = zeros(nx-1, ny-1)
    D        = zeros(nx-1, ny-1)
    qHx      = zeros(nx-1, ny-2)
    qHy      = zeros(nx-2, ny-1)
    dtau     = zeros(nx-2, ny-2)
    ResH     = zeros(nx-2, ny-2)
    dHdt     = zeros(nx-2, ny-2)
    Vx       = zeros(nx-1, ny-1)
    Vy       = zeros(nx-1, ny-1)
    M        = zeros(nx  , ny  )
    B        = zeros(nx  , ny  )
    H        = zeros(nx  , ny  )
    S        = zeros(nx  , ny  )
    # initial conditions
    B       .= Zbed
    H       .= Hice
    S       .= B .+ H
    # iteration loop
    println(" starting iteration loop:")
    iter = 1; err = 2*tolnl
    while err>tolnl && iter<itMax
        Err   .= H
        # mass balance
        M     .= min.(grad_b.*(S .- z_ELA), b_max)
        # compute diffusivity
        dSdx  .= diff(S, dims=1)/dx
        dSdy  .= diff(S, dims=2)/dy
        gradS .= sqrt.(av_ya(dSdx).^2 .+ av_xa(dSdy).^2)
        D     .= a*av(H).^(npow+2) .* gradS.^(npow-1)
        # compute flux
        qHx   .= .-av_ya(D).*diff(S[:,2:end-1], dims=1)/dx
        qHy   .= .-av_xa(D).*diff(S[2:end-1,:], dims=2)/dy
        # update ice thickness
        dtau  .= dtausc*min.(10.0, cfl./(epsi .+ av(D)))
        ResH  .= .-(diff(qHx, dims=1)/dx .+ diff(qHy, dims=2)/dy) .+ inn(M)
        dHdt  .= dHdt.*damp .+ ResH
        H[2:end-1,2:end-1] .= max.(0.0, inn(H) .+ dtau.*dHdt)
        # apply mask (a very poor-man's calving law)
        H[Mask.==0] .= 0.0
        # update surface
        S     .= B .+ H
        # error check
        if mod(iter, nout)==0
            Err .= Err .- H
            err = norm(Err)/length(Err)
            @printf(" iter = %d, error = %1.2e \n", iter, err)
            if isnan(err)
                error("""NaNs encountered.  Try a combination of:
                             decreasing `damp` and/or `dtausc`, more smoothing steps""")
            end
        end
        iter += 1
    end
    # compute velocities
    Vx .= -D./(av(H) .+ epsi).*av_ya(dSdx)
    Vy .= -D./(av(H) .+ epsi).*av_xa(dSdy)
    # return as GeoArrays
    return  as_geoarray(H,  Zbed, name=:thickness),
            as_geoarray(S,  Zbed, name=:surface),
            as_geoarray(M,  Zbed, name=:smb),
            as_geoarray(Vx, Zbed, name=:vel_x, staggerd=true),
            as_geoarray(Vy, Zbed, name=:vel_y, staggerd=true)
end
# ------------------------------------------------------------------------------
include(joinpath(@__DIR__, "helpers.jl"))

# load the data
print("Loading the data ... ")
Zbed, Hice, Mask, dx, dy, xc, yc = load_data(; nx=96) # nx=96,160 are included in the repo
                                                      # other numbers will trigger a 2GB download
println("done.")

# apply some smoothing
print("Applying some smoothing ... ")
for is=1:2 # two smoothing steps
    smooth!(Zbed)
    smooth!(Hice)
end
println("done.")

# calculate mass balance coefficients for given spatial grid
grad_b, z_ELA, b_max = mass_balance_constants(xc, yc)

# run the SIA flow model
H, S, M, Vx, Vy = iceflow(dx, dy, Zbed, Hice, Mask, grad_b, z_ELA, b_max)


# visualization and save
nx, ny = size(H)
if do_visu
    !ispath("../output") && mkdir("../output")

    H_v = copy(H); H_v[Mask.==0].=NaN
    Hice_v = copy(Hice); Hice_v[Mask.==0].=NaN
    S_v = copy(S); S_v[Mask.==0].=NaN
    M_v = copy(M); M_v[Mask.==0].=NaN
    V_v = sqrt.(Vx.^2 .+ Vy.^2)

    # outputs
    fontsize  = 7
    opts = (aspect_ratio=1, yaxis=font(fontsize, "Courier"), xaxis=font(fontsize, "Courier"),
            ticks=nothing, framestyle=:box, titlefontsize=fontsize, titlefont="Courier", colorbar_title="",
            xlabel="", ylabel="", xlims=(dims(H_v)[1][1],dims(H_v)[1][end]), ylims=(dims(H_v)[2][end],dims(H_v)[2][1]) )
    p1 = heatmap(S_v; c=:davos, title="Surface elev. [m]", opts...)
    p2 = heatmap(H_v; c=:davos, title="Ice thickness [m]", opts...)
    p3 = heatmap(log10.(V_v); clims=(0.1, 2.0), title="log10(vel) [m/yr]", opts...)
    p4 = heatmap(M_v; c=:devon, title="Mass Bal. rate [m/yr]", opts...)
    p = plot(p1, p2, p3, p4, size=(400,400), dpi=200) #background_color=:transparent, foreground_color=:white
    ## uncomment if you want a pop-up plot pane showing:
    # display(p)
    savefig("../output/iceflow_out1.png")

    # error
    H_diff = Hice_v.-H_v
    fontsize = 7
    p1 = heatmap(Hice_v; c=:davos, title="H data [m]", opts...)
    p2 = heatmap(H_v; c=:davos, title="H model [m]", opts...)
    clim = max(abs.(extrema(H_diff[.!isnan.(H_diff)]))...)
    p3 = heatmap(H_diff; title="H (data-model) [m]",  clims=(-clim,clim), seriescolor=:balance, opts...)
    p = plot(p1, p2, p3, layout=(1, 3), size=(500,160), dpi=200) #background_color=:transparent, foreground_color=:white
    ## uncomment if you want a pop-up plot pane showing:
    # display(p)
    savefig("../output/iceflow_out2.png")
end

if do_save
    save("../output/iceflow_$(nx)x$(ny).jld", "Hice", Hice,
                                              "Mask", Mask,
                                              "H"   , H,
                                              "S"   , S,
                                              "M"   , M,
                                              "Vx"  , Vx,
                                              "Vy"  , Vy,
                                              "xc", xc, "yc", yc)
end

println("... done.")
