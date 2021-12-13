using AbstractPlotting
using GLMakie # openGL backend, uses GPu, very interactive, runs outside browser
# using WGLMakie # WebGL backend, interactive, runs inside browser
# using Cairo # 2d plots
using AbstractPlotting.MakieLayout

# Plots latitudes and longitude of sphere.
function plotLatLon3d!(
        scene :: Union{Scene,LScene};
        lat_num :: Int64 = 10, # number of latitudes
        lon_num :: Int64 = 20, # number of longitude
        lat_den :: Int64 = 100, # number of points per latitude, den for density
        lon_den :: Int64 = 50 # number of points per longitude
) :: Nothing
    lats = [ [ sphericalToCartesian( SphericalCoord(theta, phi) ) for phi in LinRange(0, 2π, lat_den) ] for theta in LinRange(0, π, lat_num)]
    lons = [ [ sphericalToCartesian( SphericalCoord(theta, phi) ) for theta in LinRange(0, π, lon_den)] for phi in LinRange(0, 2π, lon_num)]

    # plot each line separately because plotting library connects consecutive ponts in the array in second argument using a line
    for l=lons
       lines!(scene, [l[j].x for j in 1:lon_den], [l[j].y for j in 1:lon_den], [l[j].z for j in 1:lon_den])
    end

    for l=lats
       lines!(scene, [l[j].x for j in 1:lat_den], [l[j].y for j in 1:lat_den], [l[j].z for j in 1:lat_den])
    end
    nothing
end

# Plots latitudes and longitude of sphere.
function plotLatLon2d!(
        scene :: Union{Scene,LScene};
        lat_num :: Int64 = 10, # number of latitudes
        lon_num :: Int64 = 20, # number of longitude
        lat_den :: Int64 = 100, # number of points per latitude, den for density
        lon_den :: Int64 = 50 # number of points per longitude
) :: Nothing
    lats = [ [ SphericalCoord(theta, phi) for phi in LinRange(0, 2π, lat_den) ] for theta in LinRange(0, π, lat_num)]
    lons = [ [ SphericalCoord(theta, phi) for theta in LinRange(0, π, lon_den)] for phi in LinRange(0, 2π, lon_num)]

    # plot each line separately because plotting library connects consecutive ponts in the array in second argument using a line
    for l=lons
       lines!(scene, [l[j].phi for j in 1:lon_den], [l[j].theta for j in 1:lon_den])
    end

    for l=lats
       lines!(scene, [l[j].phi for j in 1:lat_den], [l[j].theta for j in 1:lat_den])
    end
    nothing
end

# convert field specified in spherical coords to normal cartesian coordinates
function cartesianField( sphericalField )
    field = (x,y,z) ->  sphericalToCartesianVelocity(
        cartesianToSpherical( CartesianCoord( x,y,z ) ),
        sphericalField( cartesianToSpherical( CartesianCoord( x,y,z ) ) )
    )
    return( field )
end

function plotVelocityField3d!( threeDPlot::Union{Scene,LScene}, fieldGenerator, objects::Vector{<:Object}, lat_den::Int64, lon_den::Int64 )::Nothing
    sphericalField = @lift( p -> fieldGenerator( p, $objects ) )
    ∇f = @lift( cartesianField( $sphericalField ) )

    # create arrows plot for velocity vector field on a lot long grid
    gx = reduce(vcat,[[cospi(φ)*sinpi(θ) for φ in LinRange(0, 2π, round( Int, lat_den*sin(θ)))] for θ in LinRange(0, π, lon_den) ])
    gy = reduce(vcat,[[sinpi(φ)*sinpi(θ) for φ in LinRange(0, 2π, round( Int, lat_den*sin(θ)))] for θ in LinRange(0, π, lon_den) ])
    gz = reduce(vcat,[[cospi(θ)          for φ in LinRange(0, 2π, round( Int, lat_den*sin(θ)))] for θ in LinRange(0, π, lon_den) ])

    # we normalize the vectors because vectors can be too large at some places messing the entire plot
    n∇fxyz = @lift( map((x,y,z) -> normalize($∇f(x,y,z)), gx,gy,gz) )
    vx = @lift( map(v->v.x, $n∇fxyz) )
    vy = @lift( map(v->v.y, $n∇fxyz) )
    vz = @lift( map(v->v.z, $n∇fxyz) )
    arrows!(threeDPlot, gx, gy, gz, vx, vy, vz, arrowsize = 0.01, lengthscale = 0.05 )

    φs = LinRange(0, 2π, lat_den)
    θs = LinRange(0, π, lon_den)

    # plot the magnitude of velocity vector field on spherical surface
    sphereGrid = [ sphericalToCartesian( SphericalCoord(theta, phi) ) for phi in φs, theta in θs ]
    # TODO: we are recomputing ∇f on the sphere grid again, we already did that for a 'similar' grid when making arrows, can we reuse those computations
    colorsGrid = @lift( map( p->log(1 + norm($∇f(p.x,p.y,p.z))), sphereGrid ) )
    AbstractPlotting.:surface!(threeDPlot, map(s->s.x,sphereGrid), map(s->s.y,sphereGrid), map(s->s.z,sphereGrid), color=colorsGrid, colormap=:dense )
    nothing
end

function plotVelocityField2d!( twoDPlot::Union{Scene,LScene}, fieldGenerator, objects::Vector{<:Object}, lat_den::Int64, lon_den::Int64 )::Nothing
    # now we plot streams on theta-phi chart
    # phi is x axis and theta is y axis
    sphereCoordToPoint2f(vector::SphericalCoord, position::SphericalCoord ) = Point2f0( csc( position.theta)*vector.phi, vector.theta )
    # velocity func in polar coordinates
    # this gives (dϕ/dt, dθ/dt)
    ∇f_polar = @lift( (phi,theta) -> sphereCoordToPoint2f(
        fieldGenerator(
            SphericalCoord(theta,phi),
            $objects
        ),
        SphericalCoord(theta, phi)
    ) )

    φs = LinRange(0, 2π, lat_den)
    θs = LinRange(0, π, lon_den)

    # plot streams on theta phi chart
    streamplot!( twoDPlot, ∇f_polar, φs, θs, arrow_size=0.1 ) # streamplot is computationally expensive

    # plot vector magnitude
    # we don't need to use spherical coordinate norm as the spherical vector already include sin(theta) in e_phi component
    # TODO: we are recomputing ∇f on the sphere grid again, we already did this when making density surface plot on 3d chart, we can reuse the computations
    # TODO: in here vector norm should be on (sin(θ) dϕ/dt, dθ/dt), rather than (dϕ/dt, dθ/dt)
    colorsGridTwoD = @lift( [ log(1+norm($∇f_polar(phi,theta))) for phi in φs, theta in θs ] )
    AbstractPlotting.:heatmap!(twoDPlot, φs, θs, colorsGridTwoD, colormap=:dense)
    nothing
end


# Create a slider animation.
function animate(
    simObjects :: Vector{<:Vector{<:Object}}; #array of all time simuatled array of objects
    plotLatLon :: Bool = true,
    fieldGenerator = nothing, # a (point::SphericalCoord, objects::Vector{Object}} -> vector::SphericalCoord) function, if want to plot arrows and density on sphere
    resolution :: Tuple{Int64,Int64} = (1000,600),
    makeMovie :: Bool = true,
    cameraPosition :: Vec{3,Float32} = Vec3f0(-2, 0, 0),
    cameraLookingAt :: Vec{3,Float32} = Vec3f0(-1, 0, 0),
    frameRate ::Int64 = 1,
    movieName ::String = "animation.mp4",
) :: Nothing
    numFrames = length(simObjects)
    scene, layout = layoutscene(resolution = resolution) # create a new scene

    # a slider to move forward or backward in animation
    frameSlider = layout[1,:] = Slider(scene, range = LinRange(1, numFrames, numFrames), startvalue = 1)

    threeDPlot = layout[2,1] = LScene(scene) # full interactive shphere plot
    twoDPlot = layout[2,2] = LScene(scene) # theta phi chart

    AbstractPlotting.cam3d!(threeDPlot.scene)
    update_cam!( threeDPlot.scene, cameraPosition, cameraLookingAt )

    # create an observable state which the plots will 'watch'
    # later we will simply update the state and scene will be automatically updated
    # vortexState is dependent on slider so we 'lift' it to an observable
    sliderVal = frameSlider.value
    simObjectstate = @lift( simObjects[round( Int, $sliderVal)] )

    lat_den = 100 # number of points per latitude (= number of longitudes for grid below)
    lon_den = 50 # number of points per longitude (=number of longitude for grid below)

    if plotLatLon
        plotLatLon3d!(threeDPlot, lat_den = lat_den, lon_den = lon_den) # plot latitude and longitude
        plotLatLon2d!(twoDPlot, lat_den = lat_den, lon_den = lon_den)
    end

    # plot arrows and density for the fields if the fieldGenerator is provided
    if fieldGenerator != nothing
        plotVelocityField3d!( threeDPlot, fieldGenerator, simObjectstate, lat_den, lon_den )
        plotVelocityField2d!( twoDPlot, fieldGenerator, simObjectstate, lat_den, lon_den )
    else
        φs = LinRange(0, 2π, lat_den)
        θs = LinRange(0, π, lon_den)
        sphereGrid = [ sphericalToCartesian( SphericalCoord(theta, phi) ) for phi in φs, theta in θs ]
        colorsGrid =  map( p->2, sphereGrid )
        AbstractPlotting.:surface!(threeDPlot, map(s->s.x,sphereGrid), map(s->s.y,sphereGrid), map(s->s.z,sphereGrid), color=colorsGrid, colormap=:dense )
    end

    # plot all simObjects
    for i in 1:length(simObjects[1])
        o = @lift( $simObjectstate[i] )
        plot3d!( threeDPlot, o )
        plot2d!( twoDPlot, o )
    end

    if makeMovie
        # create animation: update the state obervable node by moving slider, record, loop
        record(scene, movieName, 1:numFrames; framerate = frameRate) do frameNum
            set_close_to!(frameSlider, frameNum)
        end
    end

    # create an interactive simulation controlled by slider
    RecordEvents(scene, "output")
    nothing
end
