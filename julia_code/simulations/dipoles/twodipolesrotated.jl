include("../../geometry/sphere.jl")
include("../../physics/fluid.jl")
include("../../physics/dipoleforce.jl")
include("../../utils/simulator.jl")
include("../../utils/plot.jl")

fluid = FluidParams( 1. )

rs = 0.1

# two dipoles perpendicular to each other with orientations parallel and perpendicular to geodesic
# cot θ = a cos(φ − φ0) with a = 0 and ϕ0 = π/2 i.e. cot θ = cos(φ − π/2)
d2θ = π/2 + 0.1
d2ϕ = acos(cot(d2θ)) + π/2
d2α = atan(sin(d2θ)*sin(d2ϕ - π/2)) + pi/2 # perpendicular to tangent to geodesic ( 1/sin(θ)*(dθ/dϕ) + π/2 )
dipoles = [
    DipoleForce( SphericalCoord(π/2, π), π/4, 1 ),
    DipoleForce( SphericalCoord(d2θ, d2ϕ), d2α, 1 )
]

# Generating a frame is computationally expensive,
# but each step forward in differential equation is computationally cheap.
# Higher steps reduce errors in numerical integration.
numFrames = 500 # number of frames in animation,
numSteps = 10000 # number of steps forward in each frame,
endTime = 5 # time till when we want to simulate
adaptivePosMove = rs/10

# do the simulation
simulatedObjects = [ dipoles ]

for i in 2:numFrames
    newState = stepForward( last(simulatedObjects), fluid, endTime/numFrames/numSteps, numSteps, adaptivePosMove)
    nvortices = newState
    append!(simulatedObjects, [ newState ] )
    print(i, "\n")
end

# do the plitting/animation
animate(
    simulatedObjects,
    plotLatLon = true,
    # fieldGenerator = (p, vortices, pointforces, dipoles) -> sum( map( s -> velocityField(p, s, fluid), vcat( vortices, pointforces, dipoles ) ) ),
    resolution = (1000,600),
    makeMovie = true,
    cameraPosition = Vec3f0(-2, 0, 0),
    cameraLookingAt = Vec3f0(-1, 0, 0),
)
