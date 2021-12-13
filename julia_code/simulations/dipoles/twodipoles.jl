include("../../geometry/sphere.jl")
include("../../physics/fluid.jl")
include("../../physics/dipoleforce.jl")
include("../../utils/simulator.jl")
include("../../utils/plot.jl")

fluid = FluidParams( 1. )

rs = 0.1

dipoles = [
    DipoleForce( SphericalCoord( π/2 - 0.2, π ), 0, 1 ),
    DipoleForce( SphericalCoord( π/2, π), π/2, 1)
]

# Generating a frame is computationally expensive,
# but each step forward in differential equation is computationally cheap.
# Higher steps reduce errors in numerical integration.
numFrames = 10 # number of frames in animation,
numSteps = 100 # number of steps forward in each frame,
endTime = 1 # time till when we want to simulate
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
