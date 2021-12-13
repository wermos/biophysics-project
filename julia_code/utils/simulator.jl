using DifferentialEquations
using Setfield

# Do the simulation taking the VortexMotion state
function stepForward(
    objects:: Vector{<:Object}, # array of all objects
    fluid :: FluidParams,
    dt :: Float64, # time in each step
    n :: Int,  # number of steps
    maxPosChange :: Float64, # adapt dt so that position change doesn't exceed this in single time step
) :: Vector{<:Object}
    for j in 1:n
        remainingDt = dt

        while remainingDt > 0
            velocity( d ) = sum( map( s -> velocityField(d, s, fluid), objects ) )
            # velocityGradient( d ) = sum( map( s -> velocityGradientField(d, s, fluid), objects ) )
            velocityCurl( d ) = sum( map( s -> velocityCurlField(d, s, fluid), objects ) )

            vObjects = map( o -> velocity(o.location), objects ) # velocities that objects experience
            # vgObjects = map( o -> velocityGradient(o.location), objects ) # velocity gradients that objects experience
            vcObjects = map( o -> velocityCurl(o.location), objects ) # velocity curls that objects experience

            normVector( s ) = norm( [ s.theta, s.phi ] )
            maxV = maximum( map( normVector, vObjects ) )
            dt = min( remainingDt, maxPosChange / maxV )

            objects = map( (v,vv,vcv) -> stepForward(v, vv, vcv, dt), objects, vObjects, vcObjects )
            remainingDt -= dt
        end
    end

    return( objects )
end



# Do the simulation taking the VortexMotion state
function simulate(
    objects :: Vector{<:Object}, # array of all objects
    fluid :: FluidParams,
    endTime :: Float64, # time in each step
    saveAtTime :: Float64,  # number of steps
    maxPosChange :: Float64, # adapt dt so that position change doesn't exceed this in single time step
) :: Tuple{Vector{Float64},<:Vector{<:Object}}
    # TODO: add dynamics for dipole orientation

    n = length(objects) # number of particles

    function newLocation( o, loc )
        no = @set o.location = loc
    end

    locsToCoords(l) = [ SphericalCoord(l[2*i-1], l[2*i]) for i in 1:n ]

    function dynamics!(du,u,p,t)
        @assert n == round(Int, length(u)/2) # number of particles

        nObjects = map( newLocation, objects, locsToCoords(u) )
        velocity( d ) = sum( map( s -> velocityField(d, s, fluid), nObjects ) )

        vObjects = map( o -> velocity(o.location), nObjects ) # velocities that object experience

        newDU = reduce(
            vcat,
            map(
                (v,o) -> [v.theta, csc(o.location.theta)*v.phi],
                vObjects,
                nObjects
            )
        )
        for i in 1:(2*n)
            du[i] = newDU[i]
        end
    end

    u0 = reduce( vcat, map( o -> [ o.location.theta, o.location.phi], objects ) )
    tSpan = (0., endTime)
    prob = ODEProblem( dynamics!, u0, tSpan )
    sol = solve( prob, AutoVern7(Rodas5()), saveat=saveAtTime, reltol=maxPosChange, abstol=maxPosChange )

    simObjects = map(
        locs -> map( newLocation, objects, locsToCoords(locs) ),
        sol.u
    )

    return( sol.t, simObjects )
end
