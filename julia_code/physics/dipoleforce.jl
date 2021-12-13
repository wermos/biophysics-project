include("../geometry/sphere.jl")
include("./fluid.jl")
include("../utils/data.jl")

using Symbolics
using DataFrames
using CSV
using Interpolations
using GLMakie


#=
(θ_0,ϕ_0) is the location of the dipole
(θ,ϕ) is the location of the observation point
κ is the dipole strengh = Force * dipole length
α is the orientation angle of dipole from ∂ϕ_0
β is the angle from ∂ϕ along which we measure the component of velocity
dl is the length if dipole used in some tests but finally we take dl -> 0, so this should not appear in final functions
=#
@variables θ_0 ϕ_0 θ ϕ κ α β dl

#=
Starting from pointforce, dipole is two pointforces placed at small distance opposite to each other
and oriented along the tangents to the geodesic connecting them.

We consider the limit dipole length dl -> 0 such that F.dl = κ

v_t =   \gradient_{θ,ϕ} ( (force_t + force_t ⋅ [dθ_0, sin(θ_0)dϕ_0] ⋅ covd )  ⋅ \gradient_{θ_0+dθ_0, ϕ_0+dϕ_0} P(γ(θ,ϕ,θ_0+dθ_0,ϕ_0+dϕ_0)) )
        - \gradient_{θ,ϕ} ( force_t ⋅ \gradient_{θ_0,ϕ_0} P(γ) )

       = \gradient_{θ,ϕ} ( (force_t + force_t ⋅ [dθ_0, sin(θ_0)dϕ_0] ⋅ covd ) ⋅ ( ( 1 + [ dθ_0, dϕ_0] * [∂θ_0, ∂ϕ_0]' ) ( \gradient_{θ_0, ϕ_0} P(γ) ) ) )
       - \gradient_{θ,ϕ} ( force_t ⋅ \gradient_{θ_0,ϕ_0} P(γ) )

       = \gradient_{θ,ϕ} ( (force_t + force_t ⋅ [dθ_0, sin(θ_0)dϕ_0] ⋅ covd ) ⋅ ( ( 1 + [ dθ_0, dϕ_0] * [∂θ_0, ∂ϕ_0]' ) ( \gradient_{θ_0, ϕ_0} P(γ) ) ) )
                           - force_t ⋅ \gradient_{θ_0,ϕ_0} P(γ) )

       = \gradient_{θ,ϕ} (
              force_t ⋅ ( [∂dθ_0, (csc(θ_0)∂dϕ_0]' P(γ) )  // 000 -> cancels with last term
            + force_t ⋅ ( ( [ dθ_0, dϕ_0] * [∂θ_0, ∂ϕ_0]' ) ( \gradient_{θ_0, ϕ_0} P(γ) ) ) /// 01
            + force_t ⋅ [dθ_0, sin(θ_0)dϕ_0] ⋅ covd ⋅ ( \gradient_{θ_0, ϕ_0} P(γ) )  // 10
            + force_t ⋅ ( ( [ dθ_0, dϕ_0] * [∂θ_0, ∂ϕ_0]' ) ( \gradient_{θ_0, ϕ_0} P(γ) ) ) /// 11 -> second order -> 0
            - force_t ⋅ \gradient_{θ_0,ϕ_0} P(γ) -> cancels with first term
        )

        = \gradient_{θ,ϕ} ( force_t ⋅ ( ( [ dθ_0, dϕ_0] * [∂θ_0, ∂ϕ_0]' ) ( \gradient_{θ_0, ϕ_0} P(γ) ) ) /// 01
                            + force_t ⋅ [dθ_0, sin(θ_0)dϕ_0] ⋅ covd force_t ⋅ ( \gradient_{θ_0, ϕ_0} P(γ) )   // 10
                          )

dϕ_0 = csc(θ_0)cos(α)dl
dθ_0 = sin(α)dl

        = \gradient_{θ,ϕ} ( force_t ⋅  ( ( [ sin(α)dl, cos(α)csc(θ_0)dl] * [∂θ_0, ∂ϕ_0]' ) ( \gradient_{θ_0, ϕ_0} P(γ) ) ) /// 01
                            + force_t ⋅ [sin(α)dl, cos(α)dl] ⋅ covd ⋅ ( [∂θ_0, (csc(θ_0)∂ϕ_0]' P(γ) )  // 10
                          )

      = \gradient_{θ,ϕ} ( force_t ⋅ ( ( [ sin(α)dl, cos(α)dl] * [∂θ_0, csc(θ_0)∂ϕ_0]' ) ( \gradient_{θ_0, ϕ_0} P(γ) ) ) /// 01
                          + force_t ⋅ [sin(α)dl, cos(α)dl] ⋅ covd ⋅ ( [∂θ_0, (csc(θ_0)∂ϕ_0]' P(γ) )  // 10
                        )

      =  \gradient_{θ,ϕ} ( force_t ⋅ ( ( dl_vector ⋅ \gradient_{θ_0, ϕ_0} ) ( \gradient_{θ_0, ϕ_0} P(γ) ) ) /// 01
                          + (force_t ⋅ dl_vector ⋅ covd ) ⋅ ( \gradient_{θ_0,ϕ_0} P(γ) )  // 10
                        )

      =  \gradient_{θ,ϕ} ( force ⋅ twist ⋅ ( ( dl_vector ⋅ \gradient_{θ_0, ϕ_0} ) ( \gradient_{θ_0, ϕ_0} P(γ) ) ) /// 01
                        + (force ⋅ dl_vector ⋅ covd ) ⋅ twist  ( \gradient_{θ_0,ϕ_0} P(γ) )  // 10
                      )

αvec = [sin(α), cos(α)]

         = κ \gradient_{θ,ϕ} ( αvec ⋅ twist ⋅ ( ( dl_vector / dl ⋅ \gradient_{θ_0, ϕ_0} ) ( \gradient_{θ_0, ϕ_0} P(γ) ) ) /// 001
                         + (αvec ⋅ dl_vector / dl ⋅ covd ) ⋅ twist ⋅ ( \gradient_{θ_0,ϕ_0} P(γ) )  // 100
                       )

αvec ⋅ dl_vector ⋅ covd is change in  αvec due to parallel transport from (θ_0,ϕ_0) to (θ_0+sin(α)dl,ϕ_0+cos(α)csc(θ_0)dl)

rotation angle is ψ i.e. this is the angle by which the tangent to the geodesic moves
ψ = ψ_coeff * dl
ψ_coeff = cos(α)*cot(θ_0)

rotation matrix rot for [θ,ϕ] vector is [ cos(ψ) sin(ψ); -sin(ψ) cos(ψ) ]
rotation matrix rot_c for [θ,ϕ] co-vector is [cos(ψ) -sin(ψ); sin(ψ) cos(ψ)]
so it's a transpose of the usual rotation matrix
note that angle ψ is from ϕ and not θ

vec ⋅ dl_vector ⋅ covd = vec ⋅ rot_c - vec = vec (rot_c - 1)
dl_vector ⋅ covd  = rot_c - 1
                  = [cos(ψ)-1 -sin(ψ); sin(ψ) cos(ψ)-1]
                  = [0 -ψ_coeff*dl; ψ_coeff*dl 0]

dl_vector / dl ⋅ covd = [0 -ψ_coeff; ψ_coeff 0] = [0 ψ_coeff; -ψ_coeff 0]' = drot' = drot_c

        = κ \gradient_{θ,ϕ} ( αvec ⋅ twist ⋅ ( ( dl_vector / dl ⋅ \gradient_{θ_0, ϕ_0} ) ( \gradient_{θ_0, ϕ_0} P(γ) ) ) /// 001
                        + (αvec ⋅ [0 -ψ_coeff; ψ_coeff 0] ) ⋅ twist ⋅ ( \gradient_{θ_0,ϕ_0} P(γ) )  // 100
              )

dl_vector / dl = αvec
=#

∂θ = Differential(θ)
∂ϕ = Differential(ϕ)
∂θ_0 = Differential(θ_0)
∂ϕ_0 = Differential(ϕ_0)

grad_θϕ( f ) = [ ∂θ(f) csc(θ)*∂ϕ(f) ]'
grad_θ_0ϕ_0( f ) = [ ∂θ_0(f) csc(θ_0)*∂ϕ_0(f) ]'

γ = acos( cos(θ)*cos(θ_0) + sin(θ)*sin(θ_0)*cos(ϕ - ϕ_0) )
∂γ = Differential(γ)

twist = [ 0   1; -1  0 ]
tgrad_θϕ( f ) = twist * grad_θϕ(f)
tgrad_θ_0ϕ_0( f ) = twist * grad_θ_0ϕ_0(f)

αvec = [sin(α) cos(α)]

ψ_coeff = cos(α)*cot(θ_0)
drot = [0 ψ_coeff; -ψ_coeff 0]


pData = DataFrame(CSV.File("./data/pointforce_import_lc.tsv"))
pDataInterpolated = LinearInterpolation( pData.x, pData.y; extrapolation_bc = Line())
pDataInterpolated1 = dataDerivative( ( pDataInterpolated, pData.x ) )[1]
pDataInterpolated2 = dataDerivative( ( pDataInterpolated1, pData.x ) )[1]
pDataInterpolated3 = dataDerivative( ( pDataInterpolated2, pData.x ) )[1]
pDataInterpolated4 = dataDerivative( ( pDataInterpolated3, pData.x ) )[1]
pDataInterpolated5 = dataDerivative( ( pDataInterpolated4, pData.x ) )[1]

function Pf(n::Int64, x::Float64)
  if n==0
    pDataInterpolated(x)
  elseif n==1
    pDataInterpolated1(x)
  elseif n==2
    pDataInterpolated2(x)
  elseif n==3
    pDataInterpolated3(x)
  elseif n==4
    pDataInterpolated4(x)
  elseif n==5
    pDataInterpolated5(x)
  else
    throw( sprint(n) + "-derivative of P not precomputed" )
  end
end

Symbolics.derivative( ::typeof(Pf), args::NTuple{2,Any}, ::Val{2} ) = SymbolicUtils.Term( Pf, [ args[1]+1, args[2] ] )
Symbolics.@register Pf( data::Int64, x )

P = Pf( 0, γ )

v = κ * tgrad_θϕ( ( αvec * twist * (sin(α)*( ∂θ_0.( grad_θ_0ϕ_0( P ) ) ) + csc(θ_0)cos(α)*( ∂ϕ_0.( grad_θ_0ϕ_0( P ) ) ) ) )[ 1 ] +
                  ( (αvec * drot' ) * tgrad_θ_0ϕ_0( P ) )[ 1 ]
                )

# gradient matrix of v at (θ,ϕ)
#vg = [ ∂θ(v_θ) csc(θ)∂ϕ(v_θ); ∂θ(v_ϕ) csc(θ)∂ϕ(v_ϕ)]
# vg = [ ∂θ(v[1]) csc(θ)*∂ϕ(v[1]); ∂θ(v[2]) csc(θ)*∂ϕ(v[2]) ]

# curl of v at (θ,ϕ)
# vc = (1/sin(θ))( ∂θ(v_ϕ sin(θ)) - ∂ϕ(v_θ) )
vc = 1/sin(θ) * ( ∂θ(v[2]*sin(θ)) - ∂ϕ(v[1]) )

v_simplified = simplify.( expand_derivatives.( v ) )
# vg_simplified = simplify.( expand_derivatives.( vg ) )
vc_simplified = simplify( expand_derivatives( vc ) )


dipoleForceV = build_function( v_simplified, [θ_0, ϕ_0, θ, ϕ, κ, α ], expression=Val{false} )
# dipoleForceVG = build_function( vg_simplified, [θ_0, ϕ_0, θ, ϕ, κ, α ], expression=Val{false} )
dipoleForceVC = build_function( vc_simplified, [θ_0, ϕ_0, θ, ϕ, κ, α ], expression=Val{false} )


# v_th = ∂γ( P ) * (   f * sin(α) * csc(θ) * csc(θ_0) * ∂ϕ( ∂ϕ_0(γ) )  -  f * cos(α) * csc(θ) * ∂ϕ( ∂θ_0(γ) ) )
# v_ph = ∂γ( P ) * ( - f * sin(α)          * csc(θ_0) * ∂θ( ∂ϕ_0(γ) )  +  f * cos(α) *          ∂θ( ∂θ_0(γ) ) )

#========================================
# Interface functions
========================================#

struct DipoleForce <: Object
    location :: SphericalCoord  # (θ_0,ϕ_0) location of
    orientation :: Float64  # angle that the dipole makes with ∂/∂ϕ_0, note orientation is same as orientation + π
    magnitude :: Float64  # strength of the dipole, dipole length * Force
end

# velocity field generated by dipole force
function velocityField( destination::SphericalCoord, source::DipoleForce, fluid::FluidParams )::SphericalCoord
    θs = source.location.theta
    ϕs = source.location.phi
    θd = destination.theta
    ϕd = destination.phi
    if θs == θd && ϕs == ϕd
        return( SphericalCoord( 0, 0 ) )
    else
        velocity = dipoleForceV[1]([ θs, ϕs, θd, ϕd, source.magnitude, source.orientation ])
        return( SphericalCoord( velocity[ 1 ], velocity[ 2 ] ) )
    end
end
#
# # gradient of velocity field generated by dipole force
# function velocityGradientField( destination::SphericalCoord, source::DipoleForce, fluid::FluidParams )::Array{Float64,2}
#     θs = source.location.theta
#     ϕs = source.location.phi
#     θd = destination.theta
#     ϕd = destination.phi
#     if θs == θd && ϕs == ϕd
#         return( [0. 0.; 0. 0.] )
#     else
#         velocityGrad = dipoleForceVG[1]([ θs, ϕs, θd, ϕd, source.magnitude, source.orientation ])
#         return( velocityGrad )
#     end
# end

# gradient of velocity field generated by dipole force
function velocityCurlField( destination::SphericalCoord, source::DipoleForce, fluid::FluidParams )::Float64
    θs = source.location.theta
    ϕs = source.location.phi
    θd = destination.theta
    ϕd = destination.phi
    if θs == θd && ϕs == ϕd
        return( 0 )
    else
        velocityCurl = dipoleForceVC([ θs, ϕs, θd, ϕd, source.magnitude, source.orientation ])
        return( velocityCurl )
    end
end

# how dipole moves in the presence of external velocity field
function stepForward( object::DipoleForce, v::SphericalCoord, vc::Float64, dt::Float64 )::DipoleForce
    # velocity vector is (dtheta/dt, sin(theta)dphi/dt)
    dθ = v.theta * dt
    dϕ = csc( object.location.theta ) * v.phi * dt

    #=
    METHOD 1. assuming a dipole is two inertial masses attached with a massless rod
    TODO: add rotation due to curvature as in method 2

    dα =  tangent unit at (θ_0+sin(α)dl, ϕ_0+cos(α)csc(θ_0)dl) x velocity at (θ_0+sin(α)dl, ϕ_0+cos(α)csc(θ_0)dl) / dl
        + tangent unit at (θ_0, ϕ_0) x velocity at (θ_0, ϕ_0) / dl
        where tangents are wrt to geodesic connecting (θ_0+sin(α)dl, ϕ_0+cos(α)csc(θ_0)dl) and (θ_0, ϕ_0)

        = rotation * αvec' x v(θ_0+sin(α)dl, ϕ_0+cos(α)csc(θ_0)dl)/dl - αvec' x v(θ_0, ϕ_0)/dl

        rotation matrix rot for [θ,ϕ] vector is [ cos(ψ) sin(ψ); -sin(ψ) cos(ψ) ]
        ψ = ψ_coeff * dl
        ψ_coeff = cos(α)*cot(θ_0)

        = [cos(ψ_coeff*dl) sin(ψ_coeff*dl); -sin(ψ_coeff*dl) cos(ψ_coeff*dl)] * αvec' x ( v + ( dl_vector ⋅ \gradient_{θ_0, ϕ_0} )(v) ) / dl - αvec' x v(θ_0, ϕ_0)/dl
        ~ αvec' x v(θ_0, ϕ_0)/dl + [0 sin(ψ_coeff*dl); -sin(ψ_coeff*dl) 0] * αvec' x v(θ_0, ϕ_0) / dl + αvec' x ( ( dl_vector ⋅ \gradient_{θ_0, ϕ_0} )(v) ) / dl - - αvec' x v(θ_0, ϕ_0)/dl
        = [0 sin(ψ_coeff*dl); -sin(ψ_coeff*dl) 0] * αvec' x v(θ_0, ϕ_0) / dl + αvec' x ( ( dl_vector ⋅ \gradient_{θ_0, ϕ_0} )(v) ) / dl
        = [0 ψ_coeff; -ψ_coeff 0] * αvec' x v(θ_0, ϕ_0) + αvec' x ( ( αvec ⋅ \gradient_{θ_0, ϕ_0} )(v) )

    gv = [ ∂θ(v_θ) csc(θ)∂ϕ(v_θ); ∂θ(v_ϕ) csc(θ)∂ϕ(v_ϕ)]
        = [0 ψ_coeff; -ψ_coeff 0] * αvec' x v(θ_0, ϕ_0) + αvec' x (gv * αvec')
    =#

    #=
    METHOD 2. assuming a dipole is a spherical object

    dα = (∇×velocity field)̇ along radial direction * dt + rotation due to curvature as dipole moves
    rotation due to dipole = cos(angle of v) * cot(θ) * distance moved by dipole
                           = (sin(θ)dϕ/dp)* cot(θ) * dp
                           = cos(θ) * dϕ
                           where
                           dp = sqrt( dθ^2 + (sin(θ)*dϕ)^2 )
    =#


    #=
    # Method 1.
    ψ_coeff = cos(object.orientation) * cot(object.location.theta)
    drot = [0 ψ_coeff; -ψ_coeff 0]
    αvec = [sin(object.orientation) cos(object.orientation)]
    # norm of the cross product of two vectors a and b
    crossnorm( a, b ) = norm(a)*norm(b)*sqrt( 1 - ( ( a' * b )[ 1 ] / (norm(a)*norm(b)) )^2 )
    dα = ( crossnorm( drot * αvec', [v.theta v.phi]' ) + crossnorm( αvec', vg * αvec' ) ) * dt
    =#

    # Method 2
    dp = sqrt( dθ^2 + (sin(object.location.theta)*dϕ)^2 ) # distance by which dipole moves
    dα = vc * dt + cos(object.location.theta) * dϕ

    nθ = mod2pi( object.location.theta + dθ )
    nϕ = mod2pi( object.location.phi + dϕ )
    nα = mod2pi( object.orientation + dα )

    # ensure that new theta phi are within coordinate range
    # no need to change orientation of α, orientation is not well defined on poles
    if nθ > pi # object crossed the south pole
        newPhi = mod2pi(nϕ + π)
        newTheta = 2π - nθ
    else
        newPhi = mod2pi(nϕ)
        newTheta = nθ
    end

    newObject = DipoleForce(
        SphericalCoord( newTheta, newPhi ),
        nα,
        object.magnitude
    )
    return( newObject )
end


#plot the dipoles on 3d chart
function plot3d!( threeDPlot::Union{Scene,LScene}, dipole::Observable{DipoleForce} )::Nothing
    dipolePosState = @lift( $dipole.location )
    dipoleVectorMap( f, α ) = SphericalCoord( f*sin(α), f*cos(α) )
    dipoleForceState = @lift( dipoleVectorMap( $dipole.magnitude, $dipole.orientation ) )
    dps = @lift( sphericalToCartesian( $dipolePosState ) )
    dpx = @lift( [$dps.x] )
    dpy = @lift( [$dps.y] )
    dpz = @lift( [$dps.z] )
    dfs = @lift( sphericalToCartesianVelocity( $dipolePosState, $dipoleForceState ) )
    dfx = @lift( [$dfs.x] )
    dfy = @lift( [$dfs.y] )
    dfz = @lift( [$dfs.z] )
    dfxn = @lift( -$dfx ) # for dipoles we also show arrow in the opposite direction
    dfyn = @lift( -$dfy )
    dfzn = @lift( -$dfz )
    arrows!(threeDPlot, dpx, dpy, dpz, dfx, dfy, dfz, arrowsize = 0.05, lengthscale = 0.05 )
    arrows!(threeDPlot, dpx, dpy, dpz, dfxn, dfyn, dfzn, arrowsize = 0.05, lengthscale = 0.05 )
    nothing
end


# show dipoles on theta phi chart
function plot2d!( twoDPlot::Union{Scene,LScene}, dipole::Observable{DipoleForce} )::Nothing
    dipolePosState = @lift( $dipole.location )
    dipoleVectorMap( f, α ) = SphericalCoord( f*sin(α), f*cos(α) )
    dipoleForceState = @lift( dipoleVectorMap( $dipole.magnitude, $dipole.orientation ) )
    dptx = @lift( [$dipolePosState.phi] )
    dpty = @lift( [$dipolePosState.theta] )
    dftx = @lift( [$dipoleForceState.phi*csc($dipolePosState.theta)] )
    dfty = @lift( [$dipoleForceState.theta] )
    dftxn = @lift( -$dftx ) # for dipoles we also show arrow in the opposite direction
    dftyn = @lift( -$dfty )
    arrows!( twoDPlot, dptx, dpty, dftx, dfty, arrowsize = 0.1, lengthscale = 0.05 )
    arrows!( twoDPlot, dptx, dpty, dftxn, dftyn, arrowsize = 0.1, lengthscale = 0.05 )
    nothing
end

#========================================
# Tests
========================================#

#=
Rotational symmetry check

If we have a point force at ( θ0, ϕ0 ) making an angle α with ϕ axis,
then the look at component of velocity induced at point ( θ, ϕ )  making an angle β with ϕ axis

This should be same as component of velocity induced at point ( θ0, ϕ0 ) making an angle β - rot with ϕ axis
by a point force at ( θ, ϕ ) making an angle α + rot

where rot is π + the rotation a vector suffers when parallel transported from ( θ0, ϕ0 ) to ( θ, ϕ )

the operations involved in this symmetry are exchange force and velocity locations,
parallely transporting the force and velocity vectors along the geodesic connecting them,
and then rotating the frame 180 degrees about the diameter of sphere passing through midpoint of geodesic
=#

v_th = v[ 1 ]
v_ph = v[ 2 ]
v_beta = expand_derivatives( v_ph * cos(β) + v_th * sin(β) )

tangent_angle = expand_derivatives( atan( ∂θ(γ), csc(θ)∂ϕ(γ) ) )
tangent_0_angle = expand_derivatives( atan( ∂θ_0(γ), csc(θ_0)∂ϕ_0(γ) ) )
rot = tangent_angle - tangent_0_angle

args = Dict(
      θ => π/2.5
    , ϕ => 0.4
    , θ_0 => 2π/3
    , ϕ_0 => -π/4
    , κ => 1
    , α => 0.2
    , β => 0.4
)

# rotational symmetry check
exchange = Dict( θ_0 => θ, ϕ_0 => ϕ, θ => θ_0, ϕ => ϕ_0, α => α + rot, β => β - rot )
v_ex = substitute( v_beta, exchange )
sym_check = simplify( v_ex - v_beta )  # this should be zero, atleast in domain of interest
sym_check_eval = substitute( sym_check, args ) # this should be zero for all args as sym_check should be zero
@assert abs(sym_check_eval) < 1e-10

tangent_ex_angle = substitute( tangent_angle, exchange )
tangent_check = simplify( tangent_ex_angle - tangent_0_angle )
tangent_check_eval = substitute( tangent_check, args ) # this should be zero
@assert abs(tangent_check_eval) < 1e-10

# dipole rotated by π is the same as orifinal dipole
dipole_symmetry = Dict( α => α + π )
v_di_rot = substitute( v_beta, dipole_symmetry )
di_sym_check = simplify( v_di_rot - v_beta )
di_sym_check_eval = substitute( di_sym_check, args ) # this should be zero
@assert abs(di_sym_check_eval) < 1e-10

# check if ψ_coeff is computed correctly
# how much the vector angle from ∂ϕ changes when parallel transporting from (θ_0,ϕ_0) to (θ_0 + sin(α)dl, ϕ_0 + cos(α)csc(θ_0)dl)
dipole = Dict( θ => θ_0 + sin(α)*dl, ϕ => ϕ_0 + cos(α)*csc(θ_0)*dl )
rot_dipole = substitute( rot - π, dipole )
rot_dipole_l = Differential(dl)(rot_dipole)
rot_dipole_s = simplify(expand_derivatives(rot_dipole_l))
rot_dipole_e = simplify(substitute( rot_dipole_s, Dict( dl => 0 ) ) )
dipole_rot_check = rot_dipole_e - ψ_coeff
dipole_rot_check_eval = substitute( dipole_rot_check, args ) # this should be zero
# this is giving NaN at the moment so not putting assertion here

# check if derivative of gradient is computed correctly
g = simplify.( expand_derivatives.( grad_θ_0ϕ_0( P ) ) )
small_move = Dict( θ_0 => θ_0 + sin(α)*dl, ϕ_0 => ϕ_0 + cos(α)*csc(θ_0)*dl )
gs = [ substitute( g[1], small_move )  substitute( g[2], small_move ) ]'
gsa = simplify.(expand_derivatives.( Differential(dl).(simplify.( gs - g )) ) )
gsae = [ simplify(substitute( gsa[1], Dict( dl => 0 ) ) ) simplify(substitute( gsa[2], Dict( dl => 0 ) ) ) ]'
gsu = expand_derivatives.( sin(α)*( ∂θ_0.( grad_θ_0ϕ_0( P ) ) ) + csc(θ_0)*cos(α)*( ∂ϕ_0.( grad_θ_0ϕ_0( P ) ) ) )
grad_check = simplify.( gsu - gsae )
grad_check_eval = [ substitute( grad_check[ 1 ], args ) substitute( grad_check[ 2 ], args ) ]' # this should be zero
@assert abs(grad_check_eval[1]) < 1e-10
@assert abs(grad_check_eval[2]) < 1e-10


#========================================
# Export
========================================#

# save compiled function to file to aNothing recomputing this
# write( "./analytics/savedfunctions/dipolevelocity.jl", string(dipoleForceV[1]) )
# write( "./analytics/savedfunctions/dipolevelocitygradient.jl", string(dipoleForceVG[1]) )

#
#
# rtest1 = @rule +(~~ys) * ~x => sum(map((y->y * ~x), ~(~ys)))
# rtest2 = @rule ~x * +(~~ys) => sum(map((y->~x * y), ~(~ys)))
# rtest3 = @rule ~a*~x * +(~~ys) => sum(map((y->~a*~x * y), ~(~ys)))
# rtest4 = @rule +(~~ys) * ~a*~x => sum(map((y->y * ~a*~x), ~(~ys)))
# rtest5 = @rule +(~~ys) * *(~~xs) => sum(map((y->y * *(~~xs)), ~(~ys)))
# rtest6 = @rule *(~~xs) * +(~~ys) => sum(map((y->*(~~xs) * y), ~(~ys)))
# simplify( vtest[1], rewriter=SymbolicUtils.Postwalk(SymbolicUtils.Chain([rtest1, rtest2, rtest3, rtest4, rtest5, rtest6])))
