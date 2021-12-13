using LinearAlgebra
using Symbolics

struct SphericalCoord
    theta::Float64 # angle from z-axis
    phi::Float64 # angle from x-axis in x-y plane
end

Base.:+(x::SphericalCoord, y::SphericalCoord) = SphericalCoord(x.theta+y.theta, x.phi+y.phi)

struct CartesianCoord
    x::Float64
    y::Float64
    z::Float64
end

LinearAlgebra.normalize( c::CartesianCoord ) = begin
    n = normalize( [c.x, c.y, c.z ] )
    CartesianCoord( n[1], n[2], n[3])
end

LinearAlgebra.norm( c::CartesianCoord ) = norm( [c.x, c.y, c.z ] )

function cartesianToSpherical( p::CartesianCoord )
    r = sqrt( p.x^2 + p.y^2 + p.z^2 )
    theta = acos(p.z/r)
    phi = atan(p.y,p.x)
    SphericalCoord( theta, phi )
end

function sphericalToCartesian( p::SphericalCoord )
    x = cos(p.phi)*sin(p.theta)
    y = sin(p.phi)*sin(p.theta)
    z = cos(p.theta)
    CartesianCoord( x, y, z )
end

# p is the position, v is the velocity
function sphericalToCartesianVelocity( p::SphericalCoord, v::SphericalCoord )
    # p.theta is dtheta, p.phi is dphi
    vx = [cos(p.phi)*cos(p.theta), - sin(p.phi)]' * [v.theta, v.phi]
    vy = [sin(p.phi)*cos(p.theta), cos(p.phi)]' * [v.theta, v.phi]
    vz = -sin(p.theta)*v.theta
    CartesianCoord( vx, vy, vz )
end

# geodesic distance
rho( p1::SphericalCoord, p2 ::SphericalCoord) = acos( cos(p1.theta)*cos(p2.theta) + sin(p1.theta)*sin(p2.theta)*cos(p1.phi - p2.phi) )
# derivative of geodesic distance \rho wrt \phi
alpha( p1::SphericalCoord, p2 ::SphericalCoord) = sin(p1.theta) * sin(p2.theta) * sin(p1.phi-p2.phi) / sin(rho(p1,p2))
# derivative of geodesic distance \rho wrt \theta
beta( p1::SphericalCoord, p2 ::SphericalCoord) = ( cos(p2.theta) * sin(p1.theta) -  cos(p1.theta) * sin(p2.theta) * cos(p1.phi-p2.phi) ) / sin(rho(p1,p2))

#http://www.damtp.cam.ac.uk/user/reh10/lectures/nst-mmii-handout2.pdf

# what is ϕ at θ for a geodesic passing through point (θ0,ϕ0) and with geodesic parameter c
# geodesic_phi( c, θ, θ0, ϕ0 ) = acos( c * cot(θ) / sqrt( 1-c^2) ) - acos( c * cot(θ0) / sqrt( 1-c^2) ) + ϕ0
# what is the angle with phi axis of the tangent to the geodesic with parameter c at theta = acot( sin(θ)dϕ/dθ )
# geodesic_tangent( θ, c ) = c / sqrt( ( sin(θ)^2 −c^2 ) )

# find the c parameter of the geodesic given two points on the geodesic
# test: geodesic_param( θ, geodesic_phi( c, θ, θ0, ϕ0 ), θ0, ϕ0  ) == c
# geodesic_param( θ, ϕ, θ0, ϕ0 ) = c
# cos( ϕ - ϕ0 ) = (c^2 * cot(θ) * cot(θ0) + sqrt( (1 - c^2 * cosec(θ)^2) * (1 - c^2 * cosec(θ0)^2) ) / 1-c^2
# x = c^2
# cos( ϕ - ϕ0 )  - x ( cos( ϕ - ϕ0)  + cot(θ) * cot(θ0)) = sqrt( (1 - x * cosec(θ)^2) * (1 - x * cosec(θ0)^2))
# cos( ϕ - ϕ0 )^2 + x^2 * ( cos( ϕ - ϕ0)  + cot(θ) * cot(θ0))^2 - 2 * x * cos( ϕ - ϕ0 ) * ( cos( ϕ - ϕ0)  + cot(θ) * cot(θ0)) = (1 - x * cosec(θ)^2) * (1 - x * cosec(θ0)^2)
# x^2 *( ( cos( ϕ - ϕ0)  + cot(θ) * cot(θ0) )^2 - cosec(θ)^2 cosec(θ0)^2 ) + x (cosec(θ)^2 + cosec(θ0)^2 - 2 * cos( ϕ - ϕ0 ) * ( cos( ϕ - ϕ0)  + cot(θ) * cot(θ0) ) ) = 0
