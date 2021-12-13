using DataFrames
using Symbolics
using Interpolations
using DiffRules
using SymbolicUtils

# differentiate a function specified taken from tabular data
# data is an interpolated AbstractArray and the original data points
function dataDerivative( data::Tuple{AbstractArray,Vector{Float64}} )::Tuple{AbstractArray,Vector{Float64}}
    xs = data[ 2 ]
    ys = map( data[ 1 ], xs )

    if( length(xs) != length(ys))
        throw("Data doesn't have same number of xs and ys.")
    end

    l = length( ys )
    nys = if l > 2
        vcat(
            [ (ys[2] - ys[1])/(xs[2] - xs[1]) ],
            [ (ys[i+1] - ys[i-1])/(xs[i+1] - xs[i-1]) for i in 2:(l-1) ],
            [ (ys[l] - ys[l-1])/(xs[l] - xs[l-1])]
        )
    elseif l == 2
        [
            (ys[2] - ys[1])/(xs[2] - xs[1]),
            (ys[2] - ys[1])/(xs[2] - xs[1])
        ]
    else
        throw( "Data must have atleast two points to be able to take derivative." )
    end

    nData = ( LinearInterpolation( xs, nys; extrapolation_bc = Line() ), xs )
    return( nData )
end


# just return the value at index x from the abstractvector in data
function dataFunction( data::Tuple{AbstractArray,Vector{Float64}}, x::Float64 )::Float64
    data[ 1 ](x)
end
# if this derivative is not specified correctly then there may be a weird error when trying to expand_derivatives on this function
# Symbolics.derivative( ::typeof(dataFunction), args::NTuple{2,Any}, ::Val{2} ) = SymbolicUtils.Term( dataFunction, [ dataDerivative(args[1]), args[2] ] )
# Symbolics.@register dataFunction( data::Tuple{AbstractArray,Vector{Float64}}, x )
