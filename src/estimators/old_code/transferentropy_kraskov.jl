import StateSpaceReconstruction: Embeddings

function marginal_NN(points, dists_to_kth)
    D = pairwise(Chebyshev(), points, dims = 2)

    npts = size(points, 2)
    N = zeros(Int, npts)

    for i = 1:npts
        N[i] = sum(D[:, i] .<= dists_to_kth[i])
    end

    return N
end

"""
    transferentropy_kraskov(points::AbstractArray{T, 2}, k1::Int, k2::Int,
        v::TEVars; metric = Chebyshev(), b = 2)

Compute transfer entropy decomposed as the sum of mutual informations,
using an adapted version of the Kraskov estimator for mutual information [1].

## Arguments

- `points`: The set of points representing the embedding for which to compute
    transfer entropy. Must be provided as an array of size `dim`-by-`n` points.
- `k1`: The number of nearest neighbours for the highest-dimensional mutual
    information estimate. To minimize bias, choose ``k_1 < k_2`` if
    ``min(k_1, k_2) < 10`` (see fig. 16 in [1]). Beyond dimension 5, choosing
    ``k_1 = k_2`` results in fairly low bias, and a low number of nearest
    neighbours, say `k1 = k2 = 4`, will suffice.
- `k2`: The number of nearest neighbours for the lowest-dimensional mutual
    information estimate. To minimize bias, choose ``k_1 < k_2`` if
    if ``min(k_1, k_2) < 10`` (see fig. 16 in [1]). Beyond dimension 5, choosing
    ``k_1 = k_2`` results in fairly low bias, and a low number of nearest
    neighbours, say `k1 = k2 = 4`, will suffice.
- `v`: A `TEVars` instance, indicating which variables of the embedding should
    be grouped as what when computing the marginal entropies that go into the
    transfer entropy expression.

## Keyword arguments

- `metric`: The distance metric. Must be a valid metric from `Distances.jl`.
- `b`: The base of the logarithm, dictating the unit of the transfer entropy. 
        The default `b=2` gives the transfer entropy in bits.

## References

1. Kraskov, Alexander, Harald Stögbauer, and Peter Grassberger. "Estimating
    mutual information." Physical review E 69.6 (2004): 066138.
"""
function transferentropy_kraskov(points::AbstractArray{T, 2}, k1::Int, k2::Int,
        v::TEVars; metric = Chebyshev(), normalise = false, b = 2) where T

    # Make sure that the array contains points as columns.
    if size(points, 1) > size(points, 2)
        error("The dimension of the dataset exceeds the number of points.")
    end
    # The total number of points
    N = size(points, 2)

    # Create some dummy variable names to avoid cluttering the code too much
    X = v.target_future
    Y = v.target_presentpast
    Z = vcat(v.source_presentpast, v.conditioned_presentpast)
    XY = vcat(X, Y)
    YZ = vcat(Y, Z)

    pts_X = points[X, :]
    pts_Y = points[Y, :]
    pts_XY = points[XY, :]
    pts_YZ = points[YZ, :]
    pts_XYZ = points

    # Create trees to search for nearest neighbors
    tree_XYZ = KDTree(pts_XYZ, metric)
    tree_XY = KDTree(pts_XY, metric)

    # Find the k nearest neighbors to all of the points in each of the trees
    idxs_XYZ, dists_XYZ = knn(tree_XYZ, pts_XYZ, k1, true)
    idxs_XY, dists_XY   = knn(tree_XY,  pts_XY, k2, true)

    # In each of the trees, find the index of the k-th nearest neighbor to all
    # of the points.
    kth_NN_idx_XYZ = [idx[k1] for idx in idxs_XYZ]
    kth_NN_idx_XY  = [idx[k2] for idx in idxs_XY]

    # Distances between points in the XYZ and XY spaces and their
    # kth nearest neighbour, along marginals X, YZ (for the XYZ space)
    # and X, Y (for the XY space).
    ϵ_XYZ_X = colwise(metric, pts_X, pts_X[:, kth_NN_idx_XYZ])
    ϵ_XYZ_YZ = colwise(metric, pts_YZ, pts_YZ[:, kth_NN_idx_XYZ])
    ϵ_XY_X = colwise(metric, pts_X, pts_X[:, kth_NN_idx_XY])
    ϵ_XY_Y = colwise(metric, pts_Y, pts_Y[:, kth_NN_idx_XY])

    NXYZ_X  = marginal_NN(pts_X,  ϵ_XYZ_X)
    NXYZ_YZ = marginal_NN(pts_YZ, ϵ_XYZ_YZ)
    NXY_X   = marginal_NN(pts_X,  ϵ_XY_X)
    NXY_Y   = marginal_NN(pts_Y,  ϵ_XY_Y)

    te = sum(digamma.(NXY_X) + digamma.(NXY_Y) -
        digamma.(NXYZ_X) - digamma.(NXYZ_YZ)) / N

    # Convert transfer entropy estimate to appropriate logarithm
    te / log(b)
end


"""
    transferentropy_kraskov(points::AbstractArray{T, 2}, k1::Int, k2::Int,
        target_future::Union{Int, UnitRange{Int}, Vector{Int}, Tuple{Int}},
        target_presentpast::Union{Int, UnitRange{Int}, Vector{Int}, Tuple{Int}},
        source_presentpast::Union{Int, UnitRange{Int}, Vector{Int}, Tuple{Int}},
        conditioned_presentpast::Union{Int, UnitRange{Int}, Vector{Int}, Tuple{Int}};
        metric = Chebyshev(), b = 2) where T

Compute transfer entropy decomposed as the sum of mutual informations,
using an adapted version of the Kraskov estimator for mutual information [1].

## Arguments
- `points`: The set of points representing the embedding for which to compute
    transfer entropy. Must be provided as an array of size `dim`-by-`n` points.
- `k1`: The number of nearest neighbours for the highest-dimensional mutual
    information estimate. To minimize bias, choose ``k_1 < k_2`` if
    ``min(k_1, k_2) < 10`` (see fig. 16 in [1]). Beyond dimension 5, choosing
    ``k_1 = k_2`` results in fairly low bias, and a low number of nearest
    neighbours, say `k1 = k2 = 4`, will suffice.
- `k2`: The number of nearest neighbours for the lowest-dimensional mutual
    information estimate. To minimize bias, choose ``k_1 < k_2`` if
    if ``min(k_1, k_2) < 10`` (see fig. 16 in [1]). Beyond dimension 5, choosing
    ``k_1 = k_2`` results in fairly low bias, and a low number of nearest
    neighbours, say `k1 = k2 = 4`, will suffice.
- `v`: A `TEVars` instance, indicating which variables of the embedding should
    be grouped as what when computing the marginal entropies that go into the
    transfer entropy expression.
- `target_future`: Which rows of `points` correspond to future values of the
    target variable?
- `target_presentpast`: Which rows of `points` correspond to present and past
    values of the target variable?
- `source_presentpast`: Which rows of `points` correspond to present and past
    values of the source variable?
- `conditioned_presentpast`: Which rows of `points` correspond to present and
    past values of conditional variables?

## Keyword arguments
- `metric`: The distance metric. Must be a valid metric from `Distances.jl`.
- `b`: The base of the logarithm, dictating the unit of the transfer entropy. 
        The default `b=2` gives the transfer entropy in bits.

# References
1. Kraskov, Alexander, Harald Stögbauer, and Peter Grassberger. "Estimating
    mutual information." Physical review E 69.6 (2004): 066138.
"""
function transferentropy_kraskov(points::AbstractArray{T, 2}, k1::Int, k2::Int,
        target_future::Union{Int, UnitRange{Int}, Vector{Int}, Tuple{Int}},
        target_presentpast::Union{Int, UnitRange{Int}, Vector{Int}, Tuple{Int}},
        source_presentpast::Union{Int, UnitRange{Int}, Vector{Int}, Tuple{Int}},
        conditioned_presentpast::Union{Int, UnitRange{Int}, Vector{Int}, Tuple{Int}};
        metric = Chebyshev(), b = 2) where T

    # Make sure that the array contains points as columns.
    if size(points, 1) > size(points, 2)
        error("The dimension of the dataset exceeds the number of points.")
    end

    # Create some dummy variable names to avoid cluttering the code too much
    v = TEVars(target_future, target_presentpast,
                source_presentpast, conditioned_presentpast)

    transferentropy_kraskov(points, k1, k2, v, b = b)
end


"""
    transferentropy_kraskov(points::AbstractArray{T, 2}, k1::Int, k2::Int,
        target_future::Union{Int, UnitRange{Int}, Vector{Int}, Tuple{Int}},
        target_presentpast::Union{Int, UnitRange{Int}, Vector{Int}, Tuple{Int}},
        source_presentpast::Union{Int, UnitRange{Int}, Vector{Int}, Tuple{Int}};
        metric = Chebyshev(), b = 2) where T

Compute transfer entropy decomposed as the sum of mutual informations,
using an adapted version of the Kraskov estimator for mutual information [1].

## Arguments

- `points`: The set of points representing the embedding for which to compute
    transfer entropy. Must be provided as an array of size `dim`-by-`n` points.
- `k1`: The number of nearest neighbours for the highest-dimensional mutual
    information estimate. To minimize bias, choose ``k_1 < k_2`` if
    ``min(k_1, k_2) < 10`` (see fig. 16 in [1]). Beyond dimension 5, choosing
    ``k_1 = k_2`` results in fairly low bias, and a low number of nearest
    neighbours, say `k1 = k2 = 4`, will suffice.
- `k2`: The number of nearest neighbours for the lowest-dimensional mutual
    information estimate. To minimize bias, choose ``k_1 < k_2`` if
    if ``min(k_1, k_2) < 10`` (see fig. 16 in [1]). Beyond dimension 5, choosing
    ``k_1 = k_2`` results in fairly low bias, and a low number of nearest
    neighbours, say `k1 = k2 = 4`, will suffice.
- `v`: A `TEVars` instance, indicating which variables of the embedding should
    be grouped as what when computing the marginal entropies that go into the
    transfer entropy expression.
- `target_future`: Which rows of `points` correspond to future values of the
    target variable?
- `target_presentpast`: Which rows of `points` correspond to present and past
    values of the target variable?
- `source_presentpast`: Which rows of `points` correspond to present and past
    values of the source variable?

This version of the function assumes there is no conditioning.

## Keyword arguments

- `metric`: The distance metric. Must be a valid metric from `Distances.jl`.
- `b`: The base of the logarithm, dictating the unit of the transfer entropy. 
        The default `b=2` gives the transfer entropy in bits.


# References
1. Kraskov, Alexander, Harald Stögbauer, and Peter Grassberger. "Estimating
    mutual information." Physical review E 69.6 (2004): 066138.
"""
function transferentropy_kraskov(points::AbstractArray{T, 2}, k1::Int, k2::Int,
        target_future::Union{Int, UnitRange{Int}, Vector{Int}, Tuple{Int}},
        target_presentpast::Union{Int, UnitRange{Int}, Vector{Int}, Tuple{Int}},
        source_presentpast::Union{Int, UnitRange{Int}, Vector{Int}, Tuple{Int}};
        metric = Chebyshev(), b = 2) where T

    # Make sure that the array contains points as columns.
    if size(points, 1) > size(points, 2)
        error("The dimension of the dataset exceeds the number of points.")
    end

    # Create some dummy variable names to avoid cluttering the code too much
    v = TEVars(target_future, target_presentpast,
                source_presentpast, Int[])

    transferentropy_kraskov(points, k1, k2, v, b = b)
end

"""
    transferentropy_kraskov(E::StateSpaceReconstruction.AbstractEmbedding,
            k1::Int, k2::Int, v::TEVars; metric = Chebyshev(), b = 2)

Compute transfer entropy decomposed as the sum of mutual informations,
using an adapted version of the Kraskov estimator for mutual information [1].

## Arguments

- `E`: The embedding for which to compute transfer entropy.
- `k1`: The number of nearest neighbours for the highest-dimensional mutual
    information estimate. To minimize bias, choose ``k_1 < k_2`` if
    ``min(k_1, k_2) < 10`` (see fig. 16 in [1]). Beyond dimension 5, choosing
    ``k_1 = k_2`` results in fairly low bias, and a low number of nearest
    neighbours, say `k1 = k2 = 4`, will suffice.
- `k2`: The number of nearest neighbours for the lowest-dimensional mutual
    information estimate. To minimize bias, choose ``k_1 < k_2`` if
    if ``min(k_1, k_2) < 10`` (see fig. 16 in [1]). Beyond dimension 5, choosing
    ``k_1 = k_2`` results in fairly low bias, and a low number of nearest
    neighbours, say `k1 = k2 = 4`, will suffice.
- `v`: A `TEVars` instance, indicating which variables of the embedding should
    be grouped as what when computing the marginal entropies that go into the
    transfer entropy expression.

## Keyword arguments

- `metric`: The distance metric. Must be a valid metric from `Distances.jl`.
- `b`: The base of the logarithm, dictating the unit of the transfer entropy. 
        The default `b=2` gives the transfer entropy in bits.

## References

1. Kraskov, Alexander, Harald Stögbauer, and Peter Grassberger. "Estimating
mutual information." Physical review E 69.6 (2004): 066138.
"""
function transferentropy_kraskov(E::Embeddings.AbstractEmbedding{D, T},
            k1::Int, k2::Int, v::TEVars; metric = Chebyshev(), b = 2) where {D, T}

    # Make sure that the array contains points as columns.
    if size(E.points, 1) > size(E.points, 2)
        error("The dimension exceeds the number of points.")
    end

    transferentropy_kraskov(E.points, k1, k2, v, metric = metric, b = b)
end

"""
    transferentropy_kraskov(E::AbstractEmbedding, k1::Int, k2::Int,
        target_future::Union{Int, UnitRange{Int}, Vector{Int}, Tuple{Int}},
        target_presentpast::Union{Int, UnitRange{Int}, Vector{Int}, Tuple{Int}},
        source_presentpast::Union{Int, UnitRange{Int}, Vector{Int}, Tuple{Int}},
        conditioned_presentpast::Union{Int, UnitRange{Int}, Vector{Int}, Tuple{Int}};
        metric = Chebyshev(), b = 2) where T

Compute transfer entropy decomposed as the sum of mutual informations,
using an adapted version of the Kraskov estimator for mutual information [1].

## Arguments

- `points`: The set of points representing the embedding for which to compute
    transfer entropy. Must be provided as an array of size `dim`-by-`n` points.
- `k1`: The number of nearest neighbours for the highest-dimensional mutual
    information estimate. To minimize bias, choose ``k_1 < k_2`` if
    ``min(k_1, k_2) < 10`` (see fig. 16 in [1]). Beyond dimension 5, choosing
    ``k_1 = k_2`` results in fairly low bias, and a low number of nearest
    neighbours, say `k1 = k2 = 4`, will suffice.
- `k2`: The number of nearest neighbours for the lowest-dimensional mutual
    information estimate. To minimize bias, choose ``k_1 < k_2`` if
    if ``min(k_1, k_2) < 10`` (see fig. 16 in [1]). Beyond dimension 5, choosing
    ``k_1 = k_2`` results in fairly low bias, and a low number of nearest
    neighbours, say `k1 = k2 = 4`, will suffice.
- `v`: A `TEVars` instance, indicating which variables of the embedding should
    be grouped as what when computing the marginal entropies that go into the
    transfer entropy expression.
- `target_future`: Which rows of `points` correspond to future values of the
    target variable?
- `target_presentpast`: Which rows of `points` correspond to present and past
    values of the target variable?
- `source_presentpast`: Which rows of `points` correspond to present and past
    values of the source variable?
- `conditioned_presentpast`: Which rows of `points` correspond to present and
    past values of conditional variables?

## Keyword arguments

- `metric`: The distance metric. Must be a valid metric from `Distances.jl`.
- `b`: The base of the logarithm, dictating the unit of the transfer entropy. 
        The default `b = 2` gives the transfer entropy in bits.

# References

1. Kraskov, Alexander, Harald Stögbauer, and Peter Grassberger. "Estimating
    mutual information." Physical review E 69.6 (2004): 066138.
"""
function transferentropy_kraskov(E::Embeddings.AbstractEmbedding{D, T}, k1::Int, k2::Int,
        target_future::Union{Int, UnitRange{Int}, Vector{Int}, Tuple{Int}},
        target_presentpast::Union{Int, UnitRange{Int}, Vector{Int}, Tuple{Int}},
        source_presentpast::Union{Int, UnitRange{Int}, Vector{Int}, Tuple{Int}},
        conditioned_presentpast::Union{Int, UnitRange{Int}, Vector{Int}, Tuple{Int}};
        metric = Chebyshev(), b = 2) where {D, T}

    # Make sure that the array contains points as columns.
    if size(E.points, 1) > size(E.points, 2)
        error("The dimension of the dataset exceeds the number of points.")
    end

    # Create some dummy variable names to avoid cluttering the code too much
    v = TEVars(target_future, target_presentpast,
                source_presentpast, conditioned_presentpast)

    transferentropy_kraskov(E.points, k1, k2, v, b = b)
end

tekraskov = transferentropy_kraskov
tekNN = transferentropy_kraskov

export
transferentropy_kraskov,
tekraskov,
tekNN
