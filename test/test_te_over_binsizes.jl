using Distributions
using InvariantDistribution
using SimplexSplitting
using TransferEntropy
using PlotlyJS

function te_over_binsizes(binsizes; npts::Int = 1000, covariance = 0.4)
    # Create an uncorrelated source and target
    dist = Normal()

    source = rand(dist, npts, 1)
    dest = covariance .* source[1:end] .+ (1.0 - covariance) .* rand(dist, npts, 1)

    embedding = embed_correlated_gaussians(;npts = npts, covariance = covariance)

    # Embed using all points except the last (to allow projection of all vertices
    # in the triangulation).
    t = SimplexSplitting.triang_from_embedding(embedding)

    #points, simplex_inds = triangulate(embedding[1:end-1, :])
    #image_points = embedding[2:end, :]

    P = markovmatrix(view(t.points, :, :), view(t.impoints, :, :), view(t.simplex_inds, :, :))

    invmeasure, inds_nonzero_simplices = invariantdist(P)
    #centroids = rand(dist, npts, 3)
    #invmeasure = abs.(rand(dist, npts))
    #invmeasure[randperm(400)[50]] = 0
    #invmeasure = invmeasure ./ sum(invmeasure) # normalise to true probability dist.

    TE = zeros(length(binsizes))

    count = 0
    for binsize in binsizes
        count +=1
        te = te_from_triangulation(t.centroids, invmeasure, binsize)
        TE[count] = te
    end
    return TE
end









##################################################################
# Calculate TE for correlated gaussians over a range of bin sizes
##################################################################
binsizes = 1:1:150
covars = 0.3
reps = 1
npts = 50
TEs = zeros(Float64, length(binsizes), reps, length(covars))


for j = 1:length(covars)
    covar = covars[j]
    print("Covariance: ", covar, " Expected TE: ", log(1/(1 - covar^2)))
    for i = 1:reps
        println("\tRep #", i)
        te = te_over_binsizes(binsizes; npts = npts, covariance = covar)
        TEs[1:end, i, j] = te
    end
end
####################
# PLOT THE TE CURVES
####################
data = PlotlyJS.GenericTrace[]
layout = PlotlyJS.Layout(autosize = true,
                          margin = PlotlyJS.attr(l=0, r=0, b=0, t=65),
                          xaxis = PlotlyJS.attr(title = "Bin size", ticks = "outside"),
                          yaxis = PlotlyJS.attr(title = "TE (nats)"))
for i = 1:reps
    trace = PlotlyJS.scatter(;x = binsizes,
                               y = TEs[1:end, i],
                               mode = "markers+lines",
                               showlegend = false,
                               marker = PlotlyJS.attr(size = 3, color = "black"),
                               line = PlotlyJS.attr(size = 0.8, color = "white"))
    push!(data, trace)
end
#push!(data, PlotlyJS.scatter(;x = binsizes, y = TEs, mode = "lines"))
push!(data, PlotlyJS.scatter(; x = binsizes, y = fill(log(1/(1-0.3^2)), length(binsizes)) ))
PlotlyJS.plot(data, layout)

#plot(scatter(;x = binsizes, y = te_over_binsizes(binsizes; covariance = covar)))

#@show te
#binsizes = 1:100
#TE = zeros(size(binsizes))
#@testset "Transfer entropy over bin sizes" begin
#    @testset "k for $k" for k in binsizes
#        @show k
#        te = TransferEntropy.te_from_triangulation(centroids, invmeasure, k)
#3        @test te >= 0
#        TE[k] = te
#    end
#end
#te_from_triangulation(centroids, invmeasure, 20)