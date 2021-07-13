#if lowercase(get(ENV, "CI", "false")) == "true"
#    include("install_dependencies.jl")
#end

#using CausalityToolsBase
using Test, TransferEntropy, Entropies
KDTree = Entropies.KDTree
BruteForce = Entropies.BruteForce

@testset "Mutual information" begin
    x = rand(100)
    y = rand(100)
    z = Dataset(rand(100, 2))
    w = Dataset(rand(100, 3))

    # Estimators for which Renyi entropies can be used
    est = VisitationFrequency(RectangularBinning(0.2))
    @test mutualinfo(x, y, est) isa Real
    @test mutualinfo(x, y, est, q = 2) isa Real
    @test mutualinfo(x, z, est) isa Real
    @test mutualinfo(z, x, est) isa Real
    @test mutualinfo(z, w, est) isa Real

    # Estimators for which Renyi entropies cannot be used
    est_kl = KozachenkoLeonenko()
    est_k = Kraskov()
    est_k1 = Kraskov1(2)
    est_k2 = Kraskov2(2)

    @test mutualinfo(x, y, est_kl) isa Real
    @test mutualinfo(x, y, est_kl, base = 2) isa Real
    @test mutualinfo(x, z, est_kl) isa Real
    @test mutualinfo(z, x, est_kl) isa Real
    @test mutualinfo(z, w, est_kl) isa Real

    @test mutualinfo(x, y, est_k) isa Real
    @test mutualinfo(x, y, est_k, base = 2) isa Real
    @test mutualinfo(x, z, est_k) isa Real
    @test mutualinfo(z, x, est_k) isa Real
    @test mutualinfo(z, w, est_k) isa Real

    @test mutualinfo(x, y, est_k1) isa Real
    @test mutualinfo(x, y, est_k1, base = 2) isa Real
    @test mutualinfo(x, z, est_k1) isa Real
    @test mutualinfo(z, x, est_k1) isa Real
    @test mutualinfo(z, w, est_k1) isa Real

    @test mutualinfo(x, y, est_k2) isa Real
    @test mutualinfo(x, y, est_k2, base = 2) isa Real
    @test mutualinfo(x, z, est_k2) isa Real
    @test mutualinfo(z, x, est_k2) isa Real
    @test mutualinfo(z, w, est_k2) isa Real
end

@testset "Transfer entropy" begin 
    s, t, c = rand(100), rand(100), rand(100)
    println("Starting transfer entropy tests...")
    @testset "Generalized Renyi transfer entropy" begin
        # Straight-forward estimators
        est_vf = VisitationFrequency(RectangularBinning(4))
        ests = [
            est_vf,
            TransferOperator(RectangularBinning(5)),
            Hilbert(source = Phase(), target = Amplitude(), est_vf),
            Hilbert(source = Amplitude(), target = Amplitude(), est_vf),
            Hilbert(source = Phase(), target = Phase(), est_vf),
            Hilbert(source = Phase(), target = Phase(), cond = Amplitude(), est_vf),
            NaiveKernel(0.5, KDTree),
            NaiveKernel(0.5, BruteForce)
        ]

        @testset "Generalized Renyi transfer entropy $(ests[i])"  for i in 1:length(ests)
            println(ests[i])
            est = ests[i]
            te =  transferentropy(s, t, est, q = 2, base = 2)
            tec = transferentropy(s, t, c, est, q = 1, base = 3)

            @test te isa Real
            @test tec isa Real

            if est isa VisitationFrequency
                @test te > 0 || te ≈ 0
                @test tec > 0 || tec ≈ 0
            end
        end

        @testset "SymbolicPermutation" begin
            m, τ = 3, 1
            est = SymbolicPermutation(m = m, τ = τ)
            te =  transferentropy(s, t, est, q = 2, base = 2)
            tec = transferentropy(s, t, c, est, q = 1, base = 3)

            S, T, C = ([zeros(Int, length(s)-(m-1)*τ) for x in 1:3]...,)
            te_inplace =  transferentropy!(S, T, s, t, est, q = 2, base = 2)
            tec_inplace = transferentropy!(S, T, C, s, t, c, est, q = 1, base = 3)

            @test te isa Real
            @test tec isa Real
            @test te_inplace isa Real
            @test tec_inplace isa Real
        end
    end


    
    @testset "Shannon transfer entropy" begin 
        ests = [
            Kraskov(),
            KozachenkoLeonenko()
        ]
        @testset "Nearest neighbors $(ests[i])"  for i in 1:length(ests)
            println(ests[i])
            est = ests[i]
            te =  transferentropy(s, t, est, base = 2)
            tec = transferentropy(s, t, c, est, base = 3)
            @test te isa Real
            @test tec isa Real

        end
    end

    

end
