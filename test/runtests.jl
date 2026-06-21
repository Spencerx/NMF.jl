using NMF
using Test
using Random
using LinearAlgebra
using StatsBase
using Aqua

include("testproblems.jl")

tests = ["utils",
         "initialization",
         "spa",
         "multupd",
         "alspgrad",
         "coorddesc",
         "greedycd",
         "interf"]

println("Running tests:")
@testset "All tests" begin
    @testset "Aqua" begin
        Aqua.test_all(NMF)
    end
    for t in tests
        tp = "$t.jl"
        include(tp)
    end
end
