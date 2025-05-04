module Tst

using LinearAlgebra
using GLMakie
using SparseArrays

include("UnconstrainedOptimization.jl")
using .UnconstrainedOptimization

# Function to be optimized
function f(x::Vector{T}) where T <: Number
    return x[1]^2+ x[2]^2
end

f(x, y) = f([x, y])

# Gradient of the function to be optimized
function ∇f(x::Vector{T}) where T <: Number
    return [2x[1], 2x[2]]
end

function grad_sparse(x::Vector{T}) where T<: Number
    return [2x[1], 2x[2]]
end

gradinds = [1, 2]


function hess(x::Vector{T}) where T <: Number
    return [2 0; 0 2]
end

function hess_sparse(x)
    vals = [2, 2]
    return vals
end

rowval = [1, 2]
colptr = [1, 2, 3]
colval = [1, 2]


# Implementation follows Boyd
function back_track(f, ∇f, x, p; β = 0.5, α = 0.4)
    t = 1
    while (f(x + t*p) > f(x) + α*t*∇f(x)'*p)
        t = β*t
    end
    return t
end

function back_track_sparse(f, gradsparse, x, p; β = 0.5, α = 0.4)
    t = 1
    while (f(x + t*p) > f(x) + α*t*∇f(x)'*p)
        t = β*t
    end
    return α
end

function graddes(f, ∇f, x0; α0 = 0.1, tol = 1e-6, max_iter = 1000)
    for i = 1:max_iter
        α0 = back_track(f, ∇f, x0, ∇f(x0))
        x0 = x0 - α0*∇f(x0)
        if norm(∇f(x0)) < tol
            break
        end
    end
    return x0
end

function newtons(f, grad, hess, x0; alpha0 = 0.1, tol = 1e-6, max_iter = 1000 )
    iter_new = 0
    for i = 1:max_iter
        # alpha0 = back_track(f, ∇f, x0, ∇f(x0))
        delta_x0 = hess(x0) \ grad(x0)
        x0 = x0 - alpha0*delta_x0
        iter_new = i
        if norm(∇f(x0)) < tol
            break
        end
    end
    println("Number of newtons iteration = ",iter_new)
    return x0
end

function newtonssparse(f, gradsparse, gradinds, hesssparse, rowval, colval, x0; alpha0 = 0.1, tol = 1e-6, max_iter = 1000)
    iter_new = 0
    hess_mat = sparse(rowval, colval, hesssparse(x0))
    grad_vec = sparsevec(gradinds, gradsparse(x0))
    for i = 1:max_iter
        hess_mat.nzval .= hesssparse(x0)
        grad_vec.nzval .= gradsparse(x0)
        delta_x0 = hess_mat \ grad_vec
        x0 = x0 - alpha0*Array(delta_x0)
        iter_new = i
        if norm(grad_vec) < tol
            break
        end
    end
    println("Number of newtons iteration = ",iter_new)
    return x0
end

x0 = [1.0, 1.0] 
println("Result of gradient descent:", graddes(f, ∇f, x0))
println("Result of newtons method:", newtons(f, ∇f, hess, x0))
println("Result of newtons method:", newtonssparse(f, grad_sparse, gradinds,
                                                   hess_sparse, rowval, colval, x0))

# Plot level curves
xs = LinRange(-10, 10, 1000)
ys = LinRange(-10, 10, 1000)
# z = zeros(length(ys), length(xs))
zs = [f(x, y) for x in xs, y in ys]


f1 = Figure()
Axis(f1[1, 1])
contour!(xs, ys, zs)
f1
    
end