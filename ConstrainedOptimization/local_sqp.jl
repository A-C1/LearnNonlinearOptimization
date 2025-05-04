using LinearAlgebra
using CairoMakie
using SparseArrays

#------------------------------------------------------
# Function and its gradients and 
#------------------------------------------------------
# Function to be optimized
function f(x::Vector{T}) where T <: Number
    return sin(x[1] + x[2]) + x[3]^2 + (x[4] + x[5]^4 + x[6]/2)
end

# Gradient of the function to be optimized
function grad(x::Vector{T}) where T <: Number
    return [cos(x[1] + x[2]), cos(x[1] + x[2]), 2*x[3], 1/2, 4*x[5]^3, 1/2]
end

function grad_sparse(x::Vector{T}) where T <: Number
    return [cos(x[1] + x[2]), cos(x[1] + x[2]), 2*x[3], 1/2, 4*x[5]^3, 1/2]
end
gradinds = [1, 2, 3, 4, 5, 6]


function hess(x::Vector{T}) where T <: Number
    return [-sin(x[1] + x[2]) -sin(x[1] + x[2]) 0 0 0 0;
            -sin(x[1] + x[2]) -sin(x[1] + x[2]) 0 0 0 0;
            0 0 1 0 0 0;
            0 0 0 0 0 0;
            0 0 0 0 12*x[5]^2 0;
            0 0 0 0 0 0]
end

function hess_sparse(x)
    return [-sin(x[1] + x[2]) + 2.0, -sin(x[1] + x[2]), -sin(x[1] + x[2]) , -sin(x[1] + x[2]) + 2.0, 1.0, 1.0, 12*x[5]^2 + 1.0, 1.0]
end
rowval = [1, 1, 2, 2, 3, 4, 5, 6]
colval = [1, 2, 1, 2, 3, 4, 5, 6]


#------------------------------------------------------
# Constraints and its Jacobians and Hessians
#------------------------------------------------------
# Function to be optimized
function g(x::Vector{T}) where T
    v = zeros(2)
    v[1] = 8*x[1] - 6*x[2] + x[3] + 9*x[4] + 4*x[5] - 6
    v[2] = 3*x[1] + 2*x[2] - x[4] + 6*x[3] + 4*x[6] + 4
    return v
end

function jac_g(x::Vector{T}) where T
    return [8 -6 1 9 0 4;
            3 2 0 -1 6 4]
end

function jac_g_sparse(x::Vector{T}) where T
    return [8, -6, 1, 9, 4, 3, 2, -1, 6, 4]
end
rindg = [1 1 1 1 1 2 2 2 2 2]
cindg = [1 2 3 4 6 1 2 4 5 6]

function hess_g(x::Vector{T}, λ::Vector{T} ) where T
    return λ[1]*zeros(6,6) + λ[2]*zeros(6,6)
end

function final_hess_sparse(x::Vector{T}, λ::Vector{T}) where T
    nzv = [-sin(x[1] + x[2]), -sin(x[1] + x[2]), -sin(x[1] + x[2]), -sin(x[1] + x[2]), 1.0, 0.1, 12*x[5]^2, 0.1, (-jac_g_sparse(x))..., jac_g_sparse(x)...] 
    return nzv
end
rowval_lag = [1, 1, 2, 2, 3, 4, 5, 6, cindg..., (rindg .+ 6)...]
colval_lag = [1, 2, 1, 2, 3, 4, 5, 6, (rindg .+ 6)..., cindg...]

function final_jac(x::Vector{T}) where T
    return [-grad(x); -g(x)]
end


# Implementation follows Boyd
function back_track(f, grad, x, p; β = 0.5, α = 0.4)
    t = 1
    while (f(x + t*p) > f(x) + α*t*grad(x)'*p)
        t = β*t
    end
    return t
end

# The trial of the function starts here
iter_new = 0
x0 = rand(6)
λ0 = rand(2)
x_bar = [x0; λ0]
n = length(x0) + length(λ0)

hess_mat = sparse(rowval_lag, colval_lag, final_hess_sparse(x0, λ0), n, n) 
vec = final_jac(x0)
max_iter = 10000
tol = 1e-3
delta_x0 = zeros(8)
alpha0 = 0.01
for i = 1:max_iter
    global x0, λ0, x_bar, alpha0, iter_new, hess_mat, vec
    hess_mat.nzval .= final_hess_sparse(x0, λ0)
    vec = final_jac(x0)
    global delta_x0 = hess_mat \ vec

    alpha0 = back_track(f, grad, x0, grad(x0))
    x0 .= x0 + alpha0 * delta_x0[1:length(x0)]
    λ0 = delta_x0[length(x0)+1:end]
    iter_new = i
    if norm(vec) < tol
        break
    end
end
println("Number of newtons iteration = ", iter_new)

# function newtonssparse(f, grad, gradsparse, gradinds, hesssparse, rowval, colval, x0; alpha0 = 0.1, tol = 1e-6, max_iter = 1000)
#     iter_new = 0
#     n = length(x0)
#     hess_mat = sparse(rowval, colval, hesssparse(x0), n, n) + I(n)
#     @show rank(hess_mat)
#     grad_vec = sparsevec(gradinds, gradsparse(x0), n)
#     for i = 1:max_iter
#         # hess_mat.nzval .= hesssparse(x0)
#         hess_mat = sparse(rowval, colval, hesssparse(x0), n, n) + I(n)
#         @show hess_mat
#         grad_vec.nzval .= gradsparse(x0)
#         delta_x0 = hess_mat \ grad_vec
#         @show delta_x0

#         α0 = back_track(f, grad, x0, grad(x0))
#         x0 = x0 - alpha0*Array(delta_x0)
#         iter_new = i
#         if norm(grad_vec) < tol
#             break
#         end
#     end
#     println("Number of newtons iteration = ",iter_new)
#     return x0
# end

# x0 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0] 
# println("Result of newtons method:", newtonssparse(f, grad, grad_sparse, gradinds,
#                                                    hess_sparse, rowval, colval, x0))

# Plot level curves
# xs = LinRange(-10, 10, 1000)
# ys = LinRange(-10, 10, 1000)
# z = zeros(length(ys), length(xs))
# zs = [f([x, y]) for x in xs, y in ys]


# f1 = Figure()
# Axis(f1[1, 1])
# contour!(xs, ys, zs)
# f1
