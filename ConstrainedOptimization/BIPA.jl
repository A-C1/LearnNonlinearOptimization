# This file implements the basic interior point aalgorithm as described in  Nodcedal and Wright with dense matrices
# Line search method is used since that is the simplest
using LinearAlgebra
using Zygote


# Objective function
f(x) = x[1]*x[4]*(x[1] + x[2] + x[3]) + x[3]

# Gradient and hessian of objective function
grad_f(x) = Zygote.gradient(f, x)[1]
hess_f(x) = Zygote.hessian(f, x)

# Equality constraints
g_e1(x) = x[1]^2 + x[2]^2 + x[3]^2 + x[4]^2 - 40
g_e(x) = [g_e1(x)]

# Jacobian and hessian of equality constraints
jac_g_e(x) = Zygote.jacobian(g_e, x)[1]
hess_g_e1(x) = Zygote.hessian(g_e1, x)
hess_g_e(x, λ_e) = λ_e[1]*hess_g_e1(x)

# Inequality constraints
g_i1(x) = -x[1]*x[2]*x[3]*x[4] + 25
g_i2(x) = x[1] - 5
g_i3(x) = x[2] - 5
g_i4(x) = x[3] - 5
g_i5(x) = x[4] - 5
g_i6(x) = -x[1] + 1
g_i7(x) = -x[2] + 1
g_i8(x) = -x[3] + 1
g_i9(x) = -x[4] + 1
g_i(x) = [g_i1(x), g_i2(x), g_i3(x), g_i4(x), g_i5(x), g_i6(x), g_i7(x), g_i8(x), g_i9(x)]

# Jacobian and hessian of inequality constraints
jac_g_i(x) = Zygote.jacobian(g_i, x)[1]
hess_g_i1(x) = Zygote.hessian(g_i1, x)
hess_g_i2(x) = Zygote.hessian(g_i2, x)
hess_g_i3(x) = Zygote.hessian(g_i3, x)
hess_g_i4(x) = Zygote.hessian(g_i4, x)
hess_g_i5(x) = Zygote.hessian(g_i5, x)
hess_g_i6(x) = Zygote.hessian(g_i6, x)
hess_g_i7(x) = Zygote.hessian(g_i7, x)
hess_g_i8(x) = Zygote.hessian(g_i8, x)
hess_g_i9(x) = Zygote.hessian(g_i9, x)
hess_g_i(x, λ_i) = λ_i[1]*hess_g_i1(x) + λ_i[2]*hess_g_i2(x) + λ_i[3]*hess_g_i3(x) + λ_i[4]*hess_g_i4(x) + λ_i[5]*hess_g_i5(x) + λ_i[6]*hess_g_i6(x) + λ_i[7]*hess_g_i7(x) + λ_i[8]*hess_g_i8(x) + λ_i[9]*hess_g_i9(x)

# Compute the hessian of Lagrangian
hess_l(x, λe, λi) = hess_f(x) + hess_g_e(x, λe) + hess_g_i(x, λi)

# Compute the kkt matrix and kkt vector
# s is a slack vector
kkt_mat(x, s, λe, λi)::Matrix{Float64} = [hess_l(x, λe, λi)  zeros(4, 9)                     jac_g_e(x)'   jac_g_i(x)';
                                            zeros(9,4)           Diagonal(1 ./ s)*Diagonal(λi)  zeros(9, 1)   -I(9);
                                            jac_g_e(x)           zeros(1, 9)                     zeros(1, 1)   zeros(1,9);
                                            jac_g_i(x)           -I(9)                           zeros(9,1)    zeros(9,9)]

kkt_vec(x, s, λe, λi, μ)::Vector{Float64} = -[grad_f(x) - jac_g_e(x)'*λe - jac_g_i(x)'*λi;
                                                λi - μ*(1 ./ s);
                                                g_e(x);
                                                g_i(x)-s]

# Add merit function
ϕv(x, s, μ, v) = f(x) - μ * sum(log.(s)) + 0.5*v*(norm(g_e(x))^2 + norm(g_i(x) - s)^2)
# Derivative of merit fucntion
function Dϕv(x, s, px, ps, μ, v)
    grad_ϕ_x = grad_f(x) + v*(jac_g_e(x)'*g_e(x) + jac_g_i(x)'*(g_i(x)-s))
    grad_ϕ_s = -μ*(1 ./ s) + v*((g_i(x)-s) .* (-s))

    return grad_ϕ_x'*px + grad_ϕ_s'*ps
end
# Backtracking function
function ipbacktrack(x, s, dx, ds, μ, αm)
    β = 0.5
    c = 0.4
    t = αm
    v = μ^2
    while (ϕv(x + t*dx, s + t*ds, μ, v) > ϕv(x, s, μ, v) + c*t*Dϕv(x, s, dx, ds, μ, v))
        t = β*t
    end
    return t
end


# Add error function
function E(x, s, λ_e, λ_i, μ) 
    v1 = grad_f(x) - jac_g_e(x)'*λ_e - jac_g_i(x)'*λ_i
    v2 = λ_i - μ*(1 ./ s)
    v3 = g_e(x)
    v4 = g_i(x) - s
    return max(norm(v1), norm(v2), norm(v3), norm(v4)) 
end

# alpha max calculations
function α_maxs(s, ds, λi, dλi)
    τ = 0.095
    αms = 0.00001
    αmλi = 0.00001
    for i in eachindex(s)
        if ds[i] < 0
            αms = max(-τ*s[i] / ds[i], αms)
        end
        if dλi[i] < 0
            αmλi = max(-τ*λi[i] / dλi[i], αmλi)
        end
    end
    # α_s_max = min((abs.(τ*s) ./ abs.(ds))...)       # Primal alpha_max
    # α_z_max = min((abs.(τ*λ_i) ./ abs.(dλ_i))...)   # Dual alpha_max
    
    return αms, αmλi
end

# function main()                        

x0 = [1.0, 5, 5, 1]
s0 = ones(9)
λe0 = ones(1)
λi0 = ones(9)
delta = zeros(4+2*9+1)
μ0 = 20
#E(x0, s0, λ_e0, λ_i0, μ0) > μ0
for i = 1:10
    global x0, s0, μ0, λi0, λe0 
    nx = length(x0)
    ns = length(s0)
    ne = length(λe0)
    ni = length(λi0)
    count = 0
    # alpha = 0.001
    # while E(x0, s0, λe0, λi0, μ0) > μ0
    for j = 1:1000
        # Compute the step and make views into the step
        delta .= kkt_mat(x0, s0, λe0, λi0) \ kkt_vec(x0, s0, λe0, λi0, μ0)
        dx = delta[1:nx]
        ds = delta[nx+1:nx+ns]
        dλe = -delta[nx+ns+1:nx+ns+ne]
        dλi =  -delta[nx+ns+ne+1:nx+ns+ne+ni]

        # # Computation of valid step length
        # alphap, alphad = α_maxs(s0, ds, λi0, dλi)                   # This  ensures that s and λ remain positive 
        # alpha = ipbacktrack(x0, s0, dx, ds, μ0, alpha)
        alphap = 0.0001
        alphad = 0.0001
        # Go in the direction of the step
        x0 .= x0 + alphap*dx
        s0 .= s0 + alphap*ds 
        λe0 .= λe0 + alphad*dλe
        λi0 .= λi0 + alphad*dλi

        # if alphad < 1e-5
        #     break
        # end
        println("-----------------")
        println(alphap)
        # println("Max limits")
        println(alphad)
        # println(αm)
        println("-----------------")
    end
    μ0 = 0.2*μ0
    println("Iter No:",i, "------",x0)
end
# return(x0)
# end