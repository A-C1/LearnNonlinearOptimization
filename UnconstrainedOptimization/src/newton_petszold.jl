using LinearAlgebra
 
function fv(x) 
    return [(x[1]+3)*(x[2]^3-7) + 18, sin(x[2]*exp(x[1]) - 1)]
end

function f_jac(x)
    f_grad = zeros(2, 2)
    f_grad[1, 1] = x[2]^3 - 7
    f_grad[1, 2] = 3*(3 + x[1])*(x[2]^2)
    f_grad[2, 1] = cos(exp(x[1])*x[2] - 1)*exp(x[1])*x[2]
    f_grad[2, 2] = cos(exp(x[1])*x[2] - 1)*exp(x[1])
    return f_grad
end


x0 = [-0.5, 1.4]
xstar = [0, 1]
function ls_newton(x0, xstar, f_jac, fv)
    for i = 1:10
        delta_x0 = (f_jac(x0)'f_jac(x0)) \ -f_jac(x0)'fv(x0)
        x0 = x0 + delta_x0
        # println(x0)
        # println(norm(x0 - xstar))
    end
end

# Better Idea is to do a QR factorization and solve a least squares problem
x0 = [-0.5, 1.4]
xstar = [0, 1]
println("---------------------------------------------")
function lsqr_newton(x0, xstar, f_jac, fv)
    for i = 1:10
        S = qr(f_jac(x0))
        delta_x0 = factorize(S.R)\(-S.Q'*fv(x0))
        x0 = x0 + delta_x0
        # println(x0)
        # println(norm(x0 - xstar))
    end
    return x0
end
