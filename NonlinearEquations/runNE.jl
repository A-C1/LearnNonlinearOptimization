include("NonlinearEquations.jl")

import .NonlinearEquations as NL

function f(x)
    return [(x[1] + 3)*(x[2]^2 - 7) + 18;
            sin(x[2]*exp(x[1]) - 1)]
end

f_jac(x) = [-7 + x[2]^2                                 2*(3 + x[1])*x[2]
            x[2]*cos(-1 + x[2]*exp(x[1]))*exp(x[1])     cos(-1 + x[2]*exp(x[1]))*exp(x[1])]

x0 = [-0.5, 1.4]

xf = NL.newton_pure(f, f_jac, x0, atol = 1e-8)