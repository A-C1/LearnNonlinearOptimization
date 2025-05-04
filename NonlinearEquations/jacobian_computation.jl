using Symbolics

function f(x)
    return [(x[1] + 3)*(x[2]^2 - 7) + 18 
            sin(x[2]*exp(x[1] - 1))]
end

Symbolics.jacobian(vec(f(x)), vec(x))