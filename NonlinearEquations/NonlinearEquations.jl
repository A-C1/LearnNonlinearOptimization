module NonlinearEquations

import LinearAlgebraLocal as LA
using LinearAlgebra

function newton_pure(f, jacf, x0; atol = 1e-6)
    iter_no = 0
    while norm(f(x0)) > atol
        println("MN:Iteration Number: ", iter_no, "  ||~~~||  ", "Value of norm of f: ",norm(f(x0)))

        p = -LA.solve(jacf(x0)'*jacf(x0), jacf(x0)'*f(x0))
        # p = -jacf(x0) \ f(x0)

        x0 .= x0 + p

        iter_no = iter_no + 1
    end
    
    return x0
end

function levenberg_maqquardt(f, jacf, x0; atol = 1e-6)
    iter_no = 0
    while norm(f(x0)) > atol
        println("Iteration number:  ", iter_no, "   ||~~~||   ", "Value of norm of f: ", norm(f(x0)))

        p = -jacf(x0) \ f(x0)
        x0 .= x0 + p

        iter_no = iter_no + 1
    end
end
    
end