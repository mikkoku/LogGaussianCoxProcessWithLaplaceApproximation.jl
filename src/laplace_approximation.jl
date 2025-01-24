struct LaplaceApproximation{SM, F,O,YT,XT,WT}
    size::Tuple{Int, Int}
    Q::SM # Sparse precision matrix
    # H and chol in obj
    # H::SM # Hessian
    # chol::CH # Symbolic factorization of Q
    lik::F # loglikelihood, derivative and second derivative.
    obj::O # Optim.TwiceDifferentiable
    y::YT # Response
    x::XT # Fixed effect ???
    w::WT # Integration weights
end
function poissonlikelihood(y, x, z)
    exz = exp(x+z)
    y*(x+z) - exz, y - exz, -exz
end

function poisson_gp(locationweight, fixed_field, prec, range, nu, integration_weights, gridsize)
    Q = Qmatrix_matern2d(gridsize, prec, range, nu)
    chol = cholesky(Q)
    Qlogdet = logdet(chol)

    la = LaplaceApproximation(gridsize, locationweight, fill(integration_weights, gridsize),
        Q.data, chol)
    poisson_gp(la, fixed_field, Q, Qlogdet)
end
function LA_factors(zhat, H, Q, Qlogdet)
    d = length(zhat)

    pz = 0.5*Qlogdet - d/2*log(2*pi) - 0.5*(zhat'*Q*zhat)

    m = d/2*log(2*pi) - 0.5*logdet(H)

    a = m + pz
    ifelse(isnan(a), -Inf, a), m, pz
end

function poisson_gp(la::LaplaceApproximation, fixed_field, Q, Qlogdet)
    zhat, H = poisson_hatZ(la, fixed_field, Q)
    # H is the hessian of the negative loglikelihood
    x = zhat

    l = LA_factors(zhat, H, Q, Qlogdet)
    l, reshape(zhat, la.size), H, Q
end
function LaplaceApproximation(gridsize, locationweight, integration_weights, Q, chol)
    H = copy(Q)
    y = locationweight
    x = zeros(gridsize)
    w = integration_weights

    obj = LA_poisson_objective_chol(y, x, w, Q, H, chol)
    LaplaceApproximation(gridsize, Q, #copy(Q), cholesky(Q),
        nothing, obj, y, x, w)
end

function poisson_hatZ(la::LaplaceApproximation, fixed_field::Number, Q)
    fill!(la.x, fixed_field)
    poisson_hatZ(la, Q)
end
function poisson_hatZ(la::LaplaceApproximation, fixed_field, Q)
    copy!(la.x, fixed_field)
    poisson_hatZ(la, Q)
end
function poisson_hatZ(la::LaplaceApproximation, Q)
    copy!(la.Q, Q.data)
    initial = vec(initial_guess(la.x, la.y, Q[1]))
    obj = la.obj

    o = try
        Optim.optimize(obj, initial)
    catch e
        if e isa AssertionError
            println(e)
        end
        rethrow()
    end
    if !Optim.converged(o)
        @warn("Newton Not converged.")
    end
    x = Optim.minimizer(o)
    H = Optim.hessian!(obj, x)

    x, H
end


# Here Q is sparse matrix
function LA_poisson_objective_chol(locationweight, fixed_field, integration_weights, Q, H, cholQ)
    function fgh!(F, D, chol, x)
        # _negative_ log likelihood
        random_field = x
        if chol !== nothing
            copy!(H, Q)
        end
        if D !== nothing || F !== nothing
            Qx = Q*x
            if D !== nothing
                D .= Qx
                weightstonodes!(D, locationweight, -1)
            end
            if F !== nothing
                xy = sum(locationweight) do (i, w)
                    w*random_field[i]
                end
                F = 0.5 * x'*Qx - xy
            end
        end
        for i in eachindex(random_field)
            a = integration_weights[i] * exp(fixed_field[i] + random_field[i])
            if !isfinite(a)

                @warn("Infinite a")
                return -Inf
            end
            F !== nothing && (F += a)
            D !== nothing && (D[i] += a)
            chol !== nothing && (H[i, i] += a)
        end
        if chol !== nothing
            cholesky!(chol.data, Symmetric(H))
        end
        if F !== nothing
            return F
        end
    end
    initial = vec(zero(fixed_field))
    Optim.TwiceDifferentiable(Optim.only_fgh!(fgh!), initial, 0.0, copy(initial), OptimCholWrapper(cholQ))
end

# If Q is a dense matrix some things are different
function LA_poisson_objective_chol(locationweight, fixed_field, integration_weights, Q::Matrix{Float64}, H, cholQ)
    function fgh!(F, D, H, x)
        # _negative_ log likelihood
        random_field = x
        if H !== nothing
            copy!(H, Q)
        end
        if D !== nothing || F !== nothing
            Qx = Q*x
            if D !== nothing
                D .= Qx
                weightstonodes!(D, locationweight, -1)
            end
            if F !== nothing
                xy = sum(locationweight) do (i, w)
                    w*random_field[i]
                end
                F = 0.5 * x'*Qx - xy
            end
        end
        for i in eachindex(random_field)
            a = integration_weights[i] * exp(fixed_field[i] + random_field[i])
            if !isfinite(a)

                @warn("Infinite a")
                return -Inf
            end
            F !== nothing && (F += a)
            D !== nothing && (D[i] += a)
            H !== nothing && (H[i, i] += a)
        end
        return F
    end
    initial = vec(zero(fixed_field))
    Optim.TwiceDifferentiable(Optim.only_fgh!(fgh!), initial, 0.0, copy(initial), copy(Q))
end
