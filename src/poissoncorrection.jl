# This could be refactored in to a different project with all the modeling stuff

""" Compute expectation of the competition field in the window using quadrature and FFT.
"""
function poissoncorrection(d::Discretization, f)
    poissoncorrection(d.window, length(d.gridx), f)
end
function poissoncorrection(window, nx, f)
    (x0, x1), (y0, y1) = window.x, window.y
    dx = x1-x0
    dy = y1-y0
    ny = round(Int, nx/dx*dy)

    fk(x1, x2) = f(sqrt(x1^2+x2^2))
    f1 = Matrix{Float64}(undef, 2nx-1, 2ny-1)
    # for j in axes(f1, 2)
    #     for i in axes(f1, 1)
    #         f1[i, j] = fk((i-nx)*dx/nx, (j-ny)*dy/ny)
    #     end
    # end
    for j in 1:ny #axes(f1, 2)
        for i in 1:j #1:nx #axes(f1, 1)
            f1[i, j] = fk((i-nx)*dx/nx, (j-ny)*dy/ny)
            f1[j, i] = f1[i, j]
        end
    end
    for j in 1:ny #axes(f1, 2)
        for i in 1:nx #axes(f1, 1)
            f1[2nx-i, j] = f1[i, j]
            f1[2nx-i, 2ny-j] = f1[i, j]
            f1[i, 2ny-j] = f1[i, j]
        end
    end
    f0 = zero(f1)
    s2 = dx/nx*dy/ny
    for j in 1:ny
        for i in 1:nx
            f0[i, j] = s2
        end
    end
    # N = 2*gridn # The padding neceassary for periodicity
    # f1 = [fk(s*min(x, N-x), s*min(y, N-y)) for x in 0:1:N-1, y in 0:1:N-1]
    # f0 = [ifelse(x <= gridn && y <= gridn, 1.0*s^2, 0.0) for x in 1:N, y in 1:N]

    f2 = rfft(f1).*rfft(f0)
    fi = irfft(f2, size(f0, 1))
    I, E = quadgk(r -> r*f(r), 0.0, Inf)
    if !isfinite(I)
        error("Failed to integrate competition kernel.")
    end

    2*pi*I .- fi[nx:end, ny:end]
end
