import math
import numpy as np
from asteval import Interpreter
from scipy.special import factorial, gamma

def FODE_PECE(alpha, funcs, t0, T, y0, h, mu=1, mu_tol=1e-6, r=16):
    alpha = np.insert(alpha, 0, 0.0)

    # Calculate initial factorials
    integer_alpha = np.ceil(alpha).astype(np.int64)
    integer_alpha_factorial = np.zeros((np.max(integer_alpha) + 1, len(alpha)))

    for i in range(1, len(alpha)):
        for j in range(0, integer_alpha[i]):
            integer_alpha_factorial[j + 1][i] = factorial(j)
    integer_alpha_factorial.astype(np.int64)

    # Calculate initial points
    funcs_result = evaluate_functions(funcs, y0[:, 0], t0)

    # Calculate the number of points needed to evaluate weights and solution points
    N = math.ceil((T - t0)/h)
    rectN = math.ceil((N + 1)/r)*r
    logN = math.ceil(math.log2(rectN/r)) - 1
    powN = int(math.pow(2, (logN + 1)) * r)

    # Preallocate arrays for solutions and evaluations of predictor and corrector
    y = np.zeros((len(funcs), N+2))
    fy = np.zeros((len(funcs), N+2))
    predictor = np.zeros((len(funcs) + 1, powN+2))
    corrector = np.zeros((len(funcs) + 1, powN+2)) if mu != 0 else 0

    # Evaluation of the PECE method coefficients
    coef_vect = np.array(range(powN + 2))
    coef_vect = np.insert(coef_vect, 0, 0)
    a0 = np.zeros((len(alpha), powN+2))
    an = np.zeros((len(alpha), powN+2))
    bn = np.zeros((len(alpha), powN+2))

    for index_alpha in range(1, len(alpha)):
        find_alpha = np.where(alpha[1:index_alpha] == alpha[index_alpha])[0]
        if len(find_alpha) > 0:
            find_alpha = find_alpha[0]
            a0[index_alpha, 1:] = np.copy(a0[find_alpha, 1:])
            an[index_alpha, 1:] = np.copy(an[find_alpha, 1:])
            bn[index_alpha, 1:] = np.copy(bn[find_alpha, 1:])
        else:
            nalpha1 = coef_vect ** alpha[index_alpha]
            nalpha2 = nalpha1 * coef_vect
            a0[index_alpha, 1:] = np.concat(([0],
                nalpha2[1:-2] - nalpha1[2:-1] * (coef_vect[2:-1] - alpha[index_alpha] - 1)))
            an[index_alpha, 1:] = np.concat(([1],
                nalpha2[1:-2] - 2 * nalpha2[2:-1] + nalpha2[3:]))
            bn[index_alpha, 1:] = nalpha1[2:] - nalpha1[1:-1]

    halpha1 = h ** alpha / gamma(alpha + 1)
    halpha2 = h ** alpha / gamma(alpha + 2)

    # Evaluation of the FFT and its coefficients
    if logN >= 0:
        index_fft = np.zeros((3, logN+2))
        index_fft[1, 1] = 1
        index_fft[2, 1] = r*2

        for l in range(2, logN + 2):
            index_fft[1, l] = index_fft[2, l-1] + 1
            index_fft[2, l] = index_fft[2, l-1] + 2 ** l * r
        index_fft = index_fft.astype(np.int64)

        an_fft = np.zeros((len(alpha), index_fft[2, -1] + 1), dtype=complex)
        bn_fft = np.zeros((len(alpha), index_fft[2, -1] + 1), dtype=complex)
        for l in range(1, logN + 2):
            coef_end = 2 ** l * r
            for index_alpha in range(1, len(alpha)):
                find_alpha = np.where(alpha[1:index_alpha] == alpha[index_alpha])[0]
                if len(find_alpha) > 0:
                    find_alpha = find_alpha[0]
                    an_fft[index_alpha, index_fft[1, l]:index_fft[2, l] + 1] = \
                        np.copy(an_fft[find_alpha, index_fft[1, l]:index_fft[2, l] + 1])
                    bn_fft[index_alpha, index_fft[1, l]:index_fft[2, l] + 1] = \
                        np.copy(bn_fft[find_alpha, index_fft[1, l]:index_fft[2, l] + 1])
                else:
                    an_fft[index_alpha, index_fft[1, l]:index_fft[2, l] + 1] = \
                        np.fft.fft(an[index_alpha, 1:coef_end+1], coef_end)
                    bn_fft[index_alpha, index_fft[1, l]:index_fft[2, l] + 1] = \
                        np.fft.fft(bn[index_alpha, 1:coef_end+1], coef_end)

    # Initialization of the solution and the computation process
    t = t0 + np.array(range(N+1)) * h
    t = np.insert(t, 0, 0)
    y[:, 1] = np.copy(y0[:, 0])
    fy[:, 1] = np.copy(funcs_result)
    triangulate(1, r-1, t, y, fy, predictor, corrector, N, mu, mu_tol, a0, an, bn, halpha1, halpha2, funcs, t0, y0, integer_alpha, integer_alpha_factorial)

    # Main process of the computation by means of the FFT
    ff = np.zeros(2**(logN + 2) + 1)
    ff[2] = 2
    card_ff = 2
    nx0 = 0
    ny0 = 0

    for i in range(logN + 1):
        L = 2 ** i
        create_blocks(L, ff, r, rectN, nx0 + L * r, ny0, t, y, fy, predictor, corrector, N, mu, mu_tol, a0, an, bn, halpha1, halpha2, an_fft, bn_fft, index_fft, funcs, alpha, t0, y0, integer_alpha, integer_alpha_factorial)
        ff[1:2 * card_ff + 1] = np.hstack((np.copy(ff[1:card_ff + 1]), np.copy(ff[1:card_ff + 1])))
        card_ff *= 2
        ff[card_ff] = 4 * L

    return t[1:], y[:, 1:], fy[:, 1:]

def triangulate(nxi, nxf, t, y, fy, predictor, corrector, N,
                mu, mu_tol, a0, an, bn, halpha1, halpha2,
                funcs, t0, y0, integer_alpha, integer_alpha_factorial):
    # Seems to work alright
    for n in range(nxi, min(N, nxf) + 1):

        # Evaluation of the predictor
        phi = np.zeros(len(funcs) + 1)
        for j in range(0 if nxi == 1 else nxi, n):
            phi[1:] += bn[1:, n-j] * fy[:, j+1]

        st = equation_starting_terms(t[n+1], t0, y0, integer_alpha, integer_alpha_factorial)
        y_predicted = (st + halpha1 * (predictor[:, n+1] + phi)).T[1:]
        fy_predicted = evaluate_functions(funcs, y_predicted, t[n+1])

        # Evaluation of the corrector
        if mu == 0:
            y[:, n+1] = y_predicted.T
            fy[:, n+1] = fy_predicted.T
        else:
            phi = np.zeros(len(funcs) + 1)
            for j in range(nxi, n):
                phi[1:] += an[1:, n-j+1] * fy[:, j+1]
            nphi = st[1:] + halpha2[1:] * (a0[1:, n+1] * fy[:, 1] + corrector[1:, n+1] + phi[1:])

            yn0 = y_predicted.T
            fn0 = fy_predicted.T
            stop = False
            mu_iter = 0
            while not stop:
                mu_iter += 1
                yn1 = (nphi + halpha2[1:] * fn0).T
                if mu == np.inf:
                    stop = (np.linalg.norm(yn1-yn0)) < mu_tol
                    if mu_iter == 100 and stop:
                        raise ValueError('The corrector iteration will not converge, stopping the calculation')
                else:
                    stop = (mu_iter == mu)
                fn1 = evaluate_functions(funcs, yn1, t[n+1])
                yn0 = yn1
                fn0 = fn1
            y[:, n+1] = yn0.T
            fy[:, n+1] = fn0.T

def squarify(nxi, nxf, nyi, nyf, fy, predictor, corrector, N,
             r, an_fft, bn_fft, index_fft, mu,
             funcs):
    coefficient_end = nxf - nyi + 1
    i_fft = np.floor(np.log2(coefficient_end / r)).astype(np.int64)
    funcz_begin = nyi + 1
    funcz_end = nyf + 1
    nxfN = min(N, nxf)

    # Evaluation of the convolution segment for the predictor
    conv_funcz = np.copy(fy[:, funcz_begin:funcz_end+1])
    conv_funcz_fft = np.fft.fft(conv_funcz, coefficient_end, axis=1)
    z_predicted = np.zeros((len(funcs) + 1, coefficient_end + 1))

    for i in range(1, len(funcs) + 1):
        Z = bn_fft[i, index_fft[1, i_fft]:index_fft[2, i_fft]+1] * conv_funcz_fft[i-1, :]
        z_predicted[i, 1:] = np.real(np.fft.ifft(Z, coefficient_end))
    z_predicted = z_predicted[:, nxf-nyf:-1]
    predictor[1:, nxi + 1:nxfN + 2] += np.copy(z_predicted[1:, :nxfN - nxi + 1])

    # Evaluation of the convolution segment for the corrector
    if mu > 0:
        if nyi == 0: # Evaluation of the lowest square
            conv_funcz = np.hstack((np.zeros((len(funcs), 1)), fy[:, funcz_begin + 1:funcz_end + 1]))
            conv_funcz_fft = np.fft.fft(conv_funcz, coefficient_end, axis=1)
        z_predicted = np.zeros((len(funcs) + 1, coefficient_end + 1))

        for i in range(1, len(funcs) + 1):
            Z = an_fft[i, index_fft[1, i_fft]:index_fft[2, i_fft] + 1] * conv_funcz_fft[i-1, :]
            z_predicted[i, 1:] = np.real(np.fft.ifft(Z, coefficient_end))
        z_predicted = z_predicted[:, nxf - nyf + 1:]
        corrector[1:, nxi + 1:nxfN + 2] += np.copy(z_predicted[1:, :nxfN - nxi + 1])
    else:
        corrector = 0

def create_blocks(L, ff, r, rectN, nx0, ny0, t, y, fy, predictor, corrector, N,
                  mu, mu_tol, a0, an, bn, halpha1, halpha2, an_fft, bn_fft, index_fft,
                  funcs, alpha, t0, y0, integer_alpha, integer_alpha_factorial):
    nxi = nx0
    nxf = nx0 + L*r - 1
    nyi = ny0
    nyf = ny0 + L*r - 1
    s_nxf = nxf
    index_triangulate = 1
    stop = False

    while not stop:
        stop = (nxi + r - 1 == nx0 + L*r - 1) or (nxi + r - 1 >= rectN - 1)
        squarify(nxi, nxf, nyi, nyf, fy, predictor, corrector, N, r, an_fft, bn_fft, index_fft, mu, funcs)
        triangulate(nxi, nxi + r - 1, t, y, fy, predictor, corrector, N, mu, mu_tol, a0, an, bn, halpha1, halpha2, funcs, t0, y0, integer_alpha, integer_alpha_factorial)
        if not stop:
            if nxi + r - 1 == nxf:
                index_delta = ff[index_triangulate]
                delta = int(index_delta * r)
                nxi = s_nxf + 1
                nxf = s_nxf + delta
                nyi = s_nxf - delta + 1
                nyf = s_nxf
            else:
                nxi = nxi + r
                nxf = nxi + r - 1
                nyi = nyf + 1
                nyf = nyf + r

        s_nxf = nxf
        index_triangulate += 1

def equation_starting_terms(t, t0, y0, integer_alpha, integer_alpha_factorial):
    st = np.zeros((len(y0) + 1, 1))
    for k in range(1, np.max(integer_alpha) + 1):
        if len(integer_alpha) == 2:
            st += (t-t0) ** (k-1) * y0[:, k-1] / integer_alpha_factorial[k, 1]
        else:
            index_alpha = np.where(k <= integer_alpha)[0]
            st[index_alpha,0] += (t-t0) ** (k-1) * y0[index_alpha - 1, k - 1] / integer_alpha_factorial[k, index_alpha]
    return st.T[0]

def evaluate_functions(funcs, y, t):
    aeval = Interpreter()
    results = []
    aeval.symtable['t'] = t
    for index, value in enumerate(y):
        aeval.symtable[f'y{index}'] = y[index]
    for func in funcs:
        results.append(aeval(func))
    results = np.array(results)
    return results
