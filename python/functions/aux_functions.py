"""
Auxiliary funtions that replicate Plagborg-Moller and Wolf (2020)
Author: Joao B. Duarte
Last update: 2021-04-14
This is an alpha version of the code. Any bugs or mistakes are my own. If you find them please report them to joao.duarte@novasbe.pt
"""
import math
import numpy as np
from scipy.stats import chi2
from scipy.stats import invgauss
from scipy.stats import norm
from scipy import linalg
from scipy import optimize as opt
from statsmodels.tsa.tsatools import lagmat
from scipy.optimize import NonlinearConstraint
import scipy.integrate as integrate


def stoye_bound(c, rho, Delta_over_sigma):
    # Integral used in Stoye(2009) confidence interval routine
    # See Appendix B of Stoye(2009)
    integr = integrate.quad(lambda z: norm.cdf((rho * z + c(2) + Delta_over_sigma) / np.sqrt(1 - rho ** 2)) * norm.pdf(z),
                          - 10,
                          c[0])
    return integr


def stoye_CI(lower_estim, upper_estim, varcov, signif_level):
    # Stoye (2009) confidence interval for partially identified parameter
    # Inputs to routine
    Delta = upper_estim - lower_estim  # Point estimate of length of identif. set
    sigma_lower = np.sqrt(varcov[0, 0])  # Std. dev. of lower bound estimate
    sigma_upper = np.sqrt(varcov[1, 1])  # Std. dev. of upper bound estimate
    rho = varcov[0, 1] / (sigma_lower * sigma_upper)  # Correlation of lower and upper bound estimates

    # Numerically minimize CI length subject to coverage constraints
    con = lambda c: [0, 1 - signif_level - np.r_[stoye_bound(c, rho, Delta / sigma_upper),
                                                 stoye_bound(np.fliplr(c), rho, Delta / sigma_lower)]]
    nlc = NonlinearConstraint(con, -np.inf, 1.9)
    c_opt = opt.minimize(lambda c: np.c_[sigma_lower, sigma_upper] @ c.T,
                         x0=np.array([lower_estim - invgauss.cdf(1 - signif_level / 2) * sigma_lower,
                                      upper_estim + invgauss.cdf(1 - signif_level / 2) * sigma_upper]),
                         constraints=nlc)

    # Confidence interval
    CI = np.c_[lower_estim - sigma_lower * c_opt(1), upper_estim + sigma_upper * c_opt(2)]
    return CI


def CI_fun(bounds_boot, bounds_OLS, settings):
    # Bootstrap confidence intervals

    # ----------------------------------------------------------------
    # Get Inputs
    # ----------------------------------------------------------------

    fields = list(bounds_OLS.__dict__.keys())
    signif_level = settings.signif_level

    # optimopts = settings.optimopts

    # ----------------------------------------------------------------
    # Quantiles of Bootstrap Draws
    # ----------------------------------------------------------------
    class bounds_boot_mean():
        for j in range(len(fields)):
            locals()[fields[j]] = np.squeeze(np.mean(getattr(bounds_boot, fields[j]), 2))  # Average

    class bounds_boot_plow():
        for j in range(len(fields)):
            if type(getattr(bounds_OLS, fields[j])) == np.float64:
                locals()[fields[j]] = np.squeeze(np.quantile(getattr(bounds_boot, fields[j]) - (getattr(bounds_OLS,
                                                                                                        fields[j])),
                                                             signif_level / 2, axis=2))  # Lower quantile
            elif type(getattr(bounds_OLS, fields[j])) == np.ndarray:
                locals()[fields[j]] = np.squeeze(np.quantile(getattr(bounds_boot, fields[j]) - (getattr(bounds_OLS,
                                                                                                        fields[j]))[:,
                                                                                               :,
                                                                                               np.newaxis],
                                                             signif_level / 2, axis=2))  # Lower quantile
            else:
                locals()[fields[j]] = []

    class bounds_boot_phigh():
        for j in range(len(fields)):
            if type(getattr(bounds_OLS, fields[j])) == np.float64:
                locals()[fields[j]] = np.squeeze(np.quantile(getattr(bounds_boot, fields[j]) - (getattr(bounds_OLS,
                                                                                                        fields[j])),
                                                             1 - signif_level / 2, axis=2))  # Upper quantile
            elif type(getattr(bounds_OLS, fields[j])) == np.ndarray:
                locals()[fields[j]] = np.squeeze(np.quantile(getattr(bounds_boot, fields[j]) - (getattr(bounds_OLS,
                                                                                                        fields[j]))[:,
                                                                                               :,
                                                                                               np.newaxis],
                                                             1 - signif_level / 2, axis=2))  # Upper quantile
            else:
                locals()[fields[j]] = []

    bounds_boot_mean = bounds_boot_mean()
    bounds_boot_plow = bounds_boot_plow()
    bounds_boot_phigh = bounds_boot_phigh()

    # ----------------------------------------------------------------
    # CI for IS
    # ----------------------------------------------------------------

    class bounds_CI_IS():
        class OLS_biascorr():
            for j in range(len(fields)):
                locals()[fields[j]] = 2 * getattr(bounds_OLS, fields[j]) - getattr(bounds_boot_mean, fields[j])  #
                # Bias correction

        class lower():
            for j in range(len(fields)):
                if getattr(bounds_OLS, fields[j]) == []:
                    locals()[fields[j]] = []
                else:
                    locals()[fields[j]] = getattr(bounds_OLS, fields[j]) - getattr(bounds_boot_phigh,
                                                                                   fields[j])  # Hall's
                # bootstrap percentile interval, lower bound

        class upper():
            for j in range(len(fields)):
                if getattr(bounds_OLS, fields[j]) == []:
                    locals()[fields[j]] = []
                else:
                    locals()[fields[j]] = getattr(bounds_OLS, fields[j]) - getattr(bounds_boot_plow,
                                                                                   fields[j])  # Hall's

    bounds_CI_IS = bounds_CI_IS()
    bounds_CI_IS.OLS_biascorr = bounds_CI_IS.OLS_biascorr()
    bounds_CI_IS.lower = bounds_CI_IS.lower()
    bounds_CI_IS.upper = bounds_CI_IS.upper()

    # ----------------------------------------------------------------
    # CI for Parameter (Stoye, 2009)
    # ----------------------------------------------------------------

    bounds_CI_para = {'lower': {}, 'upper': {}}

    if not settings.CI_para:
        return bounds_CI_IS, bounds_CI_para

    for j in range(len(fields)):
        lb_pos = fields[j].index('_LB')
        if lb_pos == None:
            continue
        else:
            the_param = fields[j][:lb_pos]  # Name of parameter

        field_LB = fields[j]  # Lower bound field
        field_UB = the_param + '_UB'  # Upper bound field

        if type(getattr(bounds_OLS, field_LB)) == np.float64:
            bounds_CI_para['lower'][the_param] = np.empty(1)
            bounds_CI_para['upper'][the_param] = np.empty(1)
            size_l = np.size(getattr(bounds_OLS, field_LB))
            size_m = np.size(getattr(bounds_OLS, field_LB))
        else:
            bounds_CI_para['lower'][the_param] = np.empty(getattr(bounds_OLS, field_LB).shape)
            bounds_CI_para['upper'][the_param] = np.empty(getattr(bounds_OLS, field_LB).shape)
            size_l = np.size(getattr(bounds_OLS, field_LB), 0)
            size_m = np.size(getattr(bounds_OLS, field_LB), 1)
        for l in range(size_l):
            for m in range(size_m):
                # Bootstrap var-cov matrix of estimated lower and upper bounds
                varcov = np.cov(
                    np.column_stack((np.squeeze(getattr(bounds_boot, field_LB)[l, m, :]), np.squeeze(getattr(
                        bounds_boot, field_UB)[l, m, :]))))
                # Enforce parameter in [0,1] (except alpha)
                if the_param != 'alpha':
                    if type(getattr(bounds_CI_IS.OLS_biascorr, field_LB)) == np.float64:
                        getattr(bounds_CI_IS.OLS_biascorr, field_LB)[l, m] = max(0, getattr(bounds_CI_IS.OLS_biascorr,
                                                                                            field_LB))
                        getattr(bounds_CI_IS.OLS_biascorr, field_UB)[l, m] = max(0, getattr(bounds_CI_IS.OLS_biascorr,
                                                                                            field_UB))
                    else:
                        getattr(bounds_CI_IS.OLS_biascorr, field_LB)[l, m] = max(0, getattr(bounds_CI_IS.OLS_biascorr,
                                                                                            field_LB)[l, m])
                        getattr(bounds_CI_IS.OLS_biascorr, field_UB)[l, m] = max(0, getattr(bounds_CI_IS.OLS_biascorr,
                                                                                            field_UB)[l, m])

                if not (('FVD_LB' == field_LB) or ('R2_recov_LB' == field_LB)):
                    # Compute Stoye (2009) confidence interval
                    if type(getattr(bounds_CI_IS.OLS_biascorr, field_LB)) == np.float64:
                        CI = stoye_CI(getattr(bounds_CI_IS.OLS_biascorr, field_LB),
                                      getattr(bounds_CI_IS.OLS_biascorr, field_UB),
                                      varcov, signif_level)
                        bounds_CI_para['lower'][the_param] = CI[0]
                        bounds_CI_para['upper'][the_param] = CI[1]
                    else:
                        CI = stoye_CI(getattr(bounds_CI_IS.OLS_biascorr, field_LB)[l, m],
                                      getattr(bounds_CI_IS.OLS_biascorr, field_UB)[l, m],
                                      varcov,
                                      signif_level)
                        bounds_CI_para['lower'][the_param][l, m] = CI[0]
                        bounds_CI_para['upper'][the_param][l, m] = CI[1]
                else:
                    # FVD and R2_recov: one-sided lower confidence interval,
                    # since upper bound is always 1
                    if type(getattr(bounds_CI_IS.OLS_biascorr, field_LB)) == np.float64:
                        bounds_CI_para['lower'][the_param] = getattr(bounds_CI_IS.OLS_biascorr,
                                                                     field_LB) + invgauss.cdf(signif_level) * np.sqrt(
                            varcov[0, 0])
                        bounds_CI_para['lower'][the_param] = 1
                    else:
                        bounds_CI_para['lower'][the_param][l, m] = getattr(bounds_CI_IS.OLS_biascorr,
                                                                           field_LB) + invgauss.cdf(
                            signif_level) * np.sqrt(
                            varcov[0, 0])
                        bounds_CI_para['lower'][the_param][l, m] = 1
    return bounds_CI_IS, bounds_CI_para


def FVD_fun(var, hor, yzt_aux, alpha):
    # Forecast variance decomposition bound

    Sigma_yzt = yzt_aux.Sigma_yzt
    maxVar_yt_ythor = yzt_aux.maxVar_yt_ythor[:, :, hor - 1]
    FVD = sum(((1 / alpha) * Sigma_yzt[:hor, var]) ** 2) / (
            sum(((1 / alpha) * Sigma_yzt[:hor, var]) ** 2) + maxVar_yt_ythor[var, var])
    return FVD


def FVD_IS(yzt_aux, dataobj, settings, alpha):
    # Identified set for forecast variance decomposition

    # ----------------------------------------------------------------
    # Preparations
    # ----------------------------------------------------------------

    n_y = dataobj.n_y
    FVD_hor = settings.FVD_hor
    FVD_LB = np.empty((np.size(FVD_hor, 0), n_y))
    VMA_hor = settings.VMA_hor
    alpha_UB = alpha.alpha_UB
    Sigma_yzt = yzt_aux.Sigma_yzt

    Cov_y_shock1_LB = [None] * VMA_hor
    Cov_yt_LB = [None] * VMA_hor
    for i in range(VMA_hor):
        Cov_y_shock1_LB[i] = 0
        for j in range((VMA_hor - (i + 1) + 1)):
            Cov_y_shock1_LB[i] = Cov_y_shock1_LB[i] + ((1 / alpha_UB) * Sigma_yzt[j, :][:, np.newaxis]) @ \
                                 ((1 / alpha_UB) * Sigma_yzt[j + i, :][np.newaxis, :])
        Cov_yt_LB[i] = yzt_aux.Cov_y[i] - Cov_y_shock1_LB[i]

    Sigma_yt_big_LB = np.empty((n_y * VMA_hor, n_y * VMA_hor))
    for i in range(VMA_hor):
        for j in range(VMA_hor):
            if i > j:
                Sigma_yt_big_LB[i * n_y: (i + 1) * n_y, j * n_y: (j + 1) * n_y] = Cov_yt_LB[abs(i - j)]
            else:
                Sigma_yt_big_LB[i * n_y: (i + 1) * n_y, j * n_y: (j + 1) * n_y] = Cov_yt_LB[abs(i - j)].T

    yzt_aux.maxVar_yt_ythor = np.empty((n_y, n_y, FVD_hor[-1]))

    for hor in range(FVD_hor[-1]):
        Sigma_yt_ythor = Sigma_yt_big_LB
        Cov_yt = Cov_yt_LB
        if hor >= 1:
            for j in range(1, VMA_hor - hor):
                Sigma_yt_ythor[:n_y, j * n_y: (j + 1) * n_y] = Cov_yt[abs((j + hor))].T
            Sigma_yt_ythor[:n_y, (VMA_hor - hor + 2) * n_y:] = 0
            for i in range(1, VMA_hor - hor):
                Sigma_yt_ythor[i * n_y: (i + 1) * n_y, :n_y] = Cov_yt[abs(i + hor)]
            Sigma_yt_ythor[(VMA_hor - hor + 2) * n_y:, :n_y] = 0
        yzt_aux.maxVar_yt_ythor[:, :, hor] = Sigma_yt_ythor[:n_y, :n_y] - Sigma_yt_ythor[:n_y, n_y:] \
                                             @ np.linalg.solve(Sigma_yt_ythor[n_y:, n_y:], Sigma_yt_ythor[n_y:, :n_y])

    # ----------------------------------------------------------------
    # Lower Bound
    # ----------------------------------------------------------------

    for hor_indx in range(np.size(FVD_hor, 0)):
        hor = FVD_hor[hor_indx]
        for j in range(n_y):
            FVD_LB[hor_indx, j] = FVD_fun(j, hor, yzt_aux, alpha.alpha_UB)
    return FVD_LB


def FVR_fun(var, hor, yzt_aux, alpha):
    # Forecast variance ratio, given alpha
    Sigma_yzt = yzt_aux.Sigma_yzt
    Var_y_yhor = yzt_aux.Var_y_yhor[:, :, hor - 1]
    FVR = np.sum(((1 / alpha) * Sigma_yzt[:hor, var]) ** 2) / Var_y_yhor[var, var]
    return FVR


def FVR_IS(yzt_aux, dataobj, settings, alpha):
    # Identified set for forecast variance ratio

    # ----------------------------------------------------------------
    # Preparations
    # ----------------------------------------------------------------

    # imports

    n_y = dataobj.n_y
    FVR_hor = settings.FVR_hor
    FVR_true = np.empty((np.size(FVR_hor, 0), n_y))
    FVR_UB = np.empty((np.size(FVR_hor, 0), n_y))
    FVR_LB = np.empty((np.size(FVR_hor, 0), n_y))

    VMA_hor = settings.VMA_hor
    Sigma_y_yhor = yzt_aux.Sigma_y_big
    Cov_y = yzt_aux.Cov_y

    # get forecasting variance

    yzt_aux.Var_y_yhor = np.empty((n_y, n_y, FVR_hor[-1]))

    if settings.use_KF == 0:
        for hor in range(FVR_hor[-1]):
            if hor >= 1:
                for j in range(1, VMA_hor - hor):
                    Sigma_y_yhor[:n_y, j * n_y: (j + 1) * n_y] = Cov_y[abs(j + hor)].T
                Sigma_y_yhor[:n_y, (VMA_hor - hor + 2) * n_y:] = 0
                for i in range(1, VMA_hor - hor):
                    Sigma_y_yhor[i * n_y: (i + 1) * n_y, :n_y] = Cov_y[abs(i + hor)]
                Sigma_y_yhor[(VMA_hor - hor + 2) * n_y:, :n_y] = 0

            yzt_aux.Var_y_yhor[:, :, hor] = Sigma_y_yhor[:n_y, :n_y] - \
                                            Sigma_y_yhor[:n_y, n_y:] @ np.linalg.solve(Sigma_y_yhor[n_y:, n_y:],
                                                                                       Sigma_y_yhor[n_y:, :n_y])
    else:
        A_KF = yzt_aux.A_KF
        B_KF = yzt_aux.B_KF
        C_KF = yzt_aux.C_KF
        cvar_states_1 = yzt_aux.cvar_states_1

        for hor in range(FVR_hor[-1]):
            if hor == 0:
                cvar_states = cvar_states_1
            else:
                cvar_states = A_KF @ cvar_states @ A_KF.T + B_KF @ B_KF.T

            yzt_aux.Var_y_yhor[:, :, hor] = C_KF @ cvar_states @ C_KF.T

    # ----------------------------------------------------------------
    # Truth
    # ----------------------------------------------------------------

    if hasattr(alpha, 'alpha_true') == 0:
        for hor_indx in range(np.size(FVR_hor, 0)):
            hor = FVR_hor[hor_indx]
            for j in range(n_y):
                FVR_true[hor_indx, j] = FVR_fun(j, hor, yzt_aux, alpha.alpha_true)
    else:
        FVR_true = []

    # ----------------------------------------------------------------
    # Upper Bound
    # ----------------------------------------------------------------
    for hor_indx in range(np.size(FVR_hor, 0)):
        hor = FVR_hor[hor_indx]
        for j in range(n_y):
            FVR_UB[hor_indx, j] = FVR_fun(j, hor, yzt_aux, alpha.alpha_LB)

    # ----------------------------------------------------------------
    # Lower Bound
    # ----------------------------------------------------------------
    for hor_indx in range(np.size(FVR_hor, 0)):
        hor = FVR_hor[hor_indx]
        for j in range(n_y):
            FVR_LB[hor_indx, j] = FVR_fun(j, hor, yzt_aux, alpha.alpha_UB)

    return FVR_LB, FVR_UB, FVR_true


def R2_IS(yzt_aux, dataobj, settings, R2_hor, alpha):
    # Identified set for degree of invertibility out to time t+l
    # ----------------------------------------------------------------
    # Get Inputs
    # ----------------------------------------------------------------
    Var_zt = yzt_aux.Var_zt
    Sigma_yzt = yzt_aux.Sigma_yzt
    Sigma_y_big = yzt_aux.Sigma_y_big
    VMA_hor = settings.VMA_hor
    n_y = dataobj.n_y
    alpha_true = alpha.alpha_true
    alpha_LB = alpha.alpha_LB
    alpha_UB = alpha.alpha_UB

    # ----------------------------------------------------------------
    # Auxiliary Computations
    # ----------------------------------------------------------------

    Cov_zty_lagged = np.zeros((1, n_y * VMA_hor))
    for i in range(R2_hor):
        Cov_zty_lagged[0, i * n_y: (i + 1) * n_y] = Sigma_yzt[R2_hor - i -1, :]

    Var_zt_yhor = Var_zt - Cov_zty_lagged @ np.linalg.solve(Sigma_y_big, Cov_zty_lagged.T)

    Rtilde2 = 1 - Var_zt_yhor[0, 0] / Var_zt

    # ----------------------------------------------------------------
    # Truth
    # ----------------------------------------------------------------
    if (not alpha.alpha_true) == 0:
        R2_true = Var_zt / alpha_true ** 2 * Rtilde2
    else:
        R2_true = []

    # ----------------------------------------------------------------
    # Upper Bound
    # ----------------------------------------------------------------
    R2_UB = Var_zt / alpha_LB ** 2 * Rtilde2

    # ----------------------------------------------------------------
    # Lower Bound
    # ----------------------------------------------------------------
    R2_LB = Var_zt / alpha_UB ** 2 * Rtilde2
    return R2_LB, R2_UB, R2_true


def alpha_IS(yzt_aux, dataobj, settings):
    # Identified set for scale parameter alpha

    # ----------------------------------------------------------------
    # Get Inputs
    # ----------------------------------------------------------------
    global alpha_LB
    Var_zt = yzt_aux.Var_zt
    s_y = yzt_aux.s_y
    s_yzt = yzt_aux.s_yzt
    n_y = dataobj.n_y
    ngrid = settings.alpha_ngrid
    bnd_recov = settings.bnd_recov
    hor_pred = round(settings.VMA_hor / 2) - 1  # largest you can go with a two-sided prediction

    # ----------------------------------------------------------------
    # Truth
    # ----------------------------------------------------------------
    if hasattr(dataobj, 'alpha'):
        alpha_true = dataobj.alpha
    else:
        alpha_true = []

    # ----------------------------------------------------------------
    # Upper Bound
    # ----------------------------------------------------------------
    alpha_UB = np.sqrt(Var_zt)

    # ----------------------------------------------------------------
    # Lower Bound
    # ----------------------------------------------------------------
    class alpha_plot():
        pass

    alpha_plot = alpha_plot()

    if bnd_recov == 0:  # Sharp lower bound

        alpha_LB_fun = lambda omega: np.sqrt((s_yzt(omega).conj().T @ np.linalg.solve(s_y(omega), s_yzt(omega))).real)[
            0, 0]

        omega_grid = np.linspace(0, 2 * math.pi, ngrid).T
        alpha_LB_vals = np.empty((ngrid, 1))
        for i in range(ngrid):
            alpha_LB_vals[i] = alpha_LB_fun(omega_grid[i])
        alpha_LB = np.max(alpha_LB_vals)
        alpha_plot.alpha_LB_vals = alpha_LB_vals
        alpha_plot.omega_grid = omega_grid

    elif bnd_recov == 1:  # Weaker lower bound (sharp under recoverability)
        Var_y = np.empty((((2 * hor_pred + 1) * n_y, (2 * hor_pred + 1) * n_y)))
        for i in range(2 * hor_pred + 1):
            for j in range(2 * hor_pred + 1):
                if i > j:
                    Var_y[i * n_y: (i + 1) * n_y, j * n_y: (j + 1) * n_y] = yzt_aux.Cov_y[abs(i - j)]
                else:
                    Var_y[i * n_y: (i + 1) * n_y, j * n_y: (j + 1) * n_y] = yzt_aux.Cov_y[abs(i - j)].T

        Cov_zy = np.zeros((1, (2 * hor_pred + 1) * n_y))

        for i in range(hor_pred + 1):
            Cov_zy[0, i * n_y: (i + 1) * n_y] = yzt_aux.Sigma_yzt[hor_pred + 1 - (i + 1), :]

        alpha_LB = np.sqrt(Cov_zy @ np.linalg.solve(Var_y, Cov_zy.T))[0, 0]

        alpha_plot.alpha_LB_vals = []
        alpha_plot.omega_grid = []
    return alpha_LB, alpha_UB, alpha_true, alpha_plot


def get_IS(yzt_aux, dataobj, settings):
    class bounds():
        pass

    bounds = bounds()
    # ----------------------------------------------------------------
    # Alpha
    # ----------------------------------------------------------------
    [bounds.alpha_LB, bounds.alpha_UB, bounds.alpha_true, bounds.alpha_plot] = alpha_IS(yzt_aux, dataobj, settings)

    class alpha():
        pass

    alpha = alpha()
    alpha.alpha_LB = bounds.alpha_LB
    alpha.alpha_UB = bounds.alpha_UB
    if hasattr(dataobj, 'alpha'):
        alpha.alpha_true = dataobj.alpha
    else:
        alpha.alpha_true = []
    # ----------------------------------------------------------------
    # R2
    # ----------------------------------------------------------------
    # invertibility
    if settings.CI_for_R2_inv == 1:
        [bounds.R2_inv_LB, bounds.R2_inv_UB, bounds.R2_inv_true] = R2_IS(yzt_aux, dataobj, settings, 1, alpha)
    else:
        bounds.R2_inv_LB = []
        bounds.R2_inv_UB = []

    # recoverability
    if settings.CI_for_R2_recov == 1:
        [bounds.R2_recov_LB, bounds.R2_recov_UB, bounds.R2_recov_true] = R2_IS(yzt_aux, dataobj, settings,
                                                                               round(settings.VMA_hor / 2) - 1,
                                                                               alpha)  # use exactly same bound as for two-sided alpha recoverability computation
    else:
        bounds.R2_recov_LB = []
        bounds.R2_recov_UB = []

    # ----------------------------------------------------------------
    # FVR
    # ----------------------------------------------------------------
    if settings.CI_for_FVR == 1:
        [bounds.FVR_LB, bounds.FVR_UB, bounds.FVR_true] = FVR_IS(yzt_aux, dataobj, settings, alpha)
    else:
        bounds.FVR_LB = []
        bounds.FVR_UB = []

    # ----------------------------------------------------------------
    # FVD
    # ----------------------------------------------------------------
    if settings.CI_for_FVD == 1:
        bounds.FVD_LB = FVD_IS(yzt_aux, dataobj, settings, alpha)
        bounds.FVD_UB = np.ones((np.size(bounds.FVD_LB, 0), dataobj.n_y))
    else:
        bounds.FVD_LB = []
        bounds.FVD_UB = []
    return bounds


def cross_sd_fun(omega, Sigma_yztilde):
    # preparations
    hor = np.size(Sigma_yztilde, 0)

    # derive lambda_yz_star(omega)
    cross_sd = 0
    for l in range(hor):
        cross_sd = cross_sd + Sigma_yztilde[l, :] * np.exp(complex(-1) * omega * l)
    cross_sd = cross_sd[np.newaxis, :]
    cross_sd = cross_sd.reshape((np.size(cross_sd, 1), np.size(cross_sd, 0)), order='F')
    return cross_sd


def sd_fun(omega, Theta_Wold):
    # Spectral density of moving average

    # preparations
    hor = np.size(Theta_Wold, 0)

    # derive Theta(omega)
    Theta_aux = 0
    for l in range(hor):
        Theta_aux = Theta_aux + np.squeeze(Theta_Wold[l, :, :]) * np.exp(complex(-1) * omega * l)

    # derive final result
    sd = Theta_aux @ Theta_aux.conj().T
    return sd


def kalman_filter(A, B, C, y):
    # Kalman filter for conditional variance calculations

    # number of observations and variables
    T = np.size(y, 0)
    n_s = np.size(A, 0)

    # initial conditions
    st = np.zeros((n_s, 1))
    Pt = np.zeros((n_s, n_s))

    # filtering
    for t in range(T):
        yt = y[t, :].T

        # prediction equations
        st_1 = A @ st
        Pt_1 = A @ Pt @ A.T + B @ B.T

        # innovations
        yt_1 = C @ st_1
        vt = yt - yt_1

        # updating equations
        Kt = Pt_1 @ C.T @ np.linalg.inv(C @ Pt_1 @ C.T)
        st = st_1 + Kt @ vt
        Pt = Pt_1 - Kt @ C @ Pt_1

    cond_var = Pt
    cond_var_1 = Pt_1

    return cond_var, cond_var_1


def get2ndmoments_VAR(VAR, dataobj, settings):
    # Compute autocovariances implied by reduced-form VAR

    # ----------------------------------------------------------------
    # Preparations
    # ----------------------------------------------------------------

    VAR_coeff = VAR.VAR_coeff
    Sigma_u = VAR.Sigma_u
    n_x = dataobj.n_x
    n_y = dataobj.n_y
    p = VAR.laglength
    VMA_hor = settings.VMA_hor

    # ----------------------------------------------------------------
    # Wold Representation
    # ----------------------------------------------------------------

    Theta_Wold = np.empty((VMA_hor, n_x, n_x))
    VAR_coeff = np.row_stack((VAR_coeff[:p * n_x, :], np.zeros((1 + VMA_hor * n_x - p * n_x, n_x))))

    for i in range(VMA_hor):
        if i == 0:
            Theta_Wold[i, :, :] = linalg.sqrtm(Sigma_u)
        else:
            Theta_Wold[i, :, :] = np.zeros((n_x, n_x))
            for j in range(i):
                Theta_Wold[i, :, :] = np.squeeze(Theta_Wold[i, :, :]) + VAR_coeff[j * n_x: (j + 1) * n_x,
                                                                        :].T @ np.squeeze(
                    Theta_Wold[i - j - 1, :, :])  # just compute Cov(x_t, eps_{t-h})

    Theta_Wold_y = Theta_Wold[:, :n_y, :]
    Theta_Wold_z = Theta_Wold[:, n_y, :]
    Theta_Wold_z = Theta_Wold_z[:, np.newaxis, :]
    Theta_Wold_zt = np.squeeze(Theta_Wold[0, n_y, :])

    # ----------------------------------------------------------------
    # Variances and Covariances
    # ----------------------------------------------------------------

    Cov_x = [None] * VMA_hor
    for i in range(VMA_hor):
        Cov_x[i] = np.zeros((n_x, n_x))
        for j in range(VMA_hor - i):
            Cov_x[i] = Cov_x[i] + np.squeeze(Theta_Wold[j, :, :]) @ np.squeeze(Theta_Wold[j + i, :, :]).T

    Cov_y = [None] * VMA_hor
    for i in range(VMA_hor):
        Cov_y[i] = Cov_x[i][:n_y, :n_y]

    Var_zt = Theta_Wold_zt.T @ Theta_Wold_zt
    Sigma_yzt = np.empty((VMA_hor, n_y))
    for i in range(VMA_hor):
        Sigma_yzt[i, :] = np.squeeze(Theta_Wold[i, :-1, :]) @ np.squeeze(Theta_Wold[0, -1, :])

    Sigma_y_big = np.empty((n_y * VMA_hor, n_y * VMA_hor))
    for i in range(VMA_hor):
        for j in range(VMA_hor):
            if i > j:
                Sigma_y_big[i * n_y: (i + 1) * n_y, j * n_y: (j + 1) * n_y] = Cov_y[abs(i - j)]
            else:
                Sigma_y_big[i * n_y: (i + 1) * n_y, j * n_y: (j + 1) * n_y] = Cov_y[abs(i - j)].T

    if settings.use_KF == 1:
        A_KF = np.zeros((n_x * p, n_x * p))
        for i in range(1, p + 1):
            A_KF[:n_x, (i - 1) * n_x: i * n_x] = VAR_coeff[(i - 1) * n_x: i * n_x, :].T

        A_KF[n_x: n_x * p, :n_x * (p - 1)] = np.eye(n_x * (p - 1))

        B_KF = np.zeros((n_x * p, n_x))
        B_KF[:n_x, :] = linalg.sqrtm(Sigma_u)

        C_KF = np.zeros((n_y, n_x * p))
        C_KF[:, :n_y] = np.eye(n_y)

        T_KF = 100
        y_KF = np.zeros((T_KF, n_y))
        _, cvar_states_1 = kalman_filter(A_KF, B_KF, C_KF, y_KF)
    else:
        A_KF = []
        B_KF = []
        C_KF = []
        cvar_states_1 = []

    # ----------------------------------------------------------------
    # Spectral Densities
    # ----------------------------------------------------------------

    s_y = lambda omega: sd_fun(omega, Theta_Wold_y)
    s_yzt = lambda omega: cross_sd_fun(omega, Sigma_yzt)

    # ----------------------------------------------------------------
    # Collect Results
    # ----------------------------------------------------------------
    class yzt_aux():
        pass

    yzt_aux = yzt_aux()
    yzt_aux.Theta_Wold = Theta_Wold
    yzt_aux.Theta_Wold_y = Theta_Wold_y
    yzt_aux.Theta_Wold_z = Theta_Wold_z
    yzt_aux.Theta_Wold_zt = Theta_Wold_zt
    yzt_aux.Cov_x = Cov_x
    yzt_aux.Cov_y = Cov_y
    yzt_aux.Var_zt = Var_zt
    yzt_aux.Sigma_yzt = Sigma_yzt
    yzt_aux.Sigma_u = Sigma_u
    yzt_aux.s_y = s_y
    yzt_aux.s_yzt = s_yzt
    yzt_aux.Sigma_y_big = Sigma_y_big
    yzt_aux.cvar_states_1 = cvar_states_1
    yzt_aux.A_KF = A_KF
    yzt_aux.B_KF = B_KF
    yzt_aux.C_KF = C_KF
    return yzt_aux


def bootstrapVAR(VAR, data, settings):
    # Homoskedastic recursive residual VAR bootstrap

    # ----------------------------------------------------------------
    # Estimate VAR
    # ----------------------------------------------------------------

    # preliminaries
    n_x = data.n_x
    data = data.data.x
    T = settings.T
    p = VAR.laglength
    n_boot = settings.n_boot

    # sample size
    T_VAR = T - p

    # estimate VAR
    X = lagmat(data, p, original="ex")
    X = X[p:, :]
    Y = data[p:, :]
    VAR_coeff = np.linalg.lstsq(np.c_[X, np.ones((len(X), 1))], Y, rcond=None)[0]
    VAR_res = Y - np.c_[X, np.ones((len(X), 1))] @ VAR_coeff

    # ----------------------------------------------------------------
    # Bootstrap
    # ----------------------------------------------------------------

    VAR_coeff_boot = np.empty((p * n_x + 1, n_x, n_boot))
    Sigma_u_boot = np.empty((n_x, n_x, n_boot))
    data_start = data[:p, :]

    for i in range(n_boot):
        if T >= 1000:
            if i % 10 == 0:
                print('I am at iteration ', i)
        u_boot = VAR_res[np.floor(np.size(VAR_res, 0) * np.random.rand(T_VAR, 1)).astype(int), :]  # reshuffle residuals

        # create new artificial data
        data_boot = np.empty((T, n_x))
        data_boot[:p, :] = data_start
        Xlag = X[0, :]

        for t in range(T_VAR):
            data_boot[p + t, :] = np.append(Xlag, 1.0) @ VAR_coeff + u_boot[t, :]
            Xlag = np.concatenate((data_boot[p + t, :], Xlag[:-n_x]), axis=0)

        # estimate VAR on artificial data
        X_boot = lagmat(data_boot, p, original='ex')
        X_boot = X_boot[p:, :]
        Y_boot = data_boot[p:, :]
        VAR_coeff_boot[:, :, i] = np.linalg.lstsq(np.c_[X_boot, np.ones((len(X_boot), 1))], Y_boot, rcond=None)[0]
        VAR_res_boot = Y_boot - np.c_[X_boot, np.ones((len(X_boot), 1))] @ VAR_coeff_boot[:, :, i]
        Sigma_u_boot[:, :, i] = (VAR_res_boot.T @ VAR_res_boot) / (T_VAR - n_x * p - 1)

    # collect results
    class VAR_boot():
        pass

    VAR_boot = VAR_boot()

    VAR_boot.VAR_coeff = VAR_coeff_boot
    VAR_boot.Sigma_u = Sigma_u_boot
    return VAR_boot


def test_invertibility(VAR):
    # Granger causality pre-test for invertibility

    # Dimensions
    [k, n_x] = VAR.VAR_coeff.shape
    p = int((k - 1) / n_x)

    # Var-cov matrix of vec(VAR_coeff)
    varcov = np.kron(VAR.Sigma_u, np.linalg.inv(VAR.X.T @ VAR.X))

    # Selection matrix picking out lags of z in each y equation
    the_eye = np.eye(n_x)
    sel = np.r_[np.tile(the_eye[:, n_x - 1], p), 0] == 1

    # Test in each y equation separately
    class wald_stat():
        pass
    wald_stat = wald_stat()
    class df():
        pass
    df = df()
    class pval():
        pass
    pval = pval()

    wald_stat.eqns = np.empty((1, n_x - 1))
    df.eqns = p * np.ones((1, n_x - 1))
    pval.eqns = np.empty((1, n_x - 1))
    for i in range(n_x - 1):
        the_coef = VAR.VAR_coeff[:, i]
        the_varcov = varcov[i * k:(i + 1) * k, i * k: (i + 1) * k]
        wald_stat.eqns[0, i] = the_coef[sel] @ np.linalg.lstsq(the_varcov[np.ix_(sel, sel)], the_coef[sel],
                                                                rcond=None)[0]
        pval.eqns[0, i] = 1 - chi2.cdf(wald_stat.eqns[0, i], df.eqns[0, i])

    # Overall test for all y equations jointly
    sel2 = np.row_stack((np.tile(sel.reshape(k, 1), (n_x - 1, 1)), np.zeros((k, 1)))) == 1
    sel2 = sel2.flatten()
    coef_vec = VAR.VAR_coeff.flatten(order='F')
    wald_stat.all = coef_vec[sel2].T @ ((np.linalg.lstsq(varcov[np.ix_(sel2, sel2)], coef_vec[sel2][:, np.newaxis], rcond=None))[0])
    df.all = p * (n_x - 1)
    pval.all = 1 - chi2.cdf(wald_stat.all, df.all)
    return wald_stat, df, pval


def selectlag_IC(data, maxlag, penalty):
    # Select VAR lag length using information criterion

    # preliminaries
    n_x = np.size(data, 1)

    # compute information criterion for each possible lag length
    IC = np.empty((maxlag + 1, 1))
    for i in range(1, maxlag + 1):
        # set lag length
        p = i

        # estimate VAR
        data_est = data[maxlag - p:, :]
        T = np.size(data_est, 0)
        T_VAR = T - p
        X = lagmat(data_est, p, original='ex')
        X = X[p:, :]
        Y = data_est[p:, :]
        X = np.column_stack((X, np.ones((len(X), 1))))
        VAR_coeff = (np.linalg.inv(X.T @ X)) @ (X.T @ Y)
        VAR_res = Y - X @ VAR_coeff
        #     Sigma_u    = (VAR_res'*VAR_res)/(T_VAR-n_x*p-1)
        Sigma_u = (VAR_res.T @ VAR_res) / T_VAR
        # compute information criterion
        T = np.size(data, 0) - maxlag
        #     IC(i)  = log(det(Sigma_u)) + i * n_x^2 * 2/T
        IC[i] = np.log(np.linalg.det(Sigma_u)) + i * n_x ** 2 * penalty(T)
    lag = np.where(IC == min(IC))
    return int(lag[0][0])


def estimateVAR(data, settings):
    # Least-squares VAR estimation
    # preliminaries
    n_x = np.size(data, 1)
    T = np.size(data, 0)

    # select lag length
    if settings.select_VAR_simlaglength == 1:
        p = selectlag_IC(data, settings.max_simlaglength, settings.penalty)
    else:
        p = settings.VAR_simlaglength

    T_VAR = T - p

    # estimate VAR
    X = lagmat(data, p, original='ex')
    X = X[p:, :]
    X = np.column_stack((X, np.ones((len(X), 1))))
    Y = data[p:, :]
    VAR_coeff = (np.linalg.inv(X.T @ X)) @ (X.T @ Y)
    VAR_res = Y - X @ VAR_coeff
    # Sigma_u    = (VAR_res'*VAR_res)/(T_VAR-n_x*p-1)
    Sigma_u = (VAR_res.T @ VAR_res) / T_VAR

    # collect results
    class VAR_sim():
        pass

    VAR_sim = VAR_sim()
    VAR_sim.T_VAR = T_VAR
    VAR_sim.VAR_coeff = VAR_coeff
    VAR_sim.Sigma_u = Sigma_u
    VAR_sim.VAR_res = VAR_res
    VAR_sim.X = X
    VAR_sim.laglength = p
    return VAR_sim