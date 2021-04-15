"""
SVMA-IV class that bounds identified shocks. Replicates Plagborg-Moller and Wolf (2020)
Author: Joao B. Duarte
Last update: 2021-04-14
This is an alpha version of the code. Any bugs or mistakes are my own. If you find them please report them to joao.duarte@novasbe.pt
"""
from python.functions.aux_functions import *
from tqdm import tqdm

def SVMAIV_estim(data_Y, data_z, settings_dic, ic_max=24, compute_R2_inv=True, compute_R2_recov=True, compute_FVR=True,
                 compute_FVD=True, horiz=np.arange(1, 25), ci_param=False, verbose=True, signif=0.1, n_boot=1000,
                 use_kalman=True, VMA_hor=100):
    """
    Inference routines for SVMA-IV analysis
    Point estimates and bootstrap confidence intervals for identification bounds

    Reference: Mikkel Plagborg-Moller & Christian K. Wolf (2020)
    "Instrumental Variable Identification of Dynamic Variance Decompositions"
    https://scholar.princeton.edu/mikkelpm/decomp_iv

    Inputs: see below

    Outputs:
    bounds    struct  Partial identification results:
                   - field "estim" contains estimates of bounds (bootstrap bias-corrected)
                   - field "ci" contains confidence intervals for identified intervals
                   - field "ci_param" contains Stoye (2009) confidence intervals for parameters (if option 'ci_param'=true)
    id_recov  struct  Point identification results under assumption of recoverability:
                   - field "estim" contains parameter estimates (bootstrap bias-corrected)
                   - field "ci" contains confidence intervals
    inv_test struct  Granger casuality pre-test of invertibility
                   - field "wald_stat" contains Wald statistics
                   - field "df" contains degrees of freedom
                   - field "pval" contains p-values
                   - subfield "all" is joint test in all y equations
                   - subfield "eqns" treats each y equation separately
    settings  struct  Settings (see below)
    VAR_OLS   struct  Estimated reduced-form VAR

    Parameter names in output:
    alpha     scale parameter
    R2_inv    degree of invertibility
    R2_recov  degree of recoverability
    FVR       Forecast Variance Ratio
    FVD       Forecast Variance Decomposition
    """

    # Inputs
    #
    # ip = inputParser
    #
    # # Required inputs
    # addRequired(ip, 'Y', @ isnumeric);  # T x n_y   endogenous variable data matrix
    # addRequired(ip, 'Z', @ isnumeric);  # T x 1     instrument data vector
    #
    # # Optional inputs: VAR specification
    # addParameter(ip, 'p',
    #              [], @ isnumeric);  # 1 x 1     VAR lag length, [] means use information criterion (default: [])
    # addParameter(ip, 'ic', 'aic', @ ischar);  # 1 x 1     information criterion, 'aic' or 'bic' (default: 'aic')
    # addParameter(ip, 'ic_max', 24, @ isnumeric);  # 1 x 1     max lag length for information criterion (default: 24)
    #
    # # Optional inputs: output requested
    # addParameter(ip, 'compute_R2_inv', true, @ islogical);  # bool      Compute degree of invertibility? (default: yes)
    # addParameter(ip, 'compute_R2_recov',
    #              true, @ islogical);  # bool      Compute degree of recoverability? (default: yes)
    # addParameter(ip, 'compute_FVR', true, @ islogical);  # bool      Compute Forecast Variance Ratio? (default: yes)
    # addParameter(ip, 'compute_FVD',
    #              true, @ islogical);  # bool      Compute Forecast Variance Decomposition? (default: yes)
    # addParameter(ip, 'horiz', 1: 24,
    #
    # @isnumeric
    #
    # );  # 1 x k     Horizons of FVR/FVD to report (default: 1:24)
    # addParameter(ip, 'ci_param',
    #              false, @ islogical);  # bool      Compute confidence intervals for parameters themselves (not identified sets)? (default: no)
    # addParameter(ip, 'verbose', true, @ islogical);  # bool      Print progress to screen? (default: yes)
    #
    # # Optional inputs: inference/bootstrap
    # addParameter(ip, 'signif', 0.1, @ isnumeric);  # 1 x 1     Significance level (default: 10#)
    # addParameter(ip, 'n_boot', 1000, @ isnumeric);  # 1 x 1     No. of bootstrap repetitions (default: 1000)
    # addParameter(ip, 'optim_opts', optimoptions('fmincon', 'Display', 'notify'), @ (x)
    # isobject(x) | isempty(x));  # obj  Numerical options for Stoye CI construction
    #
    # # Optional inputs: numerical settings
    # addParameter(ip, 'use_kalman',
    #              true, @ islogical);  # bool      Use Kalman filter for conditional variance calculations? (default: yes)
    # addParameter(ip, 'VMA_hor',
    #              100, @ isnumeric);  # 1 x 1     Truncation horizon for VMA representation of VAR (default: 100)
    #
    # parse(ip, Y, Z, varargin
    # {:});  # Parse inputs

    ''' Create settings structure '''

    class settings():
        pass

    settings = settings()
    settings.select_VAR_simlaglength = not 'p' in settings_dic  # Use information criterion?
    if settings.select_VAR_simlaglength == 0:
        settings.VAR_simlaglength = settings_dic['p']  # Lag length (if pre-set)
    settings.max_simlaglength = ic_max  # Max lag length for information criterion
    if settings_dic['ic'] == 'bic':
        settings.penalty = lambda T: np.log(T)  # BIC
    else:
        settings.penalty = lambda T: 2 / T  # AIC

    settings.CI_for_R2_inv = compute_R2_inv  # CI for R2_inv?
    settings.CI_for_R2_recov = compute_R2_recov  # CI for R2_rec?
    settings.CI_for_FVR = compute_FVR  # CI for FVR?
    settings.CI_for_FVD = compute_FVD  # CI for FVD?
    settings.CI_para = ci_param  # Stoye (2009) CI?

    settings.FVR_hor = horiz  # Horizons for FVR
    settings.FVD_hor = horiz  # Horizons for FVD

    settings.signif_level = signif  # Significance level
    if 'n_boot' in settings_dic:
        n_boot = settings_dic['n_boot']
    settings.n_boot = n_boot  # No. of bootstrap iterations

    settings.VMA_hor = VMA_hor  # Maximal horizon in Wold/structural VMA representation
    settings.alpha_ngrid = []  # No. of grid points for sharp lower bound on alpha (not used in estimation)
    settings.bnd_recov = 1  # Use weaker/practical lower bound on alpha
    settings.use_KF = use_kalman  # Use Kalman filter for computations?
    # settings.optimopts = ip.Results.optim_opts  # fmincon options for Stoye CI

    ''' Estimate reduced-form VAR '''
    # Data
    Y = data_Y
    Z = data_z

    class dataobj():
        pass

        class data():
            pass

    dataobj = dataobj()
    dataobj.data.x = np.column_stack((Y, Z))  # Data matrix
    dataobj.data.y = Y
    dataobj.data.z = Z

    # Model dimensions
    dataobj.n_x = np.size(dataobj.data.x, 1)
    dataobj.n_z = 1
    dataobj.n_y = dataobj.n_x - 1
    settings.T = np.size(dataobj.data.x, 0)

    # Estimate VAR
    if verbose == True:
        print('Estimating the VAR...')
    VAR_OLS = estimateVAR(dataobj.data.x, settings)
    if verbose == True:
        print('...done!')

    ''' Pre-test for invertibility '''

    class inv_test():
        pass
    inv_test = inv_test()

    inv_test.wald_stat, inv_test.df, inv_test.pval = test_invertibility(VAR_OLS)

    ''' Bootstrap VAR '''
    if verbose == True:
        print('Bootstrapping the VAR...')
    VAR_boot = bootstrapVAR(VAR_OLS, dataobj, settings)
    if verbose == True:
        print('...done!')

    ''' Bound point estimates '''
    if verbose == True:
        print('Getting the OLS point estimates of the identified sets...')
    yzt_aux = get2ndmoments_VAR(VAR_OLS, dataobj, settings)
    bounds_OLS = get_IS(yzt_aux, dataobj, settings)
    if verbose == True:
        print('...done!')

    ''' Compute bounds for each bootstrap iteration '''

    VAR_sim = VAR_OLS
    if 'alpha_plot' in bounds_OLS.__dict__:
        bounds_OLS.__dict__.pop('alpha_plot')
    fields = list(bounds_OLS.__dict__.keys())

    class bounds_boot():
        for j in range(len(fields)):
            shape = np.shape(getattr(bounds_OLS, fields[j]))
            if shape == () or shape == (0,):
                shape = (1, 1)
            size = tuple(list(shape) + [settings.n_boot])
            locals()[fields[j]] = np.empty(size)

    bounds_boot = bounds_boot()

    if verbose == 1:
        print('Mapping each bootstrap draw into objects of interest...')
    for i_boot in tqdm(range(settings.n_boot)):
        VAR_sim.VAR_coeff = VAR_boot.VAR_coeff[:, :, i_boot]
        VAR_sim.Sigma_u = VAR_boot.Sigma_u[:, :, i_boot]
        the_yzt_aux = get2ndmoments_VAR(VAR_sim, dataobj, settings)
        the_bounds = get_IS(the_yzt_aux, dataobj, settings)
        for j in range(len(fields)):
            if type(getattr(the_bounds, fields[j])) == np.float64:
                getattr(bounds_boot, fields[j])[:, :, i_boot] = np.array([getattr(the_bounds, fields[j])]).reshape(
                    (1, 1))
            elif type(getattr(the_bounds, fields[j])) == np.ndarray:
                getattr(bounds_boot, fields[j])[:, :, i_boot] = getattr(the_bounds, fields[j])  # Store bounds

    if verbose == 1:
        print('...done!')

    ''' Construct CIs '''
    if verbose == 1:
        print('Constructing the confidence intervals...')

    class CI():
        pass

    CI = CI()
    CI.bounds_CI_IS, CI.bounds_CI_para = CI_fun(bounds_boot, bounds_OLS, settings)
    if verbose == 1:
        print('...done!')

    ''' Collect results '''
    bounds = {'estim': {'lower': {}, 'upper': {}}, 'ci': {'lower': {}, 'upper': {}}}
    id_recov = {'estim': {}, 'ci': {'lower': {}, 'upper': {}}}
    params = ['alpha', 'R2_inv', 'R2_recov', 'FVR', 'FVD']

    for ip in range(len(params)):
        # Parameter name
        the_param = params[ip]
        the_param_LB = the_param + '_LB'
        the_param_UB = the_param + '_UB'

        # Point estimates of bounds
        bounds['estim']['lower'][the_param] = getattr(CI.bounds_CI_IS.OLS_biascorr, the_param_LB)
        bounds['estim']['upper'][the_param] = getattr(CI.bounds_CI_IS.OLS_biascorr, the_param_UB)

        # Confidence intervals for identified set
        bounds['ci']['lower'][the_param] = getattr(CI.bounds_CI_IS.lower, the_param_LB)
        bounds['ci']['upper'][the_param] = getattr(CI.bounds_CI_IS.upper, the_param_UB)

        # Estimates/CIs under recoverability
        if the_param == 'alpha':
            id_recov['estim'][the_param] = bounds['estim']['lower'][the_param]
            id_recov['ci']['lower'][the_param] = getattr(CI.bounds_CI_IS.lower, the_param_LB)
            id_recov['ci']['upper'][the_param] = getattr(CI.bounds_CI_IS.upper, the_param_LB)
        elif the_param == 'FVD':
            # Do nothing FVD is not point-identified even under recoverability
            continue
        else:
            id_recov['estim'][the_param] = bounds['estim']['upper'][the_param]
            id_recov['ci']['lower'][the_param] = getattr(CI.bounds_CI_IS.lower, the_param_UB)
            id_recov['ci']['upper'][the_param] = getattr(CI.bounds_CI_IS.upper, the_param_UB)

    return bounds, id_recov, inv_test, settings, VAR_OLS
