#%% Importing libraries
import numpy as np
import pandas as pd
import timeit
from scipy.stats import norm
from numba import njit
from tqdm import tqdm
import matplotlib.pyplot as plt
# import pickle

#%%

dashed_line_kws = {"color": "black", "linestyle": "--", "alpha": 0.6}

def discretizeNormal(N, mu, rho, sigma, m):
    '''
    Discretization of AR(1) process according following Tauchen

    :param N: number of points
    :param mu: unconditional mean
    :param rho: persistence
    :param sigma: volatility of innovations
    :param m: boundaies in terms of standard deviation, i.e. make a grid for -m*sigma to m*sigma
    :return: a tuple: (grid, transition probabilities matrix, ij index = P(i -> j)
    '''
    Zmax  = mu + m*np.sqrt(sigma*sigma/(1-rho*rho))
    Zmin  = mu - m*np.sqrt(sigma*sigma/(1-rho*rho))
    Z = np.linspace(Zmin, Zmax, num=N)
    zstep = Z[1] - Z[0]

    a = (1-rho)*mu
    Z = Z + a/(1-rho)

    # Filling the transition matrix
    Zprob = np.zeros((N, N))
    for j in range(N):
        Zprob[j,0] = norm.cdf((Z[0]-a-rho*Z[j]+zstep/2)/sigma)
        Zprob[j,-1] = 1-norm.cdf((Z[-1]-a-rho*Z[j]-zstep/2)/sigma)
        for k in range(1,N-1):
            Zprob[j,k] = norm.cdf((Z[k]-a-rho*Z[j]+zstep/2)/sigma) - norm.cdf((Z[k]-a-rho*Z[j]-zstep/2)/sigma)

    return Z, Zprob


def discretizeNormalStVol(N, mu, rho, sigma0, b, mmin, mmax):
    '''State varying volatility. For b = 0, volatility = sigma0 = const'''
    Zmax  = mu + mmax*np.sqrt(sigma0*sigma0/(1-rho*rho))
    Zmin  = mu - mmin*np.sqrt(sigma0*sigma0/(1-rho*rho))
    Z = np.linspace(Zmin, Zmax, num=N)
    zstep = Z[1] - Z[0]

    sigmaZ = sigma0*(1+1)/(1 + np.exp(b*Z))

    # For simpler code, calculate bounds of state regions:
    Zlb = np.zeros(N)
    Zub = np.zeros(N)
    Zlb[0], Zub[-1] = -1.0e8, 1.0e8
    Zlb[1:] = Z[1:] - zstep/2
    Zub[:-1] = Z[:-1] + zstep/2

    arg_ub = np.tile(Zub.reshape((1, -1)), (N, 1))
    arg_ub += -(1-rho)*mu - rho*np.tile(Z.reshape((-1, 1)), (1, N))
    arg_ub *= 1/np.tile(sigmaZ.reshape((-1, 1)), (1, N))

    arg_lb = np.tile(Zlb.reshape((1, -1)), (N, 1))
    arg_lb += -(1-rho)*mu - rho*np.tile(Z.reshape((-1, 1)), (1, N))
    arg_lb *= 1/np.tile(sigmaZ.reshape((-1, 1)), (1, N))

    Pz = norm.cdf(arg_ub) - norm.cdf(arg_lb)

    return Z, Pz


# @njit("float64[:,:](float64[:], float64[:], float64, float64, float64)")
def calcEntryProdDistribution(q, ygrid, muy, rhoy, sigmay):
    num_firms = q.shape[0]
    logq = np.log(q)
    ystep = ygrid[1] - ygrid[0]
    Ny = ygrid.shape[0]

    ylb = np.zeros(Ny)
    yub = np.zeros(Ny)
    ylb[0], yub[-1] = -1.0e8, 1.0e8
    ylb[1:] = ygrid[1:] - ystep/2
    yub[:-1] = ygrid[:-1] + ystep/2

    Ny = ygrid.shape[0]
    arg_ub = np.tile(yub.reshape((1, -1)), (num_firms, 1))
    arg_ub += -(1-rhoy)*muy - rhoy*np.tile(logq.reshape((-1, 1)), (1, Ny))
    arg_ub *= 1/sigmay

    arg_lb = np.tile(ylb.reshape((1, -1)), (num_firms, 1))
    arg_lb += -(1-rhoy)*muy - rhoy*np.tile(logq.reshape((-1, 1)), (1, Ny))
    arg_lb *= 1/sigmay

    Pprod = norm.cdf(arg_ub) - norm.cdf(arg_lb)

    return Pprod


def solveForAgentUtility(xgrid, Px, delta, gamma, psi):
    isUnitPsi = (psi == 1.0)
    Cgrid = np.exp(xgrid)

    if isUnitPsi:
        Uprev = np.power(Cgrid, 1-delta)
    else:
        Uprev = np.power((1-delta)*np.power(Cgrid, 1-1/psi), 1/(1-1/psi))

    dist, tol, it, max_it = 1e6, 1e-8, 0, 10000
    while it < max_it and dist > tol:
        # Can calculate certainity equivalent for U/C separately from consumption growth 
        # since shocks to consumption and consumption growth are uncorrelated
        Uceq = np.power(Px.dot(np.power(Uprev.reshape((-1, 1)), 1-gamma)), 1/(1-gamma))

        # Updating differs depending on whether psi is equal to unity or not
        if isUnitPsi:
            Unew = np.power(Cgrid.reshape((-1, 1)), 1 - delta)*np.power(Uceq, delta)
        else:
            Unew = (1-delta)*np.exp((1-1/psi)*xgrid.reshape((-1, 1)))
            Unew += delta*np.power(Uceq, 1-1/psi)
            Unew = np.power(Unew, 1/(1-1/psi))

        # Calculating sup-distance between next period and current value
        # and updating the next period value
        dist = np.amax(np.abs(Unew - Uprev))
        Uprev = Unew
        it += 1

    U = Uprev.flatten()
    assert it < max_it
    print(f"Agent's problem converged in {it} iterations\n")

    # Constructing an SDF based on consumption and utility
    Mtp1 = np.exp(-(1/psi)*xgrid)*np.power(U, -(gamma-1/psi))
    Mtp1 = Mtp1.reshape((1, -1))

    Uceq = np.power(Px.dot(np.power(U.reshape((-1, 1)), 1-gamma)), 1/(1-gamma))
    Uceq = Uceq.flatten()
    Mt = np.exp(-(1/psi)*xgrid)*np.power(Uceq, -(gamma-1/psi))
    Mt = Mt.reshape((-1, 1))

    M = delta * Mtp1/Mt

    # Calculating certain moments related to the SDF
    # 1. Risk free rates by state
    Rf = np.array(np.diag(1/Px.dot(M.T) - 1))  # for each state

    # 2. Maximum achievable Sharpe Ratio
    sdM = np.diag(Px.dot(np.power(M.T, 2))) - np.power(np.diag(Px.dot(M.T)),2)
    sdM = np.sqrt(sdM)
    EM = np.diag(Px.dot(M.T))
    maxSR = sdM/EM  # for each state

    return U, M, Rf, maxSR
    

def stationaryDistMarkov(P):
    '''
    Calculates the vector corresponding to a stationary distribution
    for a discrete state Markov process with transition matrix P where
    P_{ij} = P(i -> j) by calculating the eignevectors and finding the
    one with entries of the same sign (either all positive or all
    negative).

    :param P: transition matrix P_{ij} = P(i -> j)
    :return: vector of stationary distribution
    '''

    w, v = np.linalg.eig(P.T)
    same_sign_index = np.logical_or(
        np.sum(v > 0, axis=0) == P.shape[0],
        np.sum(v < 0, axis=0) == P.shape[0]
    )
    pi = v[:, np.argmax(same_sign_index)]

    # Switching sign if necessary
    if pi[0] < 0:
        pi = -pi

    # Normalizing by the sum
    return pi/np.sum(pi)


@njit('float64[:,:,:](float64[:,:,:,:], float64[:,:,:])')
def inner_loop(F, EMV):
    '''Inner loop for value function iteration optimized using numba'''
    Nk, Nx, Ny = EMV.shape
    value = np.zeros(EMV.shape)

    for ix in range(Nx):
        for iy in range(Ny):
            # For given productivity (x, y), k' is an increasing function of k
            # and, hence, we can start searching for optimum k' from prev. max
            # that we store in start_ikp and reuse in the next iteration
            start_ikp = 0
            for ik in range(Nk):
                current_max = -1e10
                for ikp in range(start_ikp, Nk):
                    new_value = F[ik, ikp, ix, iy] + EMV[ikp, ix, iy]
                    if new_value > current_max:
                        current_max = new_value
                        start_ikp = ikp
                    else:
                        break

                value[ik, ix, iy] = current_max

    return value


# @njit('float64[:,:,:](float64[:,:,:], float64[:,:,:,:], float[:,:,:,:])')
def bellmanOperatorEquity(V, H, F):
    # Calculating expected continuation value
    EMV = np.tensordot(V, H, axes=([1, 2], [1, 3]))

    # Optimal investment conditional on not exiting
    tomax = F + EMV[None, :, :, :]
    Vmax = np.max(tomax, axis = 1)

    # If Value < 0 => exit
    return np.maximum(Vmax, 0.0)


def stationaryDistMarkov(P):
    '''
    Calculates the vector corresponding to a stationary distribution
    for a discrete state Markov process with transition matrix P where
    P_{ij} = P(i -> j) by calculating the eignevectors and finding the
    one with entries of the same sign (either all positive or all
    negative).

    :param P: transition matrix P_{ij} = P(i -> j)
    :return: vector of stationary distribution
    '''

    w, v = np.linalg.eig(P.T)
    same_sign_index = np.logical_or(
        np.sum(v > 0, axis=0) == P.shape[0],
        np.sum(v < 0, axis=0) == P.shape[0]
    )
    pi = v[:, np.argmax(same_sign_index)]

    # Switching sign if necessary
    if pi[0] < 0:
        pi = -pi

    # Normalizing by the sum
    return pi/np.sum(pi)


@njit("int64[:](int64, int64)")
def unravel_index(ind, ncol):
    '''Convert 1d index <ind> of a flattened array to a 2d index
    given the number of columns <ncol>'''
    row = ind // ncol
    col = ind - row*ncol
    return np.array([row, col])


def calcPoliciesRiskyDebtInner(
    EMV, b_proceeds, F,
    d_pre_debt, Rf, bgrid, 
    b, k, cd, xid, tauc, taue, rcmax, theta, delta):
    
    Nk, Nb, Nx, Ny = EMV.shape

    # Calculating dividend flow including proceeds/payments for debt
    d = d_pre_debt - b
    d = d + b*tauc*Rf[None, :, None]
    d = d[:, None, :, :] + b_proceeds

    # Calculating issuance costs
    iss_costs = (d < 0)*(cd*d - xid)
    pay_tax = (d > 0)*(taue*d)
    F = d + iss_costs - pay_tax

    # Maximization objective: Flow + Continuation value
    tomax = F + EMV

    # # Maximizing flow + continuation value and reshaping indices
    max_ind_flat = np.argmax(tomax.reshape((Nk*Nb, Nx, Ny)), axis=0)
    max_ind_row = max_ind_flat // Nb
    max_ind_col = max_ind_flat - max_ind_row*Nb
    ix_mat = np.tile(np.array(range(Nx)).reshape((-1, 1)), (1, Ny))
    iy_mat = np.tile(np.array(range(Ny)).reshape((1, -1)), (Nx, 1))

    # Subsetting the matrices
    ikp_opt = max_ind_row
    ibp_opt = max_ind_col
    d_opt = d[max_ind_row, max_ind_col, ix_mat, iy_mat]
    exit_opt = tomax[max_ind_row, max_ind_col, ix_mat, iy_mat] < 0.0
    F_opt = F[max_ind_row, max_ind_col, ix_mat, iy_mat]
    rate_opt = bgrid[max_ind_col]/b_proceeds[max_ind_row, max_ind_col, ix_mat, iy_mat]
    Fd_opt = np.where(
        exit_opt, np.min(np.array([theta*(1-delta)*k, rcmax*b])),
        b - b_proceeds[max_ind_row, max_ind_col, ix_mat, iy_mat]
    )

    return d_opt, ibp_opt, ikp_opt, F_opt, Fd_opt, rate_opt, exit_opt


@njit("f8[:,:,:](i8[:], i8[:,:], i8[:,:], f8[:,:,:], i8[:,:,:], b1[:,:,:], f8[:,:,:], f8[:,:], f8[:,:])")
def simulatePoliciesEquity(ix_path, iy_paths, ik_paths, V, ikp_opt, exit_opt, F_opt, Py, udraws):

    num_periods, total_entrants = iy_paths.shape
    Ny, _ = Py.shape

    exit_paths = np.zeros((num_periods, total_entrants))
    F_paths = np.zeros((num_periods, total_entrants))
    V_paths = np.zeros((num_periods, total_entrants))

    for t in range(num_periods - 1):
        ix = ix_path[t]
        for ifirm in range(total_entrants):
            iy = iy_paths[t, ifirm]
            ik = ik_paths[t, ifirm]

            # 1. Does this firm operate?
            if iy > -99:

                # Will the firm exit this period
                if exit_opt[ik, ix, iy]:
                    exit_paths[t, ifirm] = True
                    F_paths[t, ifirm] = 0.0
                    iy_paths[t+1, ifirm] = -99
                    V_paths[t, ifirm] = 0.0
                else:
                    # If doesn't exit, calculate next period capital ...
                    ik_paths[t+1, ifirm] = ikp_opt[ik, ix, iy]

                    # ... simulate next period productivity (by inverting the uniform rv with discrete CDF)
                    iy_paths[t+1, ifirm] = np.searchsorted(np.cumsum(Py[iy, :]), udraws[t, ifirm])

                    # ... flows to lenders
                    F_paths[t, ifirm] = F_opt[ik, ix, iy]
                    
                    # ... and value of equity
                    V_paths[t, ifirm] = V[ik, ix, iy]

    out = np.zeros((5, num_periods, total_entrants))
    out[0, :, :] = ik_paths
    out[1, :, :] = iy_paths
    out[2, :, :] = F_paths
    out[3, :, :] = exit_paths
    out[4, :, :] = V_paths

    return out


@njit("f8[:,:,:](i8[:], i8[:,:], i8[:,:], i8[:,:], f8[:,:,:,:], i8[:,:,:,:], i8[:,:,:,:], b1[:,:,:,:],  f8[:,:,:,:], f8[:,:,:,:], f8[:,:,:,:], f8[:,:], f8[:,:])")
def simulatePoliciesDebt(ix_path, iy_entry, ik_entry, ib_entry, V, ikp_opt, ibp_opt, exit_opt, F_opt, Fd_opt, rate_opt, Py, udraws):

    num_periods, total_entrants = iy_entry.shape
    Ny, _ = Py.shape

    iy_paths = iy_entry.copy()
    ik_paths = ik_entry.copy()
    ib_paths = ib_entry.copy()
    exit_paths = np.zeros((num_periods, total_entrants))
    F_paths = np.zeros((num_periods, total_entrants))
    Fd_paths = np.zeros((num_periods, total_entrants))
    rate_paths = np.zeros((num_periods, total_entrants))
    V_paths = np.zeros((num_periods, total_entrants))

    for t in range(num_periods - 1):
        ix = ix_path[t]
        for ifirm in range(total_entrants):
            iy = iy_paths[t, ifirm]
            ik = ik_paths[t, ifirm]
            ib = ib_paths[t, ifirm]

            # 1. Does this firm operate
            if iy > -99:
                # Will the firm exit this period
                if exit_opt[ik, ib, ix, iy]:
                    exit_paths[t, ifirm] = True
                    Fd_paths[t, ifirm] = Fd_opt[ik, ib, ix, iy] # payment in default
                    iy_paths[t+1, ifirm] = -99
                    V_paths[t, ifirm] = 0.0
                else:
                    # If doesn't exit, calculate next period capital ...
                    ik_paths[t+1, ifirm] = ikp_opt[ik, ib, ix, iy]
                    ib_paths[t+1, ifirm] = ibp_opt[ik, ib, ix, iy]

                    # ... simulate next period productivity (by inverting the discrete CDF at a uniform rv)
                    iy_paths[t+1, ifirm] = np.searchsorted(np.cumsum(Py[iy, :]), udraws[t, ifirm])

                    # ... flows to equity, lenders and interest rate
                    F_paths[t, ifirm] = F_opt[ik, ib, ix, iy]
                    Fd_paths[t, ifirm] = Fd_opt[ik, ib, ix, iy]
                    rate_paths[t, ifirm] = rate_opt[ik, ib, ix, iy]

                    # ... and value
                    V_paths[t, ifirm] = V[ik, ib, ix, iy]

    out = np.zeros((8, num_periods, total_entrants))
    out[0, :, :] = ik_paths
    out[1, :, :] = ib_paths
    out[2, :, :] = iy_paths
    out[3, :, :] = F_paths
    out[4, :, :] = Fd_paths
    out[5, :, :] = rate_paths
    out[6, :, :] = exit_paths
    out[7, :, :] = V_paths

    return out


def generateVariablesEquity(sim_equity_df):
    # 1. Median firm size for each time period + ind below/above median
    k_q33 = sim_equity_df.groupby("t")["k"].apply(lambda x: np.quantile(x, 0.33)).rename("k_q33").reset_index()
    k_q66 = sim_equity_df.groupby("t")["k"].apply(lambda x: np.quantile(x, 0.66)).rename("k_q66").reset_index()
    sim_equity_df = pd.merge(sim_equity_df, k_q33, on="t", how="left")
    sim_equity_df = pd.merge(sim_equity_df, k_q66, on="t", how="left")

    sim_equity_df["size"] = np.where(
        sim_equity_df["k"] >= sim_equity_df["k_q66"], "large", 
        np.where(
            sim_equity_df["k"] >= sim_equity_df["k_q33"], "med", "small" 
        ))

    # 2. assigning three types of aggregate state
    sim_equity_df["agg_state"] = np.where(sim_equity_df["ix"] >= 13, "high", np.where(sim_equity_df["ix"] >= 8, "med", "low"))

    # 3. Normalizing flows by capital
    sim_equity_df["F_to_k"] = sim_equity_df["F"]/sim_equity_df["k"]
    sim_equity_df["V_to_k"] = sim_equity_df["V"]/sim_equity_df["k"]

    return sim_equity_df


def generateVariablesDebt(sim_debt_df, Rf):
    # 1. Median firm size for each time period + ind below/above median
    k_q33 = sim_debt_df.groupby("t")["k"].apply(lambda x: np.quantile(x, 0.33)).rename("k_q33").reset_index()
    k_q66 = sim_debt_df.groupby("t")["k"].apply(lambda x: np.quantile(x, 0.66)).rename("k_q66").reset_index()
    sim_debt_df = pd.merge(sim_debt_df, k_q33, on="t", how="left")
    sim_debt_df = pd.merge(sim_debt_df, k_q66, on="t", how="left")

    sim_debt_df["size"] = np.where(
        sim_debt_df["k"] >= sim_debt_df["k_q66"], "large", 
        np.where(
            sim_debt_df["k"] >= sim_debt_df["k_q33"], "med", "small" 
        ))

    # 2. assigning three types of aggregate state
    sim_debt_df["agg_state"] = np.where(sim_debt_df["ix"] >= 13, "high", np.where(sim_debt_df["ix"] >= 8, "med", "low"))

    # 3. Total flows
    sim_debt_df["Ft"] = sim_debt_df["F"] + sim_debt_df["Fd"]

    # 4. Normalizing flows by capital
    sim_debt_df["F_to_k"] = sim_debt_df["F"]/sim_debt_df["k"]
    sim_debt_df["Fd_to_k"] = sim_debt_df["Fd"]/sim_debt_df["k"]
    sim_debt_df["Ft_to_k"] = sim_debt_df["Ft"]/sim_debt_df["k"]
    sim_debt_df["V_to_k"] = sim_debt_df["V"]/sim_debt_df["k"]

    # 5. Spread over riskless rate
    rf_df = pd.DataFrame({"ix": range(Rf.shape[0]), "rf": Rf})
    sim_debt_df = pd.merge(sim_debt_df, rf_df, left_on="ix", right_on="ix")
    sim_debt_df["spread"] = sim_debt_df["rate"] - sim_debt_df["rf"]

    return sim_debt_df


#%%

class financingModel():
    def __init__(self, params) -> None:

        # Production parameters
        self.delta = params['delta']
        self.cf = params['cf']
        self.cfprop = params['cfprop']
        self.theta = params['theta']
        self.phi, self.phim, self.phif = params['phi'], params['phim'], params["phif"]
        self.alpha = params['alpha']
        self.betax = params['betax']
        self.cd = params['cd']
        self.xid = params['xid']
        self.tauc = params['tauc']
        self.taue = params['taue']
        self.rcmax = params['rcmax']

        self.Nx, self.Ny = params['Nx'], params['Ny']
        self.mux, self.rhox = params['mux'], params['rhox']
        self.muy, self.rhoy = params['muy'], params['rhoy']
        self.sigmax, self.sigmay = params['sigmax'], params['sigmay']

        # Discretizing using Tauchen method
        self.b = params['b']
        self.xgrid, self.Px = discretizeNormalStVol(self.Nx, self.mux, self.rhox, self.sigmax, self.b, 4, 4)  # Aggregate risk
        self.ygrid, self.Py = discretizeNormal(self.Ny, self.muy, self.rhoy, self.sigmay, 3)  # Idiosyncratic risk

        # Solving for Epstein-Zin agents utility to form the SDF
        self.beta, self.gamma, self.psi = params['beta'], params['gamma'], params['psi']
        self.U, self.M, self.Rf, self.maxSR = solveForAgentUtility(self.xgrid, self.Px, self.beta, self.gamma, self.psi)
        
        # Matrices used for VFI later on
        self.xmat = np.repeat(self.xgrid[:, np.newaxis], self.Ny, axis=1)
        self.ymat = np.repeat(self.ygrid[np.newaxis, :], self.Nx, axis=0)

        # Forming transitional matrix H(x',y'|x,y) = M(x'|x)P(x',y'|x,y)
        self.Hx = np.repeat((self.M*self.Px)[:, :, np.newaxis], self.Ny, axis=2)
        self.Hx = np.repeat(self.Hx[:, :, :, np.newaxis], self.Ny, axis=3)
        self.Hy = np.repeat(self.Py[np.newaxis, :, :], self.Nx, axis=0)
        self.Hy = np.repeat(self.Hy[np.newaxis, :, :, :], self.Nx, axis=0)
        self.H = self.Hx*self.Hy

        self.EM = np.sum(self.H, axis=(1, 3))


    ########################################################################
    # Functions to solve the unconstrained equity only model
    def solveForTerminalValue(self, max_iter=1000, tolerance=1e-8, kgrid=None, Vprev=None):
        '''Solves for infinite horizon terminal value with an inner loop optimized with numba'''

        # Getting parameters
        delta, cf, cfprop, theta = self.delta, self.cf, self.cfprop, self.theta
        alpha, beta, phi, phim = self.alpha, self.beta, self.phi, self.phim
        betax = self.betax

        if kgrid is None:
            kgrid = np.arange(0.01, 3.0, 0.01)

        Nk = kgrid.shape[0]

        if Vprev is None:
            # Starting conditions (need to expand so it is the same across shocks)
            Vprev = 1/(1 - beta) * (np.power(kgrid, alpha) - cf - cfprop*kgrid - delta*kgrid - 0.5*phi*delta*delta*kgrid)
            Vprev = np.repeat(Vprev[:, np.newaxis], self.Nx, axis=1)
            Vprev = np.repeat(Vprev[:, :, np.newaxis], self.Ny, axis=2)

        # Calculating F by utilizing broadcasting (performs much much faster)
        # Allow for asymmetric adjustment costs
        I = kgrid[None, :] - ((1-delta)*kgrid)[:, None]
        InvCost = I + 0.5*np.where(I < 0, phim, phi)*np.power(I, 2)/kgrid[:, None]

        F = np.exp(betax*self.xgrid[:, None] + self.ygrid[None, :])
        F = F[None, :, :]*np.power(kgrid, alpha)[:, None, None]
        F = F[:, None, :, :] - InvCost[:, :, None, None]
        F += -cf - cfprop*kgrid[:, None, None, None]

        # Doing an iteration step
        dist, it = 1e6, 0
        print('Iterating...')
        start = timeit.default_timer()
        while it < max_iter and dist > tolerance:
            if it % 50 == 0:
                print(f'iter={it}, dist={dist}')

            # Calculating continuation value with a tensor product over (x', y') dimensions
            contValue = np.tensordot(Vprev, self.H, axes=([1, 2], [1, 3]))

            # Finding the maximum and updating value
            Vnew = inner_loop(F, contValue)
            dist = np.amax(np.abs(Vnew - Vprev))
            Vprev = Vnew

            it += 1

        stop = timeit.default_timer()
        execution_time = stop - start
        print(f"Firm's problem converged in {execution_time} seconds, {it} iterations\n")

        # Final value function on a grid
        self.TV = Vprev
        self.kgrid = kgrid


    ########################################################################
    # Functions to solve the CONSTRAINED equity only model

    def calcFlowsEquityIssuance(self):
        # Getting parameters
        delta, cf, cfprop, theta = self.delta, self.cf, self.cfprop, self.theta
        alpha, beta, phi, phim, phif = self.alpha, self.beta, self.phi, self.phim, self.phif
        cd, xid = self.cd, self.xid
        taue = self.taue
        betax = self.betax

        kgrid = self.kgrid
        Nk = kgrid.shape[0]

        Vprev = self.TV # Starting value -- solution for fully unconstrained problem

        # Calculating F by utilizing broadcasting (performs much much faster)
        I = kgrid[None, :] - ((1-delta)*kgrid)[:, None]
        phi_effective = np.where(kgrid[None, :] >= kgrid[:, None], phi, phim)
        inv_fixed_cost = np.where(kgrid[None, :] == kgrid[:, None], 0.0, phif*kgrid[:, None])
        inv_costs = inv_fixed_cost + I + 0.5*phi_effective*np.power(I/kgrid[:, None], 2)*kgrid[:, None]

        # Calculating dividends
        d = np.exp(betax*self.xgrid[:, None] + self.ygrid[None, :])
        d = d[None, :, :]*np.power(kgrid, alpha)[:, None, None]
        d = d[:, None, :, :] - inv_costs[:, :, None, None]
        d += -cf - cfprop*kgrid[:, None, None, None]

        # If d < 0, calculating equity issuance costs
        iss_costs = np.where(d < 0, cd*d - xid, 0.0)
        pay_tax = np.where(d > 0, taue*d, 0.0)
        F = d + iss_costs - pay_tax

        # Writing into the object
        self.d = d
        self.F = F


    def solveEquityIssuance(self, Vprev=None, tolerance=1e-8, maxiter=5000):
        print("\nSolving costly equity issuance problem")
        if Vprev is None:
            Vprev = self.TV

        H = self.H
        F = self.F

        dist, it = 1e6, 0
        print('Iterating...')
        start = timeit.default_timer()
        while it < maxiter and dist > tolerance:
            if it % 50 == 0:
                print(f'iter={it}, dist={dist}')

            Vnew = bellmanOperatorEquity(Vprev, H, F)
            dist = np.max(np.abs(Vnew - Vprev))
            Vprev = Vnew
            it += 1

        stop = timeit.default_timer()
        execution_time = stop - start
        print(f"Firm's problem converged in {execution_time} seconds, {it} iterations, dist = {dist}\n")

        self.Vequity = Vprev


    def calcPoliciesEquityIssuance(self):
        V = self.Vequity
        H = self.H
        F = self.F
        d = self.d
        kgrid = self.kgrid
        cd, xid = self.cd, self.xid

        EMV = np.tensordot(V, H, axes=([1, 2], [1, 3]))

        # Optimal investment conditional on not exiting
        tomax = F + EMV[None, :, :, :]
        ikp_opt = np.argmax(tomax, axis=1)
        Vmax = np.max(tomax, axis=1)
        exit_opt = Vmax < 0.0

        # Calculating dividends
        Nk, Nx, Ny = V.shape
        d_opt = np.empty_like(V)
        kp_opt = np.empty_like(V)
        for ik in range(Nk):
            for ix in range(Nx):
                for iy in range(Ny):
                    d_opt[ik, ix, iy] = d[ik, ikp_opt[ik, ix, iy], ix, iy]
                    kp_opt[ik, ix, iy] = kgrid[ikp_opt[ik, ix, iy]]

        # Calculating flows to lenders (div - iss. costs - payout tax)
        iss_costs = np.where(d_opt < 0, cd*d_opt - xid, 0.0)
        pay_tax = np.where(d_opt > 0, self.taue*d_opt, 0.0)
        F_opt = d_opt + iss_costs - pay_tax

        # Writing into the object
        self.F_opt = F_opt
        self.d_opt = d_opt
        self.kp_opt = kp_opt
        self.ikp_opt = ikp_opt
        self.exit_opt = exit_opt


    ########################################################################
    # Functions to load debt model solution

    def writeDebtSolution(self, bgrid, Vdebt, policy_debt):
        self.bgrid = bgrid
        self.Vdebt = Vdebt
        self.ikp_opt_debt = np.array(policy_debt[0, :, :, :, :], dtype=int)
        self.ibp_opt_debt = np.array(policy_debt[1, :, :, :, :], dtype=int)
        self.d_opt_debt = policy_debt[2, :, :, :, :]
        self.F_opt_debt = policy_debt[3, :, :, :, :]
        self.Fd_opt_debt = policy_debt[4, :, :, :, :]
        self.rate_opt_debt = policy_debt[5, :, :, :, :]
        self.exit_opt_debt = np.array(policy_debt[6, :, :, :, :], dtype=bool)

    ########################################################################
    # Functions for simulations

    def calcEntryValue(self, q, ik_private, ix, debt=False, return_add=False):
        Pprod = calcEntryProdDistribution(q, self.ygrid, self.muy, self.rhoy, self.sigmay)

        kgrid = self.kgrid
        M, Px = self.M, self.Px
        phi, delta, cd, xid, taue = self.phi, self.delta, self.cd, self.xid, self.taue

        # Calculating optimal capital level subject to equity issuance costs
        k_private = kgrid[ik_private]
        d = -(kgrid - (1 - delta)*kgrid[ik_private])
        d = d - 0.5*phi*np.power(kgrid/k_private - (1-delta), 2)*k_private

        iss_costs = np.where(d < 0, cd*d - xid, 0.0)
        pay_tax = np.where(d > 0, taue*d, 0.0)
        F = d + iss_costs - pay_tax

        # Calculating expected value given current prod draw
        Hentry = (M*Px)[ix, :][None, :, None] * Pprod[:, None, :]

        # If calculating entry value of the model with debt, assume that
        # the firm is allowed to enter with zero debt only
        if debt:
            EMV = np.tensordot(self.Vdebt[:, 0, :, :], Hentry, axes=[(1, 2), (1, 2)])
        else:
            EMV = np.tensordot(self.Vequity, Hentry, axes=[(1, 2), (1, 2)])

        tomax = F[:, None] + EMV
        Vmax = np.max(tomax, axis=0)

        if return_add:
            return Vmax, Pprod, np.argmax(tomax, axis=0)
        else:
            return Vmax


    def calcEntrySignalBound(self, ik_private, centry, debt=False):
        '''For each aggregate state calculate the minimum signal above
        which a prospective entrant enters'''

        # Array to fill bounds
        q_entry_bound = np.zeros(self.Nx)

        for ix in range(self.Nx):

            # Search in a wide grid before doing bisection
            q = np.linspace(0.01, 10.0, num=110)
            entry_value = self.calcEntryValue(q, ik_private, ix, debt=debt)
            ind_above = np.searchsorted(entry_value - centry, 0.0)

            # If enter for any value of id. state
            if ind_above == 0:
                q_entry_bound[ix] = q[0]
                continue

            # If DO NOT enter for any value of id. state
            if ind_above == q.shape[0]:
                q_entry_bound[ix] = q[-1]
                continue

            ind_below = ind_above - 1
            qupp = q[ind_above]
            qlow = q[ind_below]

            # Bisection to pin down the number more precisely
            dist, it = qupp - qlow, 0
            while dist > 1e-6 and it < 100:
                qcurrent = 0.5*(qlow + qupp)
                net_value = self.calcEntryValue(np.array([qcurrent]), ik_private, ix, debt=debt)
                net_value +=  - centry
                if net_value > 0.0:
                    qupp = qcurrent
                else:
                    qlow = qcurrent
                dist = qupp - qlow
                it += 1

            q_entry_bound[ix] = qcurrent

        return q_entry_bound


    def simulateEntrants(self, ix_path, ik_private, entrants_signals, entry_prod_u, entry_signal_bound, debt=False):
        q = entrants_signals
        q_bound = entry_signal_bound
        num_periods = ix_path.shape[0]
        num_prospective_entrants = entrants_signals.shape[1]

        total_entrants = 0
        iy_entry = np.ones((num_periods, num_prospective_entrants*num_periods), dtype=int)*(-99)
        ik_entry = np.ones((num_periods, num_prospective_entrants*num_periods), dtype=int)*(-99)
        if debt:
            ib_entry = np.ones((num_periods, num_prospective_entrants*num_periods), dtype=int)*(-99)
        entry_paths = np.zeros((num_periods, num_prospective_entrants*num_periods), dtype=bool)

        for t in range(num_periods - 1):
            # Getting signals of entrants in the current periods
            ix = ix_path[t]
            q_current = q[t, :]
            q_entrants = q_current[q_current > q_bound[ix]]
            num_entrants = q_entrants.shape[0]

            # Drawing productivity and capital of entrants in the current period
            _, Pprod, ik_entrants = self.calcEntryValue(q_entrants, ik_private, ix, return_add=True, debt=debt)
            iy_entrants = np.zeros(num_entrants, dtype=int)
            for ientrant in range(num_entrants):
                # iy_entrants[ientrant] = np.random.choice(range(self.Ny), size=1, p=Pprod[ientrant, :])
                iy_entrants[ientrant] = np.searchsorted(np.cumsum(Pprod[ientrant, :]), entry_prod_u[t, ientrant])
            
            # Writing into arrays for simulations
            iy_entry[t+1, total_entrants:(total_entrants + num_entrants)] = iy_entrants
            ik_entry[t+1, total_entrants:(total_entrants + num_entrants)] = ik_entrants
            if debt:
                ib_entry[t+1, total_entrants:(total_entrants + num_entrants)] = 0
            entry_paths[t+1, total_entrants:(total_entrants + num_entrants)] = True
            
            total_entrants += num_entrants

        iy_entry = iy_entry[:, :total_entrants]
        ik_entry = ik_entry[:, :total_entrants]
        if debt:
            ib_entry = ib_entry[:, :total_entrants]
        entry_paths = entry_paths[:, :total_entrants]

        if debt:
            return iy_entry, ik_entry, ib_entry, entry_paths
        else:
            return iy_entry, ik_entry, entry_paths

    
    def formatEquitySimulation(self, sim_equity, ix_path):
        '''Rewrite the numpy array with simulation output into a pandas
        dataframe to make the analysis of simulation easier'''
        total_entrants = sim_equity.shape[2]
        num_periods = ix_path.shape[0]
        id_full = np.array(range(total_entrants), dtype=int)

        df_list=[]
        for t in range(num_periods):
            ind_operate = sim_equity[0, t, :] > -99
            id_operate = id_full[ind_operate]

            df_to_append = pd.DataFrame({
                "id": id_operate,
                "iy": np.array(sim_equity[1, t, ind_operate], dtype=int),
                "ik": np.array(sim_equity[0, t, ind_operate], dtype=int),
                "k": self.kgrid[np.array(sim_equity[0, t, ind_operate], dtype=int)],
                "F": sim_equity[2, t, ind_operate],
                "exit": sim_equity[3, t, ind_operate],
                "V": sim_equity[4, t, ind_operate]
            })
            df_to_append["ix"] = ix_path[t]
            df_to_append["t"] = t
            df_list.append(df_to_append)

        return pd.concat(df_list, ignore_index=True)


    def formatDebtSimulation(self, sim_debt, ix_path):
        '''Rewrite the numpy array with simulation output into a pandas
        dataframe to make the analysis of simulation easier'''
        total_entrants = sim_debt.shape[2]
        num_periods = ix_path.shape[0]
        id_full = np.array(range(total_entrants), dtype=int)

        df_list=[]
        for t in range(num_periods):
            ind_operate = sim_debt[0, t, :] > -99
            id_operate = id_full[ind_operate]

            df_to_append = pd.DataFrame({
                "id": id_operate,
                "iy": np.array(sim_debt[2, t, ind_operate], dtype=int),
                "ik": np.array(sim_debt[0, t, ind_operate], dtype=int),
                "k": self.kgrid[np.array(sim_debt[0, t, ind_operate], dtype=int)],
                "ib": np.array(sim_debt[1, t, ind_operate], dtype=int),
                "b": self.bgrid[np.array(sim_debt[1, t, ind_operate], dtype=int)],
                "F": sim_debt[3, t, ind_operate],
                "Fd": sim_debt[4, t, ind_operate],
                "rate": sim_debt[5, t, ind_operate],
                "exit": sim_debt[6, t, ind_operate],
                "V": sim_debt[7, t, ind_operate]
            })
            df_to_append["ix"] = ix_path[t]
            df_to_append["t"] = t
            df_list.append(df_to_append)

        return pd.concat(df_list, ignore_index=True)


    def simulate(self, ix_path, ik_private, centry, entrants_signals, entry_prod_u, prod_signals, debt=False):
        '''Master function to call all other simulation functions'''
        # 1. For each aggregate state, calculate boundary of id. productivity
        #    signal over which a prospective entrant enters
        entry_signal_bound = self.calcEntrySignalBound(ik_private, centry, debt=debt)
        
        # 2. For each time period calculating entry decisions of prospective entrants
        # entrants_signals = np.random.pareto(omega, size=(num_periods, num_prospective_entrants))
        if not debt:
            # print("Here 1")
            iy_entry, ik_entry, entry_paths = self.simulateEntrants(
                ix_path, ik_private, entrants_signals, entry_prod_u, entry_signal_bound)
            
        else:
            iy_entry, ik_entry, ib_entry, entry_paths = self.simulateEntrants(
                ix_path, ik_private, entrants_signals, entry_prod_u, entry_signal_bound, debt=True)
        total_entrants = iy_entry.shape[1]

        ################################################################################
        # 3. Drawing productivities (need to find a way to fix these transitions)
        id_prod_rvs = prod_signals[:, :total_entrants]
        ################################################################################

        # 4. Running the main simulation
        if not debt:
            sim = simulatePoliciesEquity(
                ix_path, iy_entry, ik_entry, self.Vequity, self.ikp_opt, self.exit_opt, self.F_opt, 
                self.Py, id_prod_rvs)
        else:
            sim = simulatePoliciesDebt(
                ix_path, iy_entry, ik_entry, ib_entry, self.Vdebt, self.ikp_opt_debt, self.ibp_opt_debt, self.exit_opt_debt, 
                self.F_opt_debt, self.Fd_opt_debt, self.rate_opt_debt, self.Py, id_prod_rvs)


        # 5. Formatting simulation output into a dataframe
        if not debt:
            self.sim_equity_df = self.formatEquitySimulation(sim, ix_path)
            print(self.sim_equity_df.shape)
            self.sim_equity_df = generateVariablesEquity(self.sim_equity_df)
        else:
            self.sim_debt_df = self.formatDebtSimulation(sim, ix_path)
            print(self.sim_debt_df.shape)
            self.sim_debt_df = generateVariablesDebt(self.sim_debt_df, self.Rf)


    def simulateShort(self, ix_path, ik_private, centry, entrants_signals, entry_prod_u, prod_signals, debt=False):
        '''Simulate the economy without calculating additional columns and without writing 
        the output into the model object'''
        # 1. For each aggregate state, calculate boundary of id. productivity
        #    signal over which a prospective entrant enters
        entry_signal_bound = self.calcEntrySignalBound(ik_private, centry, debt=debt)
        
        # 2. For each time period calculating entry decisions of prospective entrants
        # entrants_signals = np.random.pareto(omega, size=(num_periods, num_prospective_entrants))
        if not debt:
            iy_entry, ik_entry, entry_paths = self.simulateEntrants(
                ix_path, ik_private, entrants_signals, entry_prod_u, entry_signal_bound)
        else:
            iy_entry, ik_entry, ib_entry, entry_paths = self.simulateEntrants(
                ix_path, ik_private, entrants_signals, entry_prod_u, entry_signal_bound, debt=True)
        total_entrants = iy_entry.shape[1]

        ################################################################################
        # 3. Drawing productivities (need to find a way to fix these transitions)
        id_prod_rvs = prod_signals[:, :total_entrants]
        ################################################################################

        # 4. Running the main simulation
        if not debt:
            sim = simulatePoliciesEquity(
                ix_path, iy_entry, ik_entry, self.Vequity, self.ikp_opt, self.exit_opt, self.F_opt, 
                self.Py, id_prod_rvs)
            sim_df = self.formatEquitySimulation(sim, ix_path)
        else:
            sim = simulatePoliciesDebt(
                ix_path, iy_entry, ik_entry, ib_entry, self.Vdebt, self.ikp_opt_debt, self.ibp_opt_debt, self.exit_opt_debt, 
                self.F_opt_debt, self.Fd_opt_debt, self.rate_opt_debt, self.Py, id_prod_rvs)
            sim_df = self.formatDebtSimulation(sim, ix_path)

        return sim_df


    ########################################################################
    # Functions to calculate various components of firm value


    def calcFlowValue(self, F, debt=False):
        '''
        Calculate value of the claim that gives F(k, b, x, y) in each time
        period until the firm exits
        '''
        H = self.H

        if debt:
            Nk, Nb, Nx, Ny = self.Vdebt.shape
            Vprev = np.zeros((Nk, Nb, Nx, Ny))
        else:
            Nk, Nx, Ny = self.Vequity.shape
            Vprev = np.zeros((Nk, Nx, Ny))


        ix_full_ind = np.arange(0, self.Nx, step=1, dtype=int)
        ix_full_ind = np.repeat(ix_full_ind[:, np.newaxis], Ny, axis=1)
        ix_full_ind = np.repeat(ix_full_ind[np.newaxis, :, :], Nk, axis=0)

        iy_full_ind = np.arange(0, Ny, step=1, dtype=int)
        iy_full_ind = np.repeat(iy_full_ind[np.newaxis, :], Nx, axis=0)
        iy_full_ind = np.repeat(iy_full_ind[np.newaxis, :, :], Nk, axis=0)

        if debt:
            ix_full_ind = np.repeat(ix_full_ind[:, np.newaxis, :, :], Nb, axis=1)
            iy_full_ind = np.repeat(iy_full_ind[:, np.newaxis, :, :], Nb, axis=1)

        # Getting capital (and debt) policies
        if debt:
            ikp_opt = self.ikp_opt_debt
            ibp_opt = self.ibp_opt_debt
            exit_opt = self.exit_opt_debt   
        else:
            ikp_opt = self.ikp_opt
            exit_opt = self.exit_opt   
    
        # Iterating value of tax shield until convergence:
        it, max_iter, dist, tol = 0, 10000, 1e8, 1e-8
        while dist > tol and it < max_iter:
            if it % 50 == 0:
                print(f"{it}: {dist}")
            
            if debt:
                EMV = np.tensordot(Vprev, H, axes=([2, 3], [1, 3]))
                EMVopt = EMV[ikp_opt, ibp_opt, ix_full_ind, iy_full_ind]
            else:
                EMV = np.tensordot(Vprev, H, axes=([1, 2], [1, 3]))
                EMVopt = EMV[ikp_opt, ix_full_ind, iy_full_ind]

            Vnew = F + EMVopt
            Vnew = np.where(exit_opt, 0.0, Vnew)
            dist = np.max(np.abs(Vnew - Vprev))
            Vprev = Vnew
            it += 1

        return Vnew


    def calcValueOfTaxShield(self):
        Nk, Nb, Nx, Ny = self.Vdebt.shape
        bgrid = self.bgrid
        Rf = self.Rf

        # Calculating flow payments from tax shield for each state
        F_tax_shield = bgrid[:, None]*Rf[None, :]*self.tauc
        F_tax_shield = np.repeat(F_tax_shield[np.newaxis, :, :], Nk, axis=0)
        F_tax_shield = np.repeat(F_tax_shield[:, :, :, np.newaxis], Ny, axis=3)

        self.V_tax_shield = self.calcFlowValue(F_tax_shield, debt=True)

        
    def calcValueOfFinCostsEquity(self):
        F_fin_costs = np.where(self.d_opt < 0, -self.d_opt*self.cd + self.xid, 0.0)
        self.V_fin_costs_equity = self.calcFlowValue(F_fin_costs, debt=False)


    def calcValueOfFinCostsDebt(self):
        F_fin_costs = np.where(self.d_opt_debt < 0, -self.d_opt_debt*self.cd + self.xid, 0.0)
        self.V_fin_costs_debt = self.calcFlowValue(F_fin_costs, debt=True)


    def calcValues(self):
        '''Calculates PV of tax shield and external financing cost and adds them to the
        simulation table'''

        # Calculating values
        self.calcValueOfTaxShield()
        self.calcValueOfFinCostsEquity()
        self.calcValueOfFinCostsDebt()

        # Adding to simulation dataframes
        sim_ind = np.array(self.sim_equity_df[["ik", "ix", "iy"]])
        self.sim_equity_df["VF"] = self.V_fin_costs_equity[sim_ind[:, 0], sim_ind[:, 1], sim_ind[:, 2]]
        self.sim_equity_df["VF_to_k"] = self.sim_equity_df["VF"]/self.sim_equity_df["k"]

        sim_ind = np.array(self.sim_debt_df[["ik", "ib", "ix", "iy"]])
        self.sim_debt_df["VF"] = self.V_fin_costs_debt[sim_ind[:, 0], sim_ind[:, 1], sim_ind[:, 2], sim_ind[:, 3]]
        self.sim_debt_df["VF_to_k"] = self.sim_debt_df["VF"]/self.sim_debt_df["k"]

        sim_ind = np.array(self.sim_debt_df[["ik", "ib", "ix", "iy"]])
        self.sim_debt_df["VTS"] = self.V_tax_shield[sim_ind[:, 0], sim_ind[:, 1], sim_ind[:, 2], sim_ind[:, 3]]
        self.sim_debt_df["VTS_to_k"] = self.sim_debt_df["VTS"]/self.sim_debt_df["k"]
