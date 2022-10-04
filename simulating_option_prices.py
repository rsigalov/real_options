####################################################################
# Author: Roman Sigalov
#
# This script 
#  (1) loads solution for equity and debt models solved on 
#      a GPU server
#  (2) simulates debt and equity models
#  (3) calculates asset pricing (expected returns, physical and
#      risk neutral variance of returns) and option (iv curves)
#      moments
####################################################################

import sys
import json
import pandas as pd
import numpy as np
from FinancingModel import financingModel, stationaryDistMarkov
from scipy.stats import norm
from scipy.optimize import minimize, fsolve, newton
import matplotlib.pyplot as plt
from tqdm import tqdm
# from numba import njit
import timeit
import os

from multiprocessing import Pool

# # not sure why doesn't the interactive mode update the working directory
# import os
# os.chdir("/Users/rsigalov/Dropbox/Projects/Endogenous Financing Costs/code")

def prepareModel(model_name):
    # Loading adjusted parameters
    with open(f"risky_debt_model_solution/model_metadata_{model_name}.json") as file:
        metadata = json.load(file)

    # Loading values and policy function
    Vdebt = np.load(f"risky_debt_model_solution/Vdebt_{model_name}.npy")
    policy_debt = np.load(f"risky_debt_model_solution/policy_debt_{model_name}.npy")
    kgrid = np.load(f"risky_debt_model_solution/kgrid_{model_name}.npy")
    bgrid = np.load(f"risky_debt_model_solution/bgrid_{model_name}.npy")

    # Creating the model
    model_params = metadata["params"]
    model_params["cfprop"] = 0.0
    model = financingModel(metadata["params"])

    # Solving equity only model
    model.solveForTerminalValue(kgrid=kgrid)
    model.calcFlowsEquityIssuance()
    model.solveEquityIssuance()
    model.calcPoliciesEquityIssuance()

    # Writing risky debt model
    model.writeDebtSolution(bgrid, Vdebt, policy_debt)

    return model, metadata

def BS_call_price(S0, r, K, sigma, T):
        
    d1 = (np.log(S0/K) + (r + np.power(sigma,2)/2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    p1 = S0 * norm.cdf(d1)
    p2 = np.exp(-r*T) * K * norm.cdf(d2)
    
    return p1 - p2


def BS_put_price(S0, r, K, sigma, T):
    
    d1 = (np.log(S0/K) + (r + np.power(sigma,2)/2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)

    p1 = norm.cdf(-d2) * K * np.exp(-r*T)
    p2 = norm.cdf(-d1) * S0

    return p1 - p2


def norm_pdf(x):
    return 1/np.sqrt(2*np.pi) * np.exp(-0.5*np.power(x, 2))


def dOptdsigma(S0, r, K, sigma, T):
    '''Same for calls and puts'''
    d1 = (np.log(S0/K) + (r + np.power(sigma,2)/2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)

    dd1dsigma = np.power(sigma, 2)*np.power(T, 1.5) 
    dd1dsigma += - np.power(T, 0.5)*(np.log(S0/K) + (r + 0.5*np.power(sigma, 2))*T)
    dd1dsigma *= 1/(np.power(sigma, 2) * T)

    dOptdsigma = S0*norm_pdf(d1)*dd1dsigma 
    dOptdsigma += -K*np.exp(-r*T)*norm_pdf(d2)*(dd1dsigma - np.sqrt(T))

    return dOptdsigma
    

def calculate_implied_vol(opt_price, opt_type, S0, r, K, T):    
    def to_minimize(x):
        if opt_type == "C":
            return (BS_call_price(S0, r, K, x, T) - opt_price)**2
        elif opt_type == "P":
            return (BS_put_price(S0, r, K, x, T) - opt_price)**2
        else:
            raise ValueError("cp_flag should be either 'C' or 'P'")
        
    res = minimize(to_minimize, [0.18], method='nelder-mead', 
                options={'xtol': 1e-8, 'disp': False})
    
    if res.success:
        return res.x[0]
    else:
        return np.nan


def calculate_implied_vol_call(opt_price, S0, r, K, T):
    def f(x):
        return BS_call_price(S0, r, K, x, T) - opt_price

    def fprime(x):
        return dOptdsigma(S0, r, K, x, T)

    # sigma_start = (2*np.pi/T)**0.5 * opt_price/S0

    res = fsolve(f, [0.2], fprime=fprime)[0]
    # res = newton(f, 0.1, fprime=fprime, disp=False)
    # res = toms748(f, 0.0001, 2.0)

    return res


def calculate_implied_vol_put(opt_price, S0, r, K, T):
    def f(x):
        return BS_put_price(S0, r, K, x, T) - opt_price

    def fprime(x):
        return dOptdsigma(S0, r, K, x, T)

    # sigma_start = (2*np.pi/T)**0.5 * (opt_price + S0 - np.exp(-r*T)*K)/S0

    res = fsolve(f, [0.2], fprime=fprime)[0]
    # res = newton(f, 0.1, fprime=fprime, disp=False)
    # res = toms748(f, 0.0001, 2.0)

    return res


def priceOptionsEquity(model, ik, ix, iy):
    '''Prices 1-period options for a particular state'''

    # Getting objects from the model
    M = model.M
    H = model.H
    EM = np.sum(model.Px[ix, :]*M[ix, :])

    # Current value of equity
    V0 = model.Vequity[ik, ix, iy]
    D0 = model.F_opt[ik, ix, iy]
    P0 = V0 - D0

    # Distribution of future value of equity
    ik1 = int(model.ikp_opt[ik, ix, iy])
    V1 = model.Vequity[ik1, :, :]
    H1 = H[ix, :, iy, :]

    # Pricing calls and puts
    # Krel_grid = np.linspace(0.5, 1.5, num=15)
    Rf = 1/EM - 1
    rf = np.log(1 + Rf)
    F = np.exp(rf)*P0 # Forward rate
    Krel_grid = np.array([0.7, 1.0, 1.3])
    K_grid = F*Krel_grid
    P_grid = np.zeros_like(K_grid)
    C_grid = np.zeros_like(K_grid)
    for iK, K in enumerate(K_grid):
        P_grid[iK] = np.sum(H1*np.maximum(K - V1, 0))
        C_grid[iK] = np.sum(H1*np.maximum(V1 - K, 0))

    # Slower but more reliable way
    # P_IV = [calculate_implied_vol(price, "P", P0, rf, strike, 1.0) for price, strike in zip(P_grid, K_grid)]
    # C_IV = [calculate_implied_vol(price, "C", P0, rf, strike, 1.0) for price, strike in zip(C_grid, K_grid)]
    
    IV_low = calculate_implied_vol_put(P_grid[0], P0, rf, K_grid[0], 1.0)
    IV_ATM = calculate_implied_vol_put(P_grid[1], P0, rf, K_grid[1], 1.0)
    IV_high = calculate_implied_vol_call(C_grid[2], P0, rf, K_grid[2], 1.0)

    return IV_low, IV_ATM, IV_high



def priceOptionsDebt(model, ik, ib, ix, iy):
    '''Prices 1-period options for a particular state'''

    # Getting objects from the model
    M = model.M
    H = model.H
    EM = np.sum(model.Px[ix, :]*M[ix, :])

    # Current value of equity
    V0 = model.Vdebt[ik, ib, ix, iy]
    D0 = model.F_opt_debt[ik, ib, ix, iy]
    P0 = V0 - D0

    # Distribution of future value of equity
    ik1 = int(model.ikp_opt_debt[ik, ib, ix, iy])
    ib1 = int(model.ibp_opt_debt[ik, ib, ix, iy])
    V1 = model.Vdebt[ik1, ib1, :, :]
    H1 = H[ix, :, iy, :]

    # Pricing calls and puts
    # Krel_grid = np.linspace(0.5, 1.5, num=15)
    Rf = 1/EM - 1
    rf = np.log(1 + Rf)
    F = np.exp(rf)*P0 # Forward rate
    Krel_grid = np.array([0.7, 1.0, 1.3])
    K_grid = Krel_grid*F
    P_grid = np.zeros_like(K_grid)
    C_grid = np.zeros_like(K_grid)
    for iK, K in enumerate(K_grid):
        P_grid[iK] = np.sum(H1*np.maximum(K - V1, 0))
        C_grid[iK] = np.sum(H1*np.maximum(V1 - K, 0))

    # Inverting Black-Scholes option prices to get implied volatilities
    
    # P_IV = [calculate_implied_vol(price, "P", P0, rf, strike, 1.0) for price, strike in zip(P_grid, K_grid)]
    # C_IV = [calculate_implied_vol(price, "C", P0, rf, strike, 1.0) for price, strike in zip(C_grid, K_grid)]
    P_IV = [calculate_implied_vol_put(price, P0, rf, strike, 1.0) for price, strike in zip(P_grid, K_grid)]
    C_IV = [calculate_implied_vol_call(price, P0, rf, strike, 1.0) for price, strike in zip(C_grid, K_grid)]

    return Krel_grid, P_grid, C_grid, P_IV, C_IV


def calcVariancesEquity(model, ik, ix, iy):
    # Getting objects from the model
    M = model.M
    H = model.H
    EM = np.sum(model.Px[ix, :]*M[ix, :])

    # Current value of equity
    V0 = model.Vequity[ik, ix, iy]
    D0 = model.F_opt[ik, ix, iy]
    P0 = V0 - D0

    # Distribution of future value of equity
    ik1 = int(model.ikp_opt[ik, ix, iy])
    V1 = model.Vequity[ik1, :, :]
    H1 = H[ix, :, iy, :]
    R = V1/P0

    # Physical variance
    P = model.Px[ix, :][:, None] * model.Py[iy, :][None, :]
    ER = np.sum(P * R)
    VarPR = np.sum(P * np.power(R - ER, 2))

    # Risk-neutral variance
    Q = H1/EM
    EQR = np.sum(Q * R)
    VarQR = np.sum(Q * np.power(R - EQR, 2))

    return VarPR, VarQR


def calcVariancesDebt(model, ik, ib, ix, iy):
    # Getting objects from the model
    M = model.M
    H = model.H
    EM = np.sum(model.Px[ix, :]*M[ix, :])

    # Current value of equity
    V0 = model.Vdebt[ik, ib, ix, iy]
    D0 = model.F_opt_debt[ik, ib, ix, iy]
    P0 = V0 - D0

    # Distribution of future value of equity
    ik1 = int(model.ikp_opt_debt[ik, ib, ix, iy])
    ib1 = int(model.ibp_opt_debt[ik, ib, ix, iy])
    V1 = model.Vdebt[ik1, ib1, :, :]
    H1 = H[ix, :, iy, :]
    R = V1/P0

    # Physical variance
    P = model.Px[ix, :][:, None] * model.Py[iy, :][None, :]
    ER = np.sum(P * R)
    VarPR = np.sum(P * np.power(R - ER, 2))

    # Risk-neutral variance
    Q = H1/EM
    EQR = np.sum(Q * R)
    VarQR = np.sum(Q * np.power(R - EQR, 2))

    return VarPR, VarQR



def prepareParallelInputsEquity(model, ik, ix, iy):
    # Current value of equity excl. dividends
    V0 = model.Vequity[ik, ix, iy]
    D0 = model.F_opt[ik, ix, iy]
    P0 = V0 - D0

    # Next period value of equity
    ik1 = int(model.ikp_opt[ik, ix, iy])
    V1 = model.Vequity[ik1, :, :]

    # Other variables
    H1 = model.H[ix, :, iy, :]
    EM = np.sum(model.Px[ix, :]*model.M[ix, :])

    return (P0, V1, H1, EM)

def prepareParallelInputsDebt(model, ik, ib, ix, iy):
    # Current value of equity excl. dividends
    V0 = model.Vdebt[ik, ib, ix, iy]
    D0 = model.F_opt_debt[ik, ib, ix, iy]
    P0 = V0 - D0

    # Next period value of equity
    ik1 = int(model.ikp_opt_debt[ik, ib, ix, iy])
    ib1 = int(model.ibp_opt_debt[ik, ib, ix, iy])
    V1 = model.Vdebt[ik1, ib1, :, :]

    # Other variables
    H1 = model.H[ix, :, iy, :]
    EM = np.sum(model.Px[ix, :]*model.M[ix, :])

    return (P0, V1, H1, EM)


def priceOptionsParallel(inputs):
    '''Prices 1-period options for a particular state'''

    P0, V1, H1, EM = inputs

    # Pricing calls and puts
    Rf = 1/EM - 1
    rf = np.log(1 + Rf)
    F = np.exp(rf)*P0 # Forward rate

    # Calculating option prices
    Krel_grid = np.array([0.7, 1.0, 1.3])
    K_grid = F*Krel_grid
    P_grid = np.zeros_like(K_grid)
    C_grid = np.zeros_like(K_grid)
    for iK, K in enumerate(K_grid):
        P_grid[iK] = np.sum(H1*np.maximum(K - V1, 0))
        C_grid[iK] = np.sum(H1*np.maximum(V1 - K, 0))
    
    # Using put for low and atm strikes and call for high strike
    IV_low = calculate_implied_vol_put(P_grid[0], P0, rf, K_grid[0], 1.0)
    IV_ATM = calculate_implied_vol_put(P_grid[1], P0, rf, K_grid[1], 1.0)
    IV_high = calculate_implied_vol_call(C_grid[2], P0, rf, K_grid[2], 1.0)

    # Slow version
    # IV_low = calculate_implied_vol(P_grid[0], "P", P0, rf, K_grid[0], 1.0)
    # IV_ATM = calculate_implied_vol(P_grid[1], "P", P0, rf, K_grid[0], 1.0)
    # IV_high = calculate_implied_vol(C_grid[2], "C", P0, rf, K_grid[0], 1.0)

    # P_IV = [calculate_implied_vol(price, "P", P0, rf, strike, 1.0) for price, strike in zip(P_grid, K_grid)]
    # C_IV = [calculate_implied_vol(price, "C", P0, rf, strike, 1.0) for price, strike in zip(C_grid, K_grid)]

    return IV_low, IV_ATM, IV_high
    

def main(argv):

    model_name = argv[1]
    if argv[2] == "from_env":
        n_cores = int(os.getenv("LSB_MAX_NUM_PROCESSORS"))
    else:
        n_cores = int(argv[2])

    print("\n############################################################")
    print(f"Pricing options for model {model_name}")
    print(f"Cores used: {n_cores}")
    print("############################################################\n")

    # model_name = "options_14"

    print("\nLoading model")
    model, metadata = prepareModel(model_name)

    print("\nSimulating model")
    # Parameters for the simulation
    num_simulations = 250
    num_periods = 100
    ik_private = 30
    centry = 30.0
    np.random.seed(19960202)

    sim_equity_list = []
    sim_debt_list = []
    for isim in tqdm(range(num_simulations)):

        # Simulating path of aggregate state
        ix_path = np.zeros(num_periods, dtype=int)
        ix_path[0] = np.random.choice(range(model.Nx), size=1, p=stationaryDistMarkov(model.Px))[0]
        for t in range(num_periods-1):
            ix_path[t+1] = np.random.choice(range(model.Nx), size=1, p=model.Px[ix_path[t], :])

        # Fixing firm-specific random variables before simulating
        # to minimize the effect of randomness when comparing the results
        omega = 2.0
        num_prospective_entrants = 50
        entrants_signals = np.random.pareto(omega, size=(num_periods, num_prospective_entrants))
        entry_prod_u = np.random.uniform(size=(num_periods, num_prospective_entrants))
        prod_signals = np.random.uniform(size=(num_periods, num_prospective_entrants*num_periods))

        # Equity simulation
        sim_equity = model.simulateShort(ix_path, ik_private, centry, entrants_signals, entry_prod_u, prod_signals, debt=False)
        sim_equity["isim"] = isim + 1
        sim_equity = sim_equity[sim_equity["exit"] == 0.0]

        # Risky debt simulation
        sim_debt = model.simulateShort(ix_path, ik_private, centry, entrants_signals, entry_prod_u, prod_signals, debt=True)
        sim_debt["isim"] = isim + 1
        sim_debt = sim_debt[sim_debt["exit"] == 0.0]

        # Calculating number of firms for each state
        sim_equity = sim_equity.groupby(["ix", "iy", "ik"])["id"].count().rename("nobs").reset_index()
        sim_debt = sim_debt.groupby(["ix", "iy", "ik", "ib"])["id"].count().rename("nobs").reset_index()

        sim_equity_list.append(sim_equity)
        sim_debt_list.append(sim_debt)

    sims_equity = pd.concat(sim_equity_list, ignore_index=True)
    sims_debt = pd.concat(sim_debt_list, ignore_index=True)

    # Summing observations for each state
    sims_equity = sims_equity.groupby(["ik", "ix", "iy"])["nobs"].sum().reset_index()
    sims_debt = sims_debt.groupby(["ik", "ib", "ix", "iy"])["nobs"].sum().reset_index()

    # Ordering by frequency to exclude states that are extremely rare
    # (there is also a chance to have problems with calculating IVs
    # when a state is corner and thus extremely rare)
    sims_equity = sims_equity.sort_values("nobs", ascending=False).reset_index().drop(columns="index")
    sims_debt = sims_debt.sort_values("nobs", ascending=False).reset_index().drop(columns="index")

    # Looking at the CDF of states: where does the cdf exceeds 99.5%
    state_cdf_equity = sims_equity["nobs"].cumsum()/sims_equity["nobs"].sum()
    sims_equity = sims_equity[state_cdf_equity <= 0.995]
    state_cdf_debt = sims_debt["nobs"].cumsum()/sims_debt["nobs"].sum()
    sims_debt = sims_debt[state_cdf_debt <= 0.995]

    # ################################################################
    # # For debugging ONLY using a short dataframe
    # sims_equity = sims_equity.iloc[:10000]
    # sims_debt = sims_debt.iloc[:10000]
    # ################################################################

    print("\nCalculating physical and risk-neutral variances")
    vars_list_equity = []
    for irow, row in tqdm(sims_equity.iterrows(), total=sims_equity.shape[0]):
        vars_list_equity.append(
            calcVariancesEquity(model, row["ik"], row["ix"], row["iy"])
        )

    vars_list_debt = []
    for irow, row in tqdm(sims_debt.iterrows(), total=sims_debt.shape[0]):
        vars_list_debt.append(
            calcVariancesDebt(model, row["ik"], row["ib"], row["ix"], row["iy"])
        )

    print("\nSaving variances")
    vars_df_equity = pd.DataFrame(vars_list_equity, columns=["VarP", "VarQ"])
    vars_df_equity = sims_equity.join(vars_df_equity)
    vars_df_debt = pd.DataFrame(vars_list_debt, columns=["VarP", "VarQ"])
    vars_df_debt = sims_debt.join(vars_df_debt)
    vars_df_equity.to_csv(f"simulation_distributions/variances_equity_{model_name}.csv", index=False)
    vars_df_debt.to_csv(f"simulation_distributions/variances_debt_{model_name}.csv", index=False)

    print("\nPreparing parallel inputs")
    equity_inputs_list = []
    for irow, row in tqdm(sims_equity.iterrows(), total=sims_equity.shape[0]):
        equity_inputs_list.append(
            prepareParallelInputsEquity(model, row["ik"], row["ix"], row["iy"])
        )

    debt_inputs_list = []
    for irow, row in tqdm(sims_debt.iterrows(), total=sims_debt.shape[0]):
        debt_inputs_list.append(
            prepareParallelInputsDebt(model, row["ik"], row["ib"], row["ix"], row["iy"])
        )

    print("\nCalculating implied volatilities for EQUITY model")
    start_time = timeit.default_timer()
    with Pool(n_cores) as p:
        ivs_list_equity = p.map(priceOptionsParallel, equity_inputs_list)
    end_time = timeit.default_timer()
    print(f"processed in {end_time - start_time:.2f} s.")

    print("\nCalculating implied volatilities for DEBT model")
    start_time = timeit.default_timer()
    with Pool(n_cores) as p:
        ivs_list_debt = p.map(priceOptionsParallel, debt_inputs_list)
    end_time = timeit.default_timer()
    print(f"processed in {end_time - start_time:.2f} s")

    print("\nPreparing output and saving to simulation_distributions/")
    ivs_df_equity = pd.DataFrame(ivs_list_equity, columns=["IV_low", "IV_ATM", "IV_high"])
    ivs_df_equity = sims_equity.join(ivs_df_equity)

    ivs_df_debt = pd.DataFrame(ivs_list_debt, columns=["IV_low", "IV_ATM", "IV_high"])
    ivs_df_debt = sims_debt.join(ivs_df_debt)

    ivs_df_equity.to_csv(f"simulation_distributions/simulation_distributions_equity_{model_name}.csv", index=False)
    ivs_df_debt.to_csv(f"simulation_distributions/simulation_distributions_debt_{model_name}.csv", index=False)

    print("\nAdding firm characteristics to distributions")
    # Looping over states and adding firm characteristics
    char_list_equity = []
    for irow, row in tqdm(ivs_df_equity.iterrows(), total=ivs_df_equity.shape[0]):
        ik, ix, iy = int(row["ik"]), int(row["ix"]), int(row["iy"])
        ik_next = int(model.ikp_opt[ik, ix, iy])
        k = model.kgrid[ik]
        k_next = model.kgrid[ik_next]
        b = 0.0
        V = model.Vequity[ik, ix, iy]
        bm = (k - b)/V
        lev = b/k
        inv = k_next - (1 - model.delta)*k
        size = np.log(V)

        char_list_equity.append(
            (bm, lev, inv, size)
        )

    char_df_equity = pd.DataFrame(char_list_equity, columns=["bm", "lev", "inv", "size"])

    char_list_debt = []
    for irow, row in tqdm(ivs_df_debt.iterrows(), total=ivs_df_debt.shape[0]):
        ik, ib, ix, iy = int(row["ik"]), int(row["ib"]), int(row["ix"]), int(row["iy"])
        ik_next = int(model.ikp_opt_debt[ik, ib, ix, iy])
        k = model.kgrid[ik]
        k_next = model.kgrid[ik_next]
        b = model.bgrid[ib]
        V = model.Vdebt[ik, ib, ix, iy]
        bm = (k - b)/V
        lev = b/k
        inv = (k_next - (1 - model.delta)*k)/k
        size = np.log(V)

        char_list_debt.append(
            (bm, lev, inv, size, k, b, V)
        )

    char_df_debt = pd.DataFrame(char_list_debt, columns=["bm", "lev", "inv", "size", "k", "b", "V"])

    # Combining states, option variables with characteristics
    full_df_equity = pd.merge(ivs_df_equity, vars_df_equity, on=["ik", "ix", "iy", "nobs"])
    full_df_equity = full_df_equity.join(char_df_equity)

    full_df_debt = pd.merge(ivs_df_debt, vars_df_debt, on=["ik", "ib", "ix", "iy", "nobs"])
    full_df_debt = full_df_debt.join(char_df_debt)

    # Saving
    full_df_equity.to_csv(f"simulation_distributions/simulation_for_reg_equity_{model_name}.csv", index=False)
    full_df_debt.to_csv(f"simulation_distributions/simulation_for_reg_debt_{model_name}.csv", index=False)


if __name__ == "__main__":
    main(sys.argv)