# Code for "Real and Financial Options: Production Based Approach to Option Pricing"

## Abstract

Traditional option pricing literature starts with exogenous processes governing the joint evolution of the underlying price, stochastic volatility and jumps. This approach is, by design, silent on the fundamental determinants of the cross-section of equity options. I document a heterogeneous effect of firm fundamentals such as book-to-market on the relative prices of options that varies with the aggregate state of the economy. In particular, growth companies have higher skew in expansions but not in recessions. I develop a stylized production based asset pricing model with real options consistent with this evidence. A full structural model with leverage is able to match the observed patterns both qualitatively and quantitatively. Additionally, I show that the real options model can rationalize recently proposed delta-hedged option strategies based on profitability and book-to-market.

## Code description

There are two parts to the code.

First is the continuous time model solved via a descitization of the state space and finite differences with upwind scheme following online appendix of [Achdou et al. (2022)](https://doi.org/10.1093/restud/rdab002) available at [Benjamin Moll's website](https://benjaminmoll.com/codes/). This part solves the continuous time model to show the difference in skew of value and growth firms in expansions and recessions. Additionally, it derives the partial sensitivity of the price of an option to price of risk -- the part of expected profits of a delta hedged option position that is heterogenous.

* `sensitivities_parallel.jl` -- iterates the Kolmogorov Forward Equation forward to generate distributions of equity issuance adjusted firm values
	* In the model, firms issue equity when they expand. This generates path dependence in the option prices.
	* This prevents solving for the option prices from the terminal payoff
	* Path dependence requires extending the state space to include the **dilution** factor
	* Due to this complication, solving for value distribution takes some and is optimized to run forward iterations in parallel for different initial states
* `continuous_time_real_options.jl` -- Plots figures for the main model taking the dstributions obtained from running `sensitivities_parallel.jl`

The second is a discrete time model with a more realistic investment process and risky debt. This part is very demanding computationally and heavily relies

* `discrete_time_model_gpu_acceleration.ipynb` -- a notebook optimized to leverage multiple GPUs on such GPU clouds as [LambdaLambs](https://lambdalabs.com). 
* `FinancingModel.py` -- class with functions for simulating the structural model
* `simulating_option_prices.py` -- script that simulates equity and debt models and calculates the distribution of option implied volatilities. Optimized to work in parallel


