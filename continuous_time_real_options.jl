################################################################
# Author: Roman Sigalov
#
# Solves continuous time real options model
################################################################

using Plots
using LinearAlgebra
using SparseArrays
using Distributions
using ProgressMeter
using Random
using HypothesisTests
using NLsolve
using CSV
using DataFrames

# Setting colors as in Mathematica
mathematicaColor1 = RGBA(0.37,0.51,0.71,1);
mathematicaColor2 = RGBA(0.88,0.61,0.14,1);
mathematicaColor3 = RGBA(0.56,0.69,0.19,1);

function formTransitionMatrixSimple(model_params)
    # Since can solve for v(λ) such that V(x, λ) = xv(λ) use the raw (not repeated) λ_grid
    λ = model_params.λ_grid

    # Unpacking coefficients
    κ, λbar, σλ = model_params.κ, model_params.λbar, model_params.σλ;
    μx, σx, ϕ, ρ, bπ = model_params.μx, model_params.σx, model_params.ϕ, model_params.ρ, model_params.bπ;
    Δλ, J = model_params.Δλ, model_params.J;

    # Useful indicators for upwind scheme: using forward derivative when drift is positive
    # and backward derivative when the drift is negative
    λup = map(var -> var >= 0.0 ? 1.0 : 0.0, κ*(λbar .- λ) .- bπ(λ).*σλ(λ)*σx*ρ);
    λdown = map(var -> var < 0.0 ? 1.0 : 0.0, κ*(λbar .- λ) .- bπ(λ).*σλ(λ)*σx*ρ);

    # Coefficient on v_{i-1}
    a = -λdown/Δλ .* (κ*(λbar .- λ) .- σx*ρ*bπ(λ).*σλ(λ) .+ ϕ*σx*ρ*σλ(λ)) .+ 0.5*σλ(λ).^2/Δλ^2;

    # Coefficient on v_{i}
    b = (μx .- bπ(λ)*ϕ*σx^2) .+ (-λup + λdown)/Δλ .* (κ*(λbar .- λ) .- σx*ρ*bπ(λ).*σλ(λ) .+ ϕ*σx*ρ*σλ(λ)) .- σλ(λ).^2/Δλ^2;

    # Coefficient on v_{i+1}
    c = λup/Δλ .* (κ*(λbar .- λ) .- σx*ρ*bπ(λ).*σλ(λ) .+ ϕ*σx*ρ*σλ(λ)) .+ 0.5*σλ(λ).^2/Δλ^2;

    # Helper vectors to account for reflecting barriers at the bottom and top of the state space
    λfirst = zeros(J);
    λfirst[1] = 1.0;
    λlast = zeros(J);
    λlast[end] = 1.0;

    # Forming diagonals
    diag0 = b .+ λfirst .* a .+ λlast .* c;
    diagp1 = c .- λlast .* c;
    diagm1 = a .- λfirst .* a;

    # Forming matrix A
    A = spdiagm(
        0 => diag0,
        1 => diagp1[1:(end-1)],
        -1 => diagm1[2:end]
    );

    return A
end


function solveModelConstPriceOfRisk(model_params; K=missing)
    if ismissing(K)
        K = model_params.K1;
    end

    # Since can solve for v(λ) such that V(x, λ) = xv(λ) use the raw (not repeated) λ_grid
    λ = model_params.λ_grid;
    bπ = model_params.bπ;

    # Unpacking coefficients
    μx, σx = model_params.μx, model_params.σx;
    α, r = model_params.α, model_params.r;
    ϕ = model_params.ϕ;

    # Starting value: value of a firm with a constant price of risk
    U = K^α;
    v = U ./ (r .+ bπ(λ)*ϕ*σx^2 .- μx);

    return v
end


function solveModelMature(model_params, A; K=missing)
    if ismissing(K)
        K = model_params.K1;
    end

    # Unpacking parameters
    r, α = model_params.r, model_params.α;

    # Starting value: value of a firm with a constant price of risk
    vstart = solveModelConstPriceOfRisk(model_params; K=K);
    α = model_params.α;
    U = K^α;
    
    # Preparing for iteration
    vprev = vstart;
    Δt = model_params.Δt;

    # Iterating
    it, dist = 1, 1e8
    while (it < 1000000) & (dist > 1e-8)
        vnext = U*Δt .+  vprev*(1-r*Δt) .+ (A*Δt)*vprev;
        dist = maximum(abs.(vnext .- vprev));
        if it % 10000 == 0
            @show dist
        end
        vprev = vnext;
        it = it + 1;
    end

    v = vprev;

    return v
end


function formTransitionMatrix(model_params)
    # Grids
    λ, x = model_params.λ, model_params.x;
    I, J = model_params.I, model_params.J;

    # Unpacking coefficients
    bπ, σλ = model_params.bπ, model_params.σλ;
    κ, λbar = model_params.κ, model_params.λbar;
    μx, σx, ρ = model_params.μx, model_params.σx, model_params.ρ;
    ϕ = model_params.ϕ;
    Δx, Δλ = model_params.Δx, model_params.Δλ;

    # Forming indicators for upwind scheme
    xup = map(var -> var >= 0 ? 1 : 0, μx .- bπ(λ)*ϕ*σx^2);
    xdown = map(var -> var < 0 ? 1 : 0, μx .- bπ(λ)*ϕ*σx^2);
    λup = map(var -> var >= 0 ? 1 : 0, κ.*(λbar .- λ) .- bπ(λ).*σλ(λ)*σx*ρ);
    λdown = map(var -> var < 0 ? 1 : 0, κ.*(λbar .- λ) .- bπ(λ).*σλ(λ)*σx*ρ);

    # Coefficient on V_{i-1,j}
    a = -x .* (μx .- bπ(λ)*ϕ*σx^2)/Δx .* xdown .+ 0.5*x.^2*ϕ^2*σx^2/Δx^2;
    a = a .+ x.*σλ(λ)*ϕ*σx*ρ/(Δx*Δλ) .* (xdown .* λup - xdown .* λdown);

    # Coefficient on V_{i,j}
    b = x .* (μx .- bπ(λ)*ϕ*σx^2)/Δx .* (-xup .+ xdown) .- x.^2*ϕ^2*σx^2/Δx^2;
    b = b .+ (κ.*(λbar .- λ) .- bπ(λ).*σλ(λ)*σx*ρ)/Δλ .* (-λup .+ λdown) .- σλ(λ).^2/Δλ^2;
    b = b .+ x.*σλ(λ)*ϕ*σx*ρ/(Δx*Δλ) .* (xup.*λup .- xup.*λdown .- xdown.*λup .+ xdown.*λdown);

    # Coefficient on V_{i+1,j}
    c = x .* (μx .- bπ(λ)*ϕ*σx^2)/Δx .* xup .+ 0.5*x.^2*ϕ^2*σx^2/Δx^2;
    c = c .+ x.*σλ(λ)*ϕ*σx*ρ/(Δx*Δλ) .* (-xup.*λup .+ xup.*λdown);

    # Coefficient on V_{i-1,j-1}
    f = x.*σλ(λ)*ϕ*σx*ρ/(Δx*Δλ) .* xdown.*λdown;

    # Coefficient on V_{i,j-1}
    g = -(κ.*(λbar .- λ) .- bπ(λ).*σλ(λ)*σx*ρ)/Δλ .* λdown .+ 0.5*σλ(λ).^2/Δλ^2;
    g = g .+ x.*σλ(λ)*ϕ*σx*ρ/(Δx*Δλ) .* (xup .* λdown .- xdown.*λdown);

    # Coefficient on V_{i+1,j-1}
    h = -x.*σλ(λ)*ϕ*σx*ρ/(Δx*Δλ) .* xup .* λdown;

    # Coefficient on V_{i-1,j+1}
    m = -x.*σλ(λ)*ϕ*σx*ρ/(Δx*Δλ) .* xdown .* λup;

    # Coefficient on V_{i,j+1}
    n = (κ.*(λbar .- λ) .- bπ(λ).*σλ(λ)*σx*ρ)/Δλ .* λup .+ 0.5*σλ(λ).^2/Δλ^2;
    n = n .+ x.*σλ(λ)*ϕ*σx*ρ/(Δx*Δλ) .* (-xup .* λup .+ xdown .* λup);

    # Coefficient on V_{i+1,j+1}
    p = x.*σλ(λ)*ϕ*σx*ρ/(Δx*Δλ) .* xup .* λup;

    # Helper vectors to account for reflecting barrier at the edges of the state space
    lastx = zeros(I);
    lastx[end] = 1.0;
    lastx = repeat(lastx, outer=J);
    firstx = zeros(I);
    firstx[1] = 1.0;
    firstx = repeat(firstx, outer=J);
    lastλ = zeros(J);
    lastλ[end] = 1.0;
    lastλ = repeat(lastλ, inner=I);
    firstλ = zeros(J);
    firstλ[1] = 1.0;
    firstλ = repeat(firstλ, inner=I);

    # Main diagonal strip (staying at the same λ)
    maindiag_0 = b .+ firstx .* a .+ lastx .* c .+ firstλ .* g .+ lastλ .* n;
    maindiag_0 = maindiag_0 .+ firstx.*firstλ.*f .+ lastx.*firstλ.*h .+ firstx.*lastλ.*m .+ lastx.*lastλ.*p;
    maindiag_p1 = c .+ firstλ.*h .+ lastλ.*p;
    maindiag_p1 = maindiag_p1.*(1 .- lastx);
    maindiag_m1 = a .+ firstλ.*f .+ lastλ.*m;
    maindiag_m1 = maindiag_m1.*(1 .- firstx);

    # Upper diagonal strip (λ moves up)
    uppdiag_0 = n .+ firstx.* m .+ lastx.*p;
    uppdiag_0 = uppdiag_0.*(1 .- lastλ);
    uppdiag_p1 = (1 .- lastλ).*(1 .- lastx).*p;
    uppdiag_m1 = (1 .- lastλ).*(1 .- firstx).*m;

    # Lower diagonal strip (λ moves down)
    lowdiag_0 = g .+ firstx.*f .+ lastx.*h;
    lowdiag_0 = lowdiag_0.*(1 .- firstλ);
    lowdiag_m1 = (1 .- firstλ).*(1 .- firstx).*f;
    lowdiag_p1 = (1 .- firstλ).*(1 .- lastx).*h;

    # Filling the diagonals
    A = spdiagm(
        0 => maindiag_0,
        1 => maindiag_p1[1:(end-1)],
        -1 => maindiag_m1[2:end],
        I => uppdiag_0[1:(end-I)],
        I+1 => uppdiag_p1[1:(end-I-1)],
        I-1 => uppdiag_m1[1:(end-I+1)],
        -I => lowdiag_0[(I+1):end],
        -I+1 => lowdiag_p1[I:end],
        -I-1 => lowdiag_m1[(I+2):end]
    );

    return A
end


function formTransitionMatrixPhysical(model_params)
    # Grids
    λ, x = model_params.λ, model_params.x;
    I, J = model_params.I, model_params.J;

    # Unpacking coefficients
    bπ, σλ = model_params.bπ, model_params.σλ;
    κ, λbar = model_params.κ, model_params.λbar;
    μx, σx, ρ = model_params.μx, model_params.σx, model_params.ρ;
    ϕ = model_params.ϕ;
    Δx, Δλ = model_params.Δx, model_params.Δλ;

    # Forming indicators for upwind scheme
    xup = map(var -> var >= 0 ? 1 : 0, μx);
    xdown = map(var -> var < 0 ? 1 : 0, μx);
    λup = map(var -> var >= 0 ? 1 : 0, κ.*(λbar .- λ));
    λdown = map(var -> var < 0 ? 1 : 0, κ.*(λbar .- λ));

    # Coefficient on V_{i-1,j}
    a = -x .* μx/Δx .* xdown .+ 0.5*x.^2*ϕ^2*σx^2/Δx^2;
    a = a .+ x.*σλ(λ)*ϕ*σx*ρ/(Δx*Δλ) .* (xdown .* λup - xdown .* λdown);

    # Coefficient on V_{i,j}
    b = x .* μx/Δx .* (-xup .+ xdown) .- x.^2*ϕ^2*σx^2/Δx^2;
    b = b .+ (κ.*(λbar .- λ))/Δλ .* (-λup .+ λdown) .- σλ(λ).^2/Δλ^2;
    b = b .+ x.*σλ(λ)*ϕ*σx*ρ/(Δx*Δλ) .* (xup.*λup .- xup.*λdown .- xdown.*λup .+ xdown.*λdown);

    # Coefficient on V_{i+1,j}
    c = x .* μx/Δx .* xup .+ 0.5*x.^2*ϕ^2*σx^2/Δx^2;
    c = c .+ x.*σλ(λ)*ϕ*σx*ρ/(Δx*Δλ) .* (-xup.*λup .+ xup.*λdown);

    # Coefficient on V_{i-1,j-1}
    f = x.*σλ(λ)*ϕ*σx*ρ/(Δx*Δλ) .* xdown.*λdown;

    # Coefficient on V_{i,j-1}
    g = -(κ.*(λbar .- λ))/Δλ .* λdown .+ 0.5*σλ(λ).^2/Δλ^2;
    g = g .+ x.*σλ(λ)*ϕ*σx*ρ/(Δx*Δλ) .* (xup .* λdown .- xdown.*λdown);

    # Coefficient on V_{i+1,j-1}
    h = -x.*σλ(λ)*ϕ*σx*ρ/(Δx*Δλ) .* xup .* λdown;

    # Coefficient on V_{i-1,j+1}
    m = -x.*σλ(λ)*ϕ*σx*ρ/(Δx*Δλ) .* xdown .* λup;

    # Coefficient on V_{i,j+1}
    n = (κ.*(λbar .- λ))/Δλ .* λup .+ 0.5*σλ(λ).^2/Δλ^2;
    n = n .+ x.*σλ(λ)*ϕ*σx*ρ/(Δx*Δλ) .* (-xup .* λup .+ xdown .* λup);

    # Coefficient on V_{i+1,j+1}
    p = x.*σλ(λ)*ϕ*σx*ρ/(Δx*Δλ) .* xup .* λup;

    # Helper vectors to account for reflecting barrier at the edges of the state space
    lastx = zeros(I);
    lastx[end] = 1.0;
    lastx = repeat(lastx, outer=J);
    firstx = zeros(I);
    firstx[1] = 1.0;
    firstx = repeat(firstx, outer=J);
    lastλ = zeros(J);
    lastλ[end] = 1.0;
    lastλ = repeat(lastλ, inner=I);
    firstλ = zeros(J);
    firstλ[1] = 1.0;
    firstλ = repeat(firstλ, inner=I);

    # Main diagonal strip (staying at the same λ)
    maindiag_0 = b .+ firstx .* a .+ lastx .* c .+ firstλ .* g .+ lastλ .* n;
    maindiag_0 = maindiag_0 .+ firstx.*firstλ.*f .+ lastx.*firstλ.*h .+ firstx.*lastλ.*m .+ lastx.*lastλ.*p;
    maindiag_p1 = c .+ firstλ.*h .+ lastλ.*p;
    maindiag_p1 = maindiag_p1.*(1 .- lastx);
    maindiag_m1 = a .+ firstλ.*f .+ lastλ.*m;
    maindiag_m1 = maindiag_m1.*(1 .- firstx);

    # Upper diagonal strip (λ moves up)
    uppdiag_0 = n .+ firstx.* m .+ lastx.*p;
    uppdiag_0 = uppdiag_0.*(1 .- lastλ);
    uppdiag_p1 = (1 .- lastλ).*(1 .- lastx).*p;
    uppdiag_m1 = (1 .- lastλ).*(1 .- firstx).*m;

    # Lower diagonal strip (λ moves down)
    lowdiag_0 = g .+ firstx.*f .+ lastx.*h;
    lowdiag_0 = lowdiag_0.*(1 .- firstλ);
    lowdiag_m1 = (1 .- firstλ).*(1 .- firstx).*f;
    lowdiag_p1 = (1 .- firstλ).*(1 .- lastx).*h;

    # Filling the diagonals
    A = spdiagm(
        0 => maindiag_0,
        1 => maindiag_p1[1:(end-1)],
        -1 => maindiag_m1[2:end],
        I => uppdiag_0[1:(end-I)],
        I+1 => uppdiag_p1[1:(end-I-1)],
        I-1 => uppdiag_m1[1:(end-I+1)],
        -I => lowdiag_0[(I+1):end],
        -I+1 => lowdiag_p1[I:end],
        -I-1 => lowdiag_m1[(I+2):end]
    );

    return A
end


function exerciseValue(model_params)
    # Unpacking parameters
    K0, K1 = model_params.K0, model_params.K1;

    ASimple = formTransitionMatrixSimple(model_params);
    v = solveModelMature(model_params, ASimple; K = K1);
    
    # Value of exercising the investment option. Using it as a starting value
    Ve = model_params.x .* repeat(v, inner=model_params.I) .- (K1 - K0);

    return Ve
end


function solveModelYoung(model_params, A)
    # Unpacking parameters
    r, α, K0, K1 = model_params.r, model_params.α, model_params.K0, model_params.K1;

    # Value of exercising the investment option. Using it as a starting value
    Ve = exerciseValue(model_params)
    
    # Starting value: firm's value without growth opportuinities
    vNG = solveModelMature(model_params, AQSimple; K = model_params.K0);
    Vstart =  model_params.x .* repeat(vNG, inner=model_params.I);

    # Flow utility
    U = model_params.x * K0^α; 
    
    # Preparing for iteration
    Vprev = Vstart;
    Δt = model_params.Δt;

    # Iterating
    it, dist = 1, 1e8
    while (it < 1e8) & (dist > 1e-6)
        Vne = Vprev*(1-r*Δt) .+ (A*Δt)*Vprev;
        # Vnext = U*Δt .+ max.(Vne, Ve);
        
        Vnext = max.(U*Δt .+ Vne, Ve); # so that match to exercise value
        # Vnext = U*Δt .+ Vne;
        dist = maximum(abs.(Vnext - Vprev));
        if it % 1000 == 0
            @show dist
        end
        Vprev = Vnext;
        it += 1;
    end

    V = Vprev;

    return V
end


function calcExerciseBound(model_params, VY, Ve)
    exercise_bound = zeros(model_params.J);
    for j = 1:model_params.J
        ind_start = (j - 1)*model_params.I + 1;
        ind_end = j*model_params.I;
        exercise_bound[j] = sum(abs.(VY[ind_start:ind_end] .- Ve[ind_start:ind_end]) .> 1e-8);
    end

    return exercise_bound
end


function iterateModelForwardNoTransition(model_params, A, i, j, T; return_full=false)
    # Unpacking coefficients
    Δt, I, x = model_params.Δt, model_params.I, model_params.x;

    # Calculating forward transition matrix
    kfe_trans = A'*Δt + spdiagm(0 => ones(length(x)));

    # Point mass at the initial state s0
    s0 = (j-1)*I + i
    g = zeros(length(x));
    g[s0] = 1.0;

    num_periods = Int(T/Δt);
    gfull = zeros(length(x), num_periods);
    gfull[:, 1] = g;

    for it = 1:(num_periods - 1)
        gfull[:, it+1] = kfe_trans * gfull[:, it]
    end

    if return_full
        return gfull
    else
        return gfull[:, end]
    end
end


####################################################################
# Function to iterate the mode with real options forward
####################################################################

function iterateModelForwardGO(model_params, A, exercise_bound, VM, i, j, T)

    # Unpacking coefficients
    Δt, I, J, x = model_params.Δt, model_params.I, model_params.J, model_params.x;
    K0, K1 = model_params.K0, model_params.K1;
    IJ = I*J;

    # Calculating forward transition matrix
    kfe_trans = A'*Δt + spdiagm(0 => ones(IJ));

    # States where the firm exercises its growth option
    s_exercise = x .>= repeat(model_params.x_grid[exercise_bound], inner=I);
    s_exercise = SparseVector(s_exercise);

    # State probabilities for a firm that haven't exercised its GO
    s0 = (j-1)*I + i;
    gY = spzeros(IJ);
    gY[s0] = 1.0;

    # State probabilities for a firm that exercised its GO
    gM = spzeros(IJ);
    gM_adj = spzeros(IJ);

    # Share issuance at post issuance price
    η = (K1 - K0)./(VM .- (K1 - K0));

    # State for the dilution, make it somewhat smaller
    gη = spzeros(IJ*J); 

    num_periods = Int(T/Δt);

    @showprogress for i = 1:(num_periods - 1)

        # 1. Transition within a young and within mature firms
        gY = kfe_trans * gY;
        gM = kfe_trans * gM;

        # Transition for the dilution η
        for k = 1:J
            gη[(IJ*(k-1) + 1):(k*IJ)] = kfe_trans * gη[(IJ*(k-1) + 1):(k*IJ)]
        end

        # Inflow into mature firms
        gM = gM .+ s_exercise .* gY;
        gM_adj = gM_adj .+ s_exercise .* gY ./(1 .+ η);

        # Inflow into dilution
        for k = 1:J
            ind_of_exercise = (k - 1)*model_params.I  + exercise_bound[k]
            gη[IJ*(k-1) + ind_of_exercise] = gη[IJ*(k-1) + ind_of_exercise] .+ (s_exercise .* gY)[ind_of_exercise]
        end

        # Outflow from young
        gY = gY .- s_exercise .* gY;

        # Zeroing out very small values. Maybe it will run faster
        gY = gY .- gY .* (abs.(gY) .< 1e-24)
        gM = gM .- gM .* (abs.(gM) .< 1e-24)
        gη = gη .- gη .* (abs.(gη) .< 1e-24)

    end

    return gη, gY, gM #, g_full_adj #, EPVD

end


function calculateForwardGO(model_params, gη, gY, VM, VY, exercise_bound)
    K0, K1 = model_params.K0, model_params.K1;
    IJ = model_params.I*model_params.J;
    # Forward price is just the expected risk neutral value
    F = 0.0;
    for k = 1:model_params.J
        ind_for_exercise = (k - 1)*model_params.I  + exercise_bound[k];
        η = (K1 - K0)./(VM[ind_for_exercise] .- (K1 - K0));
        F += sum(gη[(IJ*(k-1) + 1):(k*IJ)] .* VM/(1 + η));
    end
    F += sum(gY .* VY)

    return F
end


function calculateCDFGO(model_params, gη, gY, F, exercise_bound, rel_price_list)
    K0, K1 = model_params.K0, model_params.K1;
    IJ = model_params.I*model_params.J
    CDF = zeros(length(rel_price_list));
    ind = 1
    for rel_price = rel_price_list
        Prob = 0.0;
        for k = 1:model_params.J
            ind_for_exercise = (k - 1)*model_params.I  + exercise_bound[k]
            η = (K1 - K0)./(VM[ind_for_exercise] .- (K1 - K0));
            Prob += sum(gη[(IJ*(k-1) + 1):(k*IJ)] .* (log.(VM/(1 + η)) .< rel_price .+ log.(F)));
        end
        Prob += sum(gY .* (log.(VY) .< rel_price .+ log.(F)))
        CDF[ind] = Prob
        ind += 1
    end

    return CDF
end

function calculateOptionPricesGO(model_params, strike, T, gη, gY, VM, VY, exercise_bound)
    K0, K1 = model_params.K0, model_params.K1;
    I, J = model_params.I, model_params.J;
    IJ = I*J;
    r = model_params.r;

    C, P = 0.0, 0.0
    for k = 1:J
        ind_for_exercise = (k - 1)*model_params.I  + exercise_bound[k];
        η = (K1 - K0)./(VM[ind_for_exercise] .- (K1 - K0));
        C += sum(gη[(IJ*(k-1) + 1):(k*IJ)] .* max.(VM/(1 + η) .- strike, 0.0));
        P += sum(gη[(IJ*(k-1) + 1):(k*IJ)] .* max.(strike .- VM/(1 + η), 0.0));
    end
    C += sum(gY .* max.(VY .- strike, 0.0))
    P += sum(gY .* max.(strike .- VY, 0.0))

    return exp(-r*T)*C, exp(-r*T)*P
end

function loadDistribution(dist_type, offset_type, i, j, full_size)
    filename = string("code/option_distributions/g", dist_type, "_", offset_type, "_", i, "_", j, ".csv")
    df = DataFrame(CSV.File(filename))
    g = sparsevec(df.index, df.value, full_size) 
    g = SparseVector{Float64, Int64}(g)   
    return g
end

function loadDistributionsBase(model_params, i, j)
    gη = loadDistribution("eta", "base", i, j, model_params.I * model_params.J^2);
    gM = loadDistribution("M", "base", i, j, model_params.I * model_params.J);
    gY = loadDistribution("Y", "base", i, j, model_params.I * model_params.J);
    return gη, gY, gM
end

function loadDistributionsOffset(model_params, i, j, offset)
    gη = loadDistribution("eta", offset, i, j, model_params.I * model_params.J^2);
    gM = loadDistribution("M", offset, i, j, model_params.I * model_params.J);
    gY = loadDistribution("Y", offset, i, j, model_params.I * model_params.J);
    return gη, gY, gM
end


function transformDistributionTo2D(model_params, g)
    I, J = model_params.I, model_params.J;
    g2D = zeros(I, J);

    for i = 1:I
        for j = 1:J
            g2D[i, j] = g[(j-1)*I + i]
        end
    end

    return g2D
end

####################################################################
# Functions for inverting the option prices
NCDF = x -> cdf(Distributions.Normal(), x);

function BSdpdm(F0, K, sigma, T, r)
    dp = (log(F0/K) + 0.5*sigma^2*T)/(sigma*sqrt(T))
    dm = dp - sigma*sqrt(T)
    return dp, dm
end

function BS_call_price(F0, K, sigma, T, r)
    dp, dm = BSdpdm(F0, K, sigma, T, r)
    return exp(-r*T)*(NCDF(dp)*F0 - NCDF(dm)*K)
end
    
function BS_put_price(F0, K, sigma, T, r)
    dp, dm = BSdpdm(F0, K, sigma, T, r)
    return exp(-r*T)*(NCDF(-dm)*K - NCDF(-dp)*F0)
end

function calculate_iv_put(opt_price, F0, K, T, r)
    function f!(func, x)
        func[1] = BS_put_price(F0, K, x[1], T, r) - opt_price
    end

    iv = nlsolve(f!, [0.4]).zero[1]
    return iv
end

function calculate_iv_call(opt_price, F0, K, T, r)
    function f!(func, x)
        func[1] = BS_call_price(F0, K, x[1], T, r) - opt_price
    end

    iv = nlsolve(f!, [0.4]).zero[1]
    return iv
end


########################################################
# Function to calculate direct sensitivity, requires having
# distributions calculated and saved in an appropriate folder
function calculateSensitivityGO(model_params, i_start, j_start, T, exercise_bound)
    # Calculates direct sensitivity of the option value that
    # is not coming though the forward price. Uses the previously
    # iterated models to calculate option prices. If the firm 
    # is above the exersice value, return missing values
    if i_start >= exercise_bound[j_start]
        return missing, missing, missing
    end

    Δx, Δλ = model_params.Δx, model_params.Δλ;

    # Loading base model to calculate forward price to use as a strike
    gη, gY, gM = loadDistributionsBase(model_params, i_start, j_start);
    F_base = calculateForwardGO(model_params, gη, gY, VM, VY, exercise_bound);

    # Loading densitieis at nearby strikes to calculate sensitivities
    gη_higherλ, gY_higherλ, gM_higherλ = loadDistributionsOffset(model_params, i_start, j_start, "hl");
    gη_lowerλ, gY_lowerλ, gM_lowerλ = loadDistributionsOffset(model_params, i_start, j_start, "ll");
    gη_higherx, gY_higherx, gM_higherx = loadDistributionsOffset(model_params, i_start, j_start, "hx");
    gη_lowerx, gY_lowerx, gM_lowerx = loadDistributionsOffset(model_params, i_start, j_start, "lx");

    # Calculating forward values
    F_higherλ = calculateForwardGO(model_params, gη_higherλ, gY_higherλ, VM, VY, exercise_bound);
    F_lowerλ = calculateForwardGO(model_params, gη_lowerλ, gY_lowerλ, VM, VY, exercise_bound);
    F_higherx = calculateForwardGO(model_params, gη_higherx, gY_higherx, VM, VY, exercise_bound);
    F_lowerx = calculateForwardGO(model_params, gη_lowerx, gY_lowerx, VM, VY, exercise_bound);

    # Sensitivity of forward value to (x, λ)
    dFdλ = (F_higherλ - F_lowerλ)/(2*Δλ)
    dFdx = (F_higherx - F_lowerx)/(2*Δx)

    # Calculating option sensitivity for strike = forward
    strike = F_base
    C_higherλ, P_higherλ = calculateOptionPricesGO(model_params, strike, T, gη_higherλ, gY_higherλ, VM, VY, exercise_bound)
    C_lowerλ, P_lowerλ = calculateOptionPricesGO(model_params, strike, T, gη_lowerλ, gY_lowerλ, VM, VY, exercise_bound)
    C_higherx, P_higherx = calculateOptionPricesGO(model_params, strike, T, gη_higherx, gY_higherx, VM, VY, exercise_bound)
    C_lowerx, P_lowerx = calculateOptionPricesGO(model_params, strike, T, gη_lowerx, gY_lowerx, VM, VY, exercise_bound)

    # Calculating sensitivities
    dPdλ = (P_higherλ - P_lowerλ)/(2*Δλ)
    dPdx = (P_higherx - P_lowerx)/(2*Δx)
    dCdλ = (C_higherλ - C_lowerλ)/(2*Δλ)
    dCdx = (C_higherx - C_lowerx)/(2*Δx)

    # Calculating partial sensitivities
    dCdλ_partial = dCdλ - dCdx/dFdx * dFdλ
    dPdλ_partial = dPdλ - dPdx/dFdx * dFdλ

    return dCdλ_partial, dPdλ_partial, F_base
end

####################################################################
# To get standard parameters compactly
function getParameters()
    # Grids for productivity and price of risk
    x_grid = 0.005:0.005:0.75; # Wider grid to account for exercising for low λ
    λ_grid = 0.0025:0.0025:0.25;

    I = length(x_grid);
    J = length(λ_grid);
    x = repeat(x_grid, outer=length(λ_grid));
    λ = repeat(λ_grid, inner=length(x_grid));
    Δx = x_grid[2] - x_grid[1];
    Δλ = λ_grid[2] - λ_grid[1];

    # combining with parameters
    bπ = λ -> 10*sqrt.(λ) # price of risk 
    σλ = λ -> 0.05*sqrt.(λ) # volatility of λ
    return (
        α = 0.65, r = 0.08, K1 = 5.0, K0 = 2.0, λbar = 0.05, bπ = bπ, σλ = σλ,
        μx = 0.03, σx = 0.1, ρ = -0.5, κ = 0.08, ϕ = 3.0,
        x_grid = x_grid, λ_grid = λ_grid, I = I, J = J,
        x = x, λ = λ, Δx = Δx, Δλ = Δλ, Δt = 0.0005, # Δt = 0.0002
    )
end


################################################################
# Solving for the value of firm with and without GO
################################################################

# Looking at the steady state distribution of λ
model_params = getParameters();
λshape = 2*model_params.κ*model_params.λbar/model_params.σλ(1.0)^2;
λscale = model_params.σλ(1.0)^2/(2*model_params.κ);
# Plots.plot(model_params.λ_grid, pdf.(Distributions.Gamma(λshape, λscale), model_params.λ_grid), label="")

# Iterating the mature model with the new distribution
model_params = getParameters();
AQSimple = formTransitionMatrixSimple(model_params);
vM = solveModelMature(model_params, AQSimple; K = model_params.K1);
VM = model_params.x .* repeat(vM, inner=model_params.I);
vNG = solveModelMature(model_params, AQSimple; K = model_params.K0);
VNG = model_params.x .* repeat(vNG, inner=model_params.I);
vMconst = solveModelConstPriceOfRisk(model_params; K = model_params.K1);

# Plotting the distribution and comparing to the constant price of risk model
# Plots.plot(model_params.λ_grid, vM, label="Time varying \$\\lambda\$", color=mathematicaColor1, lw=1.5, xlabel="Price of risk \$\\lambda\$", ylabel="Value of the firm \$V\$")
# Plots.plot!(model_params.λ_grid, vMconst, label="Constant \$\\lambda\$", color=mathematicaColor2, lw=1.5)

# Using the new grid to solve for firm with GO
AQ = formTransitionMatrix(model_params);
VY = solveModelYoung(model_params, AQ);
Ve = exerciseValue(model_params);

# Calculating the exercise bound
exercise_bound = calcExerciseBound(model_params, VY, Ve)
exercise_bound = Int.(exercise_bound)

# Plots.plot(model_params.λ_grid, model_params.x_grid[exercise_bound], lw=2.5, xlabel="Price of risk \$\\lambda\$", color=mathematicaColor1, label="")
# Plots.savefig("figures/options/exercise_boundary.pdf")

################################################################
# Showing the smooth pasting condition (together and step by
# step for the presentation)
################################################################

j = 20;
ind_start = (j - 1)*model_params.I + 1
ind_end = j*model_params.I
Plots.plot(model_params.x_grid, Ve[ind_start:ind_end], label="Value of exercising Real Options", xlabel="Productivity \$x\$", lw=2.5, legend=:topleft, linestyle=:dash, color=mathematicaColor2)
Plots.plot!(model_params.x_grid, VNG[ind_start:ind_end], label="Value without Real Options", lw=2.5, linestyle=:dash, color=mathematicaColor3)
Plots.plot!(model_params.x_grid, VY[ind_start:ind_end], label="Value with Real Options", lw=2.5, color=mathematicaColor1)
Plots.savefig("figures/options/young_firm_smooth_pasting.pdf")

# Doing the animation for the smooth pasting condition
plot1 = Plots.plot(model_params.x_grid, VNG[ind_start:ind_end], label="Value without Real Options", lw=2.5, linestyle=:dash, color=mathematicaColor3, legend=:topleft, xlabel="Productivity state \$x\$");
Plots.xlims!(0.0, 0.75);
Plots.ylims!(-4.0, 16.0);
Plots.plot(plot1, legendfontsize=10, guidefontsize=12)
Plots.savefig("figures/options/young_firm_smooth_pasting_1.pdf")

plot1 = Plots.plot(model_params.x_grid, VNG[ind_start:ind_end], label="Value without Real Options", lw=2.5, linestyle=:dash, color=mathematicaColor3);
Plots.plot!(model_params.x_grid, Ve[ind_start:ind_end], label="Value of exercising Real Options", xlabel="Productivity \$x\$", lw=2.5, legend=:topleft, linestyle=:dash, color=mathematicaColor2);
Plots.xlims!(0.0, 0.75);
Plots.ylims!(-4.0, 16.0);
Plots.plot(plot1, legendfontsize=10, guidefontsize=12)
Plots.savefig("figures/options/young_firm_smooth_pasting_2.pdf")

plot1 = Plots.plot(model_params.x_grid, VNG[ind_start:ind_end], label="Value without Real Options", lw=2.5, linestyle=:dash, color=mathematicaColor3);
Plots.plot!(model_params.x_grid, Ve[ind_start:ind_end], label="Value of exercising Real Options", xlabel="Productivity \$x\$", lw=2.5, legend=:topleft, linestyle=:dash, color=mathematicaColor2);
Plots.plot!(model_params.x_grid, VY[ind_start:ind_end], label="Value with Real Options", lw=2.5, color=mathematicaColor1);
Plots.xlims!(0.0, 0.75);
Plots.ylims!(-4.0, 16.0);
Plots.plot(plot1, legendfontsize=10, guidefontsize=12)
Plots.savefig("figures/options/young_firm_smooth_pasting_3.pdf")


####################################################################
# Plotting the CDF of firm value relative to forward for different
# initial states. (Under risk-neutral measure)
####################################################################

# Comparing CDF between GO and NGO firms
# i_start, j_start, T = 80, 20, 0.25;
i_start, j_start, T = 80, 10, 0.25;
gNG = iterateModelForwardNoTransition(model_params, AQ, i_start, j_start, T);
FNG = sum(gNG .* VNG)
rel_price_list = (-0.6):0.025:0.5;
CDF_NGO_lowλ = zeros(length(rel_price_list));
for ind_rel_price = 1:length(rel_price_list)
    CDF_NGO_lowλ[ind_rel_price] = sum(gNG .* (log.(VNG) .< rel_price_list[ind_rel_price] + log(FNG)))
end

i_start, j_start, T = 80, 40, 0.25;
gNG = iterateModelForwardNoTransition(model_params, AQ, i_start, j_start, T);
FNG = sum(gNG .* VNG)
rel_price_list = (-0.6):0.025:0.5;
CDF_NGO_highλ = zeros(length(rel_price_list));
for ind_rel_price = 1:length(rel_price_list)
    CDF_NGO_highλ[ind_rel_price] = sum(gNG .* (log.(VNG) .< rel_price_list[ind_rel_price] + log(FNG)))
end

i_start, j_start, T = 80, 10, 0.25;
gη, gY, gM = loadDistributionsBase(model_params, i_start, j_start);
F = calculateForwardGO(model_params, gη, gY, VM, VY, exercise_bound);
CDF_GO_lowλ = calculateCDFGO(model_params, gη, gY, F, exercise_bound, rel_price_list);

i_start, j_start, T = 80, 40, 0.25;
gη, gY, gM = loadDistributionsBase(model_params, i_start, j_start);
F = calculateForwardGO(model_params, gη, gY, VM, VY, exercise_bound);
CDF_GO_highλ = calculateCDFGO(model_params, gη, gY, F, exercise_bound, rel_price_list);

plot1 = Plots.plot(rel_price_list, CDF_NGO_lowλ, xlabel="\$\\log(V_{t+\\tau}/F_t)\$", label="Without GO (low \$\\lambda\$)", color=mathematicaColor1, lw=2.0);
Plots.plot!(rel_price_list, CDF_NGO_highλ, label="Without GO (high \$\\lambda\$)", legend=:topleft, color=mathematicaColor1, linestyle=:dash, lw=2.0);
Plots.plot!(rel_price_list, CDF_GO_lowλ, label="With GO (low \$\\lambda\$)", color=mathematicaColor2, lw=2.0);
Plots.plot!(rel_price_list, CDF_GO_highλ, label="With GO (high \$\\lambda\$)", color=mathematicaColor3, lw=2.0);
Plots.plot(plot1)
# Plots.savefig("figures/options/compare_CDF_for_NGO_vs_GO.pdf")

####################################################################
# Plotting the PDF of firm value relative to forward for different
# initial states. (Under risk-neutral measure)
####################################################################

# Transforming into a PDF
drel_price = rel_price_list[2] - rel_price_list[1];

PDF_NGO_lowλ = zeros(length(rel_price_list) - 2);
PDF_NGO_highλ = zeros(length(rel_price_list) - 2);
PDF_GO_lowλ = zeros(length(rel_price_list) - 2);
PDF_GO_highλ = zeros(length(rel_price_list) - 2);
for ind = 2:(length(rel_price_list) - 1)
    PDF_NGO_lowλ[ind-1] = (CDF_NGO_lowλ[ind+1] - CDF_NGO_lowλ[ind-1])/(2*drel_price)
    PDF_NGO_highλ[ind-1] = (CDF_NGO_highλ[ind+1] - CDF_NGO_highλ[ind-1])/(2*drel_price)
    PDF_GO_lowλ[ind-1] = (CDF_GO_lowλ[ind+1] - CDF_GO_lowλ[ind-1])/(2*drel_price)
    PDF_GO_highλ[ind-1] = (CDF_GO_highλ[ind+1] - CDF_GO_highλ[ind-1])/(2*drel_price)
end

# Plotting distributions all at once
plot1 = Plots.plot(rel_price_list[2:(end-1)], PDF_NGO_lowλ, xlabel="\$\\log(V_{t+\\tau}/F_t)\$", label="No Real Options(low \$\\lambda\$)", legend=:topleft, color=mathematicaColor1, lw=2.0);
Plots.plot!(rel_price_list[2:(end-1)], PDF_NGO_highλ, label="No Real Options(low \$\\lambda\$)", color=mathematicaColor1, linestyle = :dash, lw=2.0);
Plots.plot!(rel_price_list[2:(end-1)], PDF_GO_lowλ, label="Real Options (low \$\\lambda\$)", color=mathematicaColor2, lw=2.0);
Plots.plot!(rel_price_list[2:(end-1)], PDF_GO_highλ, label="Real Options (high \$\\lambda\$)", color=mathematicaColor3, lw=2.0);
Plots.vline!([0.0], color="black", linestyle=:dash, label="");
Plots.plot(plot1)
# Plots.savefig("figures/options/compare_PDF_for_NGO_vs_GO.pdf")

# Plotting the same distributions step by step
plot1 = Plots.plot(rel_price_list[2:(end-1)], PDF_NGO_highλ, xlabel="\$\\log(V_{t+\\tau}/F_t)\$", label="No Real Options(high \$\\lambda\$)", legend=:topleft, color=mathematicaColor3, lw=3.0);
Plots.vline!([0.0], color="black", linestyle=:dash, label="");
Plots.ylims!(0.0, 2.55);
Plots.plot(plot1, legendfontsize=9, guidefontsize=12)
Plots.savefig("figures/options/compare_PDF_for_NGO_vs_GO_1.pdf")

plot1 = Plots.plot(rel_price_list[2:(end-1)], PDF_NGO_highλ, xlabel="\$\\log(V_{t+\\tau}/F_t)\$", label="No Real Options(high \$\\lambda\$)", legend=:topleft, color=mathematicaColor3, lw=3.0);
Plots.plot!(rel_price_list[2:(end-1)], PDF_NGO_lowλ, label="No Real Options(low \$\\lambda\$)", color=mathematicaColor3, linestyle = :dash, lw=3.0);
Plots.vline!([0.0], color="black", linestyle=:dash, label="");
Plots.ylims!(0.0, 2.55);
Plots.plot(plot1, legendfontsize=9, guidefontsize=12)
Plots.savefig("figures/options/compare_PDF_for_NGO_vs_GO_2.pdf")

plot1 = Plots.plot(rel_price_list[2:(end-1)], PDF_NGO_highλ, xlabel="\$\\log(V_{t+\\tau}/F_t)\$", label="No Real Options(high \$\\lambda\$)", legend=:topleft, color=mathematicaColor3, lw=3.0);
Plots.plot!(rel_price_list[2:(end-1)], PDF_NGO_lowλ, label="No Real Options(low \$\\lambda\$)", color=mathematicaColor3, linestyle = :dash, lw=3.0);
Plots.plot!(rel_price_list[2:(end-1)], PDF_GO_highλ, label="Real Options (high \$\\lambda\$)", color=mathematicaColor2, lw=3.0);
Plots.vline!([0.0], color="black", linestyle=:dash, label="");
Plots.ylims!(0.0, 2.55);
Plots.plot(plot1, legendfontsize=9, guidefontsize=12)
Plots.savefig("figures/options/compare_PDF_for_NGO_vs_GO_3.pdf")

plot1 = Plots.plot(rel_price_list[2:(end-1)], PDF_NGO_highλ, xlabel="\$\\log(V_{t+\\tau}/F_t)\$", label="No Real Options(high \$\\lambda\$)", legend=:topleft, color=mathematicaColor3, lw=3.0);
Plots.plot!(rel_price_list[2:(end-1)], PDF_NGO_lowλ, label="No Real Options(low \$\\lambda\$)", color=mathematicaColor3, linestyle = :dash, lw=3.0);
Plots.plot!(rel_price_list[2:(end-1)], PDF_GO_highλ, label="Real Options (high \$\\lambda\$)", color=mathematicaColor2, lw=3.0);
Plots.plot!(rel_price_list[2:(end-1)], PDF_GO_lowλ, label="Real Options (low \$\\lambda\$)", color=mathematicaColor1, lw=3.0);
Plots.vline!([0.0], color="black", linestyle=:dash, label="");
Plots.ylims!(0.0, 2.55);
Plots.plot(plot1, legendfontsize=9, guidefontsize=12)
Plots.savefig("figures/options/compare_PDF_for_NGO_vs_GO_4.pdf")

####################################################################
# Calculating CDF and PDF for firm value relative to forward
# under the physical measure
####################################################################

AQ = formTransitionMatrix(model_params);
AP = formTransitionMatrixPhysical(model_params);

# 1. Firm with no real options (BOOM)
i_start, j_start, T = 80, 10, 0.25;
gNG = iterateModelForwardNoTransition(model_params, AQ, i_start, j_start, T);
FNG = sum(gNG .* VNG);

gNG = iterateModelForwardNoTransition(model_params, AP, i_start, j_start, T);
rel_price_list = (-0.6):0.025:0.5;
CDF_NGO_lowλ = zeros(length(rel_price_list));
for ind_rel_price = 1:length(rel_price_list)
    CDF_NGO_lowλ[ind_rel_price] = sum(gNG .* (log.(VNG) .< rel_price_list[ind_rel_price] + log(FNG)))
end

# 2. Firm with no real options (BUST)
i_start, j_start, T = 80, 40, 0.25;
gNG = iterateModelForwardNoTransition(model_params, AQ, i_start, j_start, T);
FNG = sum(gNG .* VNG)

gNG = iterateModelForwardNoTransition(model_params, AP, i_start, j_start, T);
rel_price_list = (-0.6):0.025:0.5;
CDF_NGO_highλ = zeros(length(rel_price_list));
for ind_rel_price = 1:length(rel_price_list)
    CDF_NGO_highλ[ind_rel_price] = sum(gNG .* (log.(VNG) .< rel_price_list[ind_rel_price] + log(FNG)))
end

# 3. Firm with real options (BOOM)
i_start, j_start, T = 80, 10, 0.25;
gη, gY, gM = iterateModelForwardGO(model_params, AQ, exercise_bound, VM, i_start, j_start, T);
F = calculateForwardGO(model_params, gη, gY, VM, VY, exercise_bound);

gη, gY, gM = iterateModelForwardGO(model_params, AP, exercise_bound, VM, i_start, j_start, T);
CDF_GO_lowλ = calculateCDFGO(model_params, gη, gY, F, exercise_bound, rel_price_list);

# 4. Firm with real options (BUST)
i_start, j_start, T = 80, 40, 0.25;
gη, gY, gM = iterateModelForwardGO(model_params, AQ, exercise_bound, VM, i_start, j_start, T);
F = calculateForwardGO(model_params, gη, gY, VM, VY, exercise_bound);

gη, gY, gM = iterateModelForwardGO(model_params, AP, exercise_bound, VM, i_start, j_start, T);
CDF_GO_highλ = calculateCDFGO(model_params, gη, gY, F, exercise_bound, rel_price_list);


# Plotting CDF
plot1 = Plots.plot(rel_price_list, CDF_NGO_lowλ, xlabel="\$\\log(V_{t+\\tau}/F_t)\$", label="Without GO (low \$\\lambda\$)", color=mathematicaColor1, lw=2.0);
Plots.plot!(rel_price_list, CDF_NGO_highλ, label="Without GO (high \$\\lambda\$)", legend=:topleft, color=mathematicaColor1, linestyle=:dash, lw=2.0);
Plots.plot!(rel_price_list, CDF_GO_lowλ, label="With GO (low \$\\lambda\$)", color=mathematicaColor2, lw=2.0);
Plots.plot!(rel_price_list, CDF_GO_highλ, label="With GO (high \$\\lambda\$)", color=mathematicaColor3, lw=2.0);
Plots.plot(plot1)

# Transforming into a PDF
drel_price = rel_price_list[2] - rel_price_list[1];

PDF_NGO_lowλ = zeros(length(rel_price_list) - 2);
PDF_NGO_highλ = zeros(length(rel_price_list) - 2);
PDF_GO_lowλ = zeros(length(rel_price_list) - 2);
PDF_GO_highλ = zeros(length(rel_price_list) - 2);
for ind = 2:(length(rel_price_list) - 1)
    PDF_NGO_lowλ[ind-1] = (CDF_NGO_lowλ[ind+1] - CDF_NGO_lowλ[ind-1])/(2*drel_price)
    PDF_NGO_highλ[ind-1] = (CDF_NGO_highλ[ind+1] - CDF_NGO_highλ[ind-1])/(2*drel_price)
    PDF_GO_lowλ[ind-1] = (CDF_GO_lowλ[ind+1] - CDF_GO_lowλ[ind-1])/(2*drel_price)
    PDF_GO_highλ[ind-1] = (CDF_GO_highλ[ind+1] - CDF_GO_highλ[ind-1])/(2*drel_price)
end


plot1 = Plots.plot(rel_price_list[2:(end-1)], PDF_NGO_lowλ, xlabel="\$\\log(V_{t+\\tau}/F_t)\$", label="No Real Options(low \$\\lambda\$)", legend=:topleft, color=mathematicaColor1, lw=2.0);
Plots.plot!(rel_price_list[2:(end-1)], PDF_NGO_highλ, label="No Real Options(low \$\\lambda\$)", color=mathematicaColor1, linestyle = :dash, lw=2.0);
Plots.plot!(rel_price_list[2:(end-1)], PDF_GO_lowλ, label="Real Options (low \$\\lambda\$)", color=mathematicaColor2, lw=2.0);
Plots.plot!(rel_price_list[2:(end-1)], PDF_GO_highλ, label="Real Options (high \$\\lambda\$)", color=mathematicaColor3, lw=2.0);
Plots.vline!([0.0], color="black", linestyle=:dash, label="");
Plots.plot(plot1)
Plots.savefig("figures/options/compare_PDF_for_NGO_vs_GO_under_physical.pdf")

####################################################################
# Loading distributions of firm values for a set of starting 
# states to compare option prices across the states
####################################################################

λ_list_to_est = Array(5:5:50);
x_list_to_est = Array(20:5:110);

# ################################################################
# # 1. Looking at change in CDF as λ changes
# rel_price_list = (-0.6):0.025:0.5;
# CDF_mat = zeros((length(rel_price_list), length(λ_list_to_est)));

# for (ind, j_start) in enumerate(λ_list_to_est)
#     gη, gY, gM = loadDistributionsBase(model_params, 80, j_start);
#     F = calculateForwardGO(model_params, gη, gY, VM, VY, exercise_bound);
#     CDF_mat[:, ind] = calculateCDFGO(model_params, gη, gY, F, exercise_bound, rel_price_list);
# end

# colors = Plots.cgrad(:viridis, 6, categorical = true);
# plot1 = Plots.vline([0.0], color="black", linestyle=:dash, label="");
# Plots.hline!([0.0], color="black", linestyle=:dash, label="");
# for (iplot, iλ) in enumerate([2; 4; 6; 8; 10])
#     CDF_diff = CDF_mat[:, iλ] - CDF_mat[:, 1];
#     λ_current = model_params.λ_grid[λ_list_to_est[iλ]];
#     Plots.plot!(rel_price_list, CDF_diff, label=string("\$\\lambda = $λ_current \$"), color=colors[iplot], lw=2.0);
# end
# Plots.ylims!((-0.075, 0.075));
# Plots.xlims!((-0.6, 0.65));
# Plots.plot(plot1)

# ################################################################
# # 2. Looking at change in CDF as x changes
# rel_price_list = (-0.6):0.025:0.5;
# CDF_mat = zeros((length(rel_price_list), length(20:5:90)));

# for (ind, i_start) in enumerate(20:5:90) # need to stop before the transitio, otherwise have some error.
#     @show i_start
#     gη, gY, gM = loadDistributionsBase(model_params, i_start, 20);
#     F = calculateForwardGO(model_params, gη, gY, VM, VY, exercise_bound);
#     CDF_mat[:, ind] = calculateCDFGO(model_params, gη, gY, F, exercise_bound, rel_price_list);
# end

# colors = Plots.cgrad(:viridis, 7, categorical = true);
# plot1 = Plots.vline([0.0], color="black", linestyle=:dash, label="");
# Plots.hline!([0.0], color="black", linestyle=:dash, label="");
# for (iplot, ix) in enumerate([2; 5; 10; 12; 13; 14; 15])
#     CDF_diff = CDF_mat[:, ix] - CDF_mat[:, 1];
#     x_current = model_params.x_grid[x_list_to_est[ix]];
#     Plots.plot!(rel_price_list, CDF_diff, label=string("\$x = $x_current\$"), color=colors[iplot], lw=2.0);
# end
# Plots.ylims!((-0.075, 0.075));
# Plots.xlims!((-0.6, 0.5));
# Plots.xlabel!("\$\\log(V_T/F)\$");
# x_base = model_params.x_grid[x_list_to_est[1]];
# Plots.ylabel!("\$\\Delta CDF\$ relative to \$x=$(x_base)\$");
# Plots.plot(plot1)

################################################################
# 3. Looking at change in IV as λ changes
T = 0.25;
λ_list_to_est = Array(5:5:50);
x_list_to_est = Array(20:5:110);

rel_strike_list = 0.6:0.025:1.4;
C_mat = zeros((length(rel_strike_list), length(λ_list_to_est)));
CIV_mat = zeros((length(rel_strike_list), length(λ_list_to_est)));

# Calculating call prices
@showprogress for (istate, j_start) in enumerate(λ_list_to_est)
    gη, gY, gM = loadDistributionsBase(model_params, 80, j_start);
    F = calculateForwardGO(model_params, gη, gY, VM, VY, exercise_bound);
    for (istrike, rel_strike) in enumerate(rel_strike_list)
        strike = rel_strike_list[istrike]*F;
        C_mat[istrike, istate], tmp = calculateOptionPricesGO(model_params, strike, T, gη, gY, VM, VY, exercise_bound)
        CIV_mat[istrike, istate] = calculate_iv_call(C_mat[istrike, istate], F, strike, T, model_params.r)
    end
end

λ_base = model_params.λ_grid[λ_list_to_est[1]]
colors = Plots.cgrad(:viridis, 6, categorical = true);
plot1 = Plots.vline([0.0], color="black", linestyle=:dash, label="");
Plots.hline!([0.0], color="black", linestyle=:dash, label="");
for (iplot, iλ) in enumerate([2; 4; 6; 8; 10])
    CIV_diff = CIV_mat[:, iλ] - CIV_mat[:, 1];
    λ_current = model_params.λ_grid[λ_list_to_est[iλ]];
    Plots.plot!(rel_strike_list, CIV_diff, label=string("\$\\lambda = $λ_current \$"), color=colors[iplot], lw=2.0);
end
Plots.ylims!((-0.075, 0.15));
Plots.xlims!((0.6, 1.4));
Plots.xlabel!("\$Strike/Forward\$");
Plots.ylabel!("\$\\Delta IV\$ relative to \$\\lambda=$(λ_base)\$");
Plots.plot(plot1)
# Plots.savefig("figures/options/sensitivity_of_iv_to_lambda.pdf")


################################################################
# 4. Comparing Implied volatilities with the NGO model

rel_strike_list = 0.6:0.025:1.4;
C_NGO_mat = zeros((length(rel_strike_list), length(λ_list_to_est)));
CIV_NGO_mat = zeros((length(rel_strike_list), length(λ_list_to_est)));

i_start = 80;
T = 0.25;

@showprogress for (istate, j_start) in enumerate(λ_list_to_est)
    s_start = (j_start - 1)*model_params.I + i_start;

    # 1. Solving the model forward
    g = iterateModelForwardNoTransition(model_params, AQ, i_start, j_start, T; return_full=false);

    # 2. Calculating the forward price following Duffie (2002)
    D = model_params.x .* model_params.K0^model_params.α;
    FNG = sum(g .* VNG)

    # Calculating option prices
    for irel_strike = 1:length(rel_strike_list)
        strike = rel_strike_list[irel_strike] * FNG;
        C = exp.(-model_params.r*T)*sum(g .* max.(VNG .- strike, 0.0))
        CIV = calculate_iv_call(C, FNG, strike, T, model_params.r)
        C_NGO_mat[irel_strike, istate] = C
        CIV_NGO_mat[irel_strike, istate] = CIV
    end
end


plot1 = Plots.plot(log.(rel_strike_list), CIV_NGO_mat[:, 1], label="Without GO (low \$\\lambda\$)", color=mathematicaColor1, lw=2.0, legend=:right, xlabel="\$\\log(V_{t+\\tau}/F_t)\$", ylabel="Implied Volatility")
Plots.plot!(log.(rel_strike_list), CIV_NGO_mat[:, 10], label="Without GO (high \$\\lambda\$)", color=mathematicaColor1, linestyle=:dash, lw=2.0)
Plots.plot!(log.(rel_strike_list), CIV_mat[:, 1], label="With GO (low \$\\lambda\$)", color=mathematicaColor2, lw=2.0);
Plots.plot!(log.(rel_strike_list), CIV_mat[:, 10], label="With GO (high \$\\lambda\$)", color=mathematicaColor3, lw=2.0);
Plots.vline!([0.0], linestyle=:dash, color="black", label="");
Plots.plot(plot1, legendfontsize=10, guidefontsize=12)
# Plots.savefig("figures/options/iv_for_low_and_high_lambda.pdf")

# Showing these step by step
plot1 = Plots.plot(log.(rel_strike_list), CIV_NGO_mat[:, 10], label="No Real Options (high \$\\lambda\$)", color=mathematicaColor3, lw = 3.0, legend=:topleft, xlabel="\$\\log(V_{t+\\tau}/F_t)\$");
Plots.vline!([0.0], linestyle=:dash, color="black", label="");
Plots.ylims!(0.30, 0.45);
Plots.plot(plot1, legendfontsize=9, guidefontsize=12)
Plots.savefig("figures/options/iv_for_low_and_high_lambda_1.pdf")


plot1 = Plots.plot(log.(rel_strike_list), CIV_NGO_mat[:, 10], label="No Real Options (high \$\\lambda\$)", color=mathematicaColor3, lw = 3.0, legend=:topleft, xlabel="\$\\log(V_{t+\\tau}/F_t)\$");
Plots.plot!(log.(rel_strike_list), CIV_NGO_mat[:, 1], label="No Real Options (low \$\\lambda\$)", color=mathematicaColor3, linestyle=:dash, lw = 3.0);
Plots.vline!([0.0], linestyle=:dash, color="black", label="");
Plots.ylims!(0.30, 0.45);
Plots.plot(plot1, legendfontsize=9, guidefontsize=12)
Plots.savefig("figures/options/iv_for_low_and_high_lambda_2.pdf")


plot1 = Plots.plot(log.(rel_strike_list), CIV_NGO_mat[:, 10], label="No Real Options (high \$\\lambda\$)", color=mathematicaColor3, lw = 3.0, legend=:topleft, xlabel="\$\\log(V_{t+\\tau}/F_t)\$");
Plots.plot!(log.(rel_strike_list), CIV_NGO_mat[:, 1], label="No Real Options (low \$\\lambda\$)", color=mathematicaColor3, linestyle=:dash, lw = 3.0);
Plots.plot!(log.(rel_strike_list), CIV_mat[:, 10], label="Real Options (high \$\\lambda\$)", color=mathematicaColor2, lw = 3.0);
Plots.vline!([0.0], linestyle=:dash, color="black", label="");
Plots.ylims!(0.30, 0.45);
Plots.plot(plot1, legendfontsize=9, guidefontsize=12)
Plots.savefig("figures/options/iv_for_low_and_high_lambda_3.pdf")


plot1 = Plots.plot(log.(rel_strike_list), CIV_NGO_mat[:, 10], label="No Real Options (high \$\\lambda\$)", color=mathematicaColor3, lw = 3.0, legend=:topleft, xlabel="\$\\log(V_{t+\\tau}/F_t)\$");
Plots.plot!(log.(rel_strike_list), CIV_NGO_mat[:, 1], label="No Real Options (low \$\\lambda\$)", color=mathematicaColor3, linestyle=:dash, lw = 3.0);
Plots.plot!(log.(rel_strike_list), CIV_mat[:, 10], label="Real Options (high \$\\lambda\$)", color=mathematicaColor2, lw = 3.0);
Plots.plot!(log.(rel_strike_list), CIV_mat[:, 1], label="Real Options (low \$\\lambda\$)", color=mathematicaColor1, lw = 3.0);
Plots.vline!([0.0], linestyle=:dash, color="black", label="");
Plots.ylims!(0.30, 0.45);
Plots.plot(plot1, legendfontsize=9, guidefontsize=12)
Plots.savefig("figures/options/iv_for_low_and_high_lambda_4.pdf")

####################################################################
# Partial sensitivity of Calls and Puts to λ -- the main unknown
# component in the expected profits of a delta hedged strategy
####################################################################

T = 0.25;

λ_list_to_est = Array(5:5:50);
x_list_to_est = [50; 55; 60; 65; 70; 71; 72; 73; 74; 75; 76; 77; 78; 79; 80];

# Getting the full list of variables
state_list = [];
for x = x_list_to_est
    for λ = λ_list_to_est
        push!(state_list, (x, λ))
    end
end

sensitivity_list_GO = @showprogress map(state_list) do state
    i_start, j_start = state
    out = calculateSensitivityGO(model_params, i_start, j_start, T, exercise_bound)
    (i_start, j_start, out)
end

sensitivity_df = DataFrame(
    :i_start => map(x -> x[1], sensitivity_list_GO),
    :j_start => map(x -> x[2], sensitivity_list_GO),
    :dCdλ => map(x -> x[3][1], sensitivity_list_GO),
    :F => map(x -> x[3][3], sensitivity_list_GO)
)

# Plotting sensitivity across the price of risk λ for different x
nx = length(unique(sensitivity_df.i_start));
plot1 = Plots.plot(
    legend=:bottomright, xlabel="Price of risk \$\\lambda\$", 
    ylabel="Partial sensitivity of \$C\$ to \$\\lambda\$",
    background=RGBA(1.0,1.0,1.0,1));
    
# Adding NGO firm sensitivity
Plots.plot!(model_params.λ_grid[5:5:30], map(x -> x[1]/x[3], sensitivity_list_NGO), label="\$No \\ GO\$", color="black", lw=5.0);
    
colors = Plots.cgrad(:viridis, nx, categorical = true);

for ix = 1:nx
    x = model_params.x_grid[x_list_to_est[ix]]
    sub_df = sensitivity_df[sensitivity_df.i_start .== x_list_to_est[ix], :];
    if x == 0.25
        label = "\$x=$x\$"
    elseif x == 0.4
        label = "\$x=$x\$"
    elseif x in [0.3; 0.35]
        label = "\$\\vdots\$"
    else
        label = ""
    end
    Plots.plot!(
        model_params.λ_grid[sub_df.j_start], sub_df.dCdλ./sub_df.F, 
        color=colors[ix], label=label, lw=2.5);
end

Plots.hline!([0.0], color="black", linestyle=:dash, label="")

Plots.plot(plot1)
# Plots.savefig("figures/options/partial_sensitivity_GO_vs_NGO.pdf")

########################################################################
# Multiplying the direct sensitivity by the price of risk λ

# Plotting sensitivity across the price of risk λ for different x
nx = length(unique(sensitivity_df.i_start));

plot1 = Plots.plot(
    legend=:bottomright, xlabel="Price of risk \$\\lambda\$", 
    ylabel="Partial sensitivity of \$C\$ to \$\\lambda\$",
    background=RGBA(1.0,1.0,1.0,1));
    
# Adding NGO firm sensitivity
Plots.plot!(model_params.λ_grid[5:5:30], model_params.λ_grid[5:5:30] .* map(x -> x[1]/x[3], sensitivity_list_NGO), label="\$No \\ GO\$", color="black", lw=5.0);
    
colors = Plots.cgrad(:viridis, nx, categorical = true);

for ix = 1:nx
    x = model_params.x_grid[x_list_to_est[ix]]
    sub_df = sensitivity_df[sensitivity_df.i_start .== x_list_to_est[ix], :];
    if x == 0.25
        label = "\$x=$x\$"
    elseif x == 0.4
        label = "\$x=$x\$"
    elseif x in [0.3; 0.35]
        label = "\$\\vdots\$"
    else
        label = ""
    end
    Plots.plot!(
        model_params.λ_grid[sub_df.j_start], model_params.λ_grid[sub_df.j_start] .* sub_df.dCdλ./sub_df.F, 
        color=colors[ix], label=label, lw=2.5);
end

Plots.hline!([0.0], color="black", linestyle=:dash, label="");

Plots.plot(plot1)
# Plots.savefig("figures/options/partial_sensitivity_GO_vs_NGO_mult_by_lambda.pdf")




