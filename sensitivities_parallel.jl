################################################################
# Author: Roman Sigalov
#
# Iterating the Kolmogorov Forward Equation to calculate the
# terminal state of firm's (diluted) value.
################################################################

################################################################
# Dealing with import libraries to work in parallel
################################################################

using Distributed

@everywhere using Pkg
@everywhere Pkg.activate(".")

using Distributed

print("\nNumber of processors ")
print(nprocs())
print("\n")

@everywhere using SparseArrays # To store and multiply spares matrices efficiently

using ProgressMeter
using CSV
using Dates
using DataFrames

################################################################
# Functions
################################################################

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


# Forming a transition matrix under the physical probability
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
    a = -x .* (μx)/Δx .* xdown .+ 0.5*x.^2*ϕ^2*σx^2/Δx^2;
    a = a .+ x.*σλ(λ)*ϕ*σx*ρ/(Δx*Δλ) .* (xdown .* λup - xdown .* λdown);

    # Coefficient on V_{i,j}
    b = x .* (μx)/Δx .* (-xup .+ xdown) .- x.^2*ϕ^2*σx^2/Δx^2;
    b = b .+ (κ.*(λbar .- λ))/Δλ .* (-λup .+ λdown) .- σλ(λ).^2/Δλ^2;
    b = b .+ x.*σλ(λ)*ϕ*σx*ρ/(Δx*Δλ) .* (xup.*λup .- xup.*λdown .- xdown.*λup .+ xdown.*λdown);

    # Coefficient on V_{i+1,j}
    c = x .* (μx)/Δx .* xup .+ 0.5*x.^2*ϕ^2*σx^2/Δx^2;
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


function exerciseValue(model_params)
    # Unpacking parameters
    K0, K1 = model_params.K0, model_params.K1;

    ASimple = formTransitionMatrixSimple(model_params);
    v = solveModelMature(model_params, ASimple; K = K1);
    
    # Value of exercising the investment option. Using it as a starting value
    Ve = model_params.x .* repeat(v, inner=model_params.I) .- (K1 - K0);

    return Ve
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


@everywhere function iterateModelForwardGO(model_params, A, exercise_bound, VM, i, j, T)

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

    for i = 1:(num_periods - 1)

        # 1. Transition within a young and within mature firms
        gY = kfe_trans * gY;
        gM = kfe_trans * gM;

        # Transition for the dilution η
        for k = 1:J
            gη[(IJ*(k-1) + 1):(k*IJ)] = kfe_trans * gη[(IJ*(k-1) + 1):(k*IJ)]
        end

        # Inflow into mature firms
        gM = gM .+ s_exercise .* gY;

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

    return gη, gY, gM
end


function saveg(g, filename)
    df_to_save = DataFrame(:index => findnz(g)[1], :value => findnz(g)[2]);
    CSV.write(filename, df_to_save);
end


function getParameters()
    # Grids
    # x_grid = 0.005:0.005:0.5;
    x_grid = 0.005:0.005:1.0; # Wider grid to account for exercising for low λ
    λ_grid = 0.001:0.001:0.25;

    # Trying sparser grids, do they work? Seems like it
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


####################################################################
# Doing preliminary work before solving solving the model forward
# (need to arrive at solution for mature model VM and young VY)
####################################################################

print(string("\n--- Solving for the value of the firm  ---\n"))

model_params = getParameters();

# Iterating the mature model with the new distribution
AQSimple = formTransitionMatrixSimple(model_params);
vM = solveModelMature(model_params, AQSimple; K = model_params.K1);
VM = model_params.x .* repeat(vM, inner=model_params.I);
vNG = solveModelMature(model_params, AQSimple; K = model_params.K0);
VNG = model_params.x .* repeat(vNG, inner=model_params.I);

# Using the new grid to solve for firm with GO
AQ = formTransitionMatrix(model_params);
VY = solveModelYoung(model_params, AQ);

# Calculating growth option execising list
Ve = exerciseValue(model_params);
exercise_bound = calcExerciseBound(model_params, VY, Ve)
exercise_bound = Int.(exercise_bound)

####################################################################
# Calculating transition densities in parallel
####################################################################

function getStateList()
    # Making a list of states to estimate
    λ_list_to_est = Array(5:5:50);
    x_list_to_est = [50; 55; 60; 65; 70; 71; 72; 73; 74; 75; 76; 77; 78; 79; 80];

    state_list_to_est = [];
    for λ = λ_list_to_est
        for x = x_list_to_est
            state_list_to_est = [state_list_to_est; (x, λ)]
        end
    end

    return state_list_to_est
end

state_list_to_est = getStateList()

@show length(state_list_to_est)

print(string("\n--- Starting parallel iteration ---\n"))

T = 0.25;

# Mapping for different state
densities_list = @showprogress pmap(state_list_to_est) do state
    # Calculating densities for nearby points
    out_base = iterateModelForwardGO(model_params, AQ, exercise_bound, VM, state[1], state[2], T);
    out_higherλ = iterateModelForwardGO(model_params, AQ, exercise_bound, VM, state[1], state[2]+1, T);
    out_lowerλ = iterateModelForwardGO(model_params, AQ, exercise_bound, VM, state[1], state[2]-1, T);
    out_higherx = iterateModelForwardGO(model_params, AQ, exercise_bound, VM, state[1]+1, state[2], T);
    out_lowerx = iterateModelForwardGO(model_params, AQ, exercise_bound, VM, state[1]-1, state[2], T);
    
    # Outputting
    (out_base, out_higherλ, out_lowerλ, out_higherx, out_lowerx)
end

print(string("\n--- Saving distributions ---\n"))

# Saving densities
offset_name_list = ["base", "hl", "ll", "hx", "lx"];
g_name_list = ["geta"; "gY"; "gM"];

for istate = 1:length(densities_list)
    for ig_list = 1:5
        for iout = 1:3
            state = state_list_to_est[istate]
            offset = offset_name_list[ig_list]
            g_name = g_name_list[iout];
            filename = "code/option_distributions/" * g_name * "_" * offset * "_" * string(state[1]) * "_" * string(state[2]) * ".csv";
            saveg(densities_list[istate][ig_list][iout], filename)
        end
    end
end


