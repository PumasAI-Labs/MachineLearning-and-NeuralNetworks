# Note: `1-linear_regression.jl` and `2-complex_relationships.jl` need to be executed 
# first and in that order.

#
# 3. BIAS-VARIANCE TRADEOFF
#
# 3.1. Exercise: Investigate the impact of the number of fitting iterations in NNs
#      (Hint: Train again the NN for few and for many iterations.)
# 3.2. Exercise: Reason about Exercise 2.2 again (that is, using a linear regression 
#      to model a quadratic relationship). Is the number of iterations relevant there?
# 3.3. The impact of the NN size
# 

# 3.1. Exercise: Investigate the impact of the number of fitting iterations in NNs
#      (Hint: Train again the NN for few and for many iterations.)

underfit_nn =
    fit(nn, target; optim_alg = DeepPumas.BFGS(), optim_options = (; iterations = 2))
ŷ_underfit = underfit_nn(x)

overfit_nn =
    fit(nn, target; optim_alg = DeepPumas.BFGS(), optim_options = (; iterations = 1_000))
ŷ_overfit = overfit_nn(x)  # clarification on the term "overfitting"

fig = scatter(vec(x), vec(y); axis = (xlabel = "x", ylabel = "y"), label = "data");
scatter!(vec(x), vec(ŷ_underfit), label = "prediction (5 iterations)");
scatter!(vec(x), vec(ŷ), label = "prediction (50 iterations)");
scatter!(vec(x), vec(ŷ_overfit), label = "prediction (1000 iterations)");
lines!(-1 .. 1, true_function; color = :gray, label = "true");
axislegend();
fig

# 3.2. Exercise: Reason about Exercise 2.2 again (that is, using a linear regression 
#      to model a quadratic relationship). Is the number of iterations relevant there?
#      Investigate the effect of `max_iterations`.

max_iterations = 2
fitted_linreg = fit(
    linreg,
    target;
    optim_alg = DeepPumas.BFGS(),
    optim_options = (; iterations = max_iterations),
)
ŷ_linreg = fitted_linreg(x)

fig = scatter(vec(x), vec(y); axis = (xlabel = "x", ylabel = "y"), label = "data");
scatter!(vec(x), vec(ŷ_linreg), label = "$max_iterations iterations");
scatter!(vec(x), vec(ŷ_ex22_50iter), label = "50 iterations");
lines!(-1 .. 1, true_function; color = :gray, label = "true");
axislegend();
fig

# 3.3. The impact of the NN size

nn = MLPDomain(1, (32, tanh), (32, tanh), (1, identity); bias = true)
fitted_nn =
    fit(nn, target; optim_alg = DeepPumas.BFGS(), optim_options = (; iterations = 1_000))

ŷ = fitted_nn(x)

fig = scatter(vec(x), vec(y); axis = (xlabel = "x", ylabel = "y"), label = "data");
scatter!(vec(x), vec(ŷ), label = "prediction MLP(1, 32, 32, 1)");
lines!(-1:0.1:1, true_function.(-1:0.1:1); color = :gray, label = "true");
axislegend();
fig
