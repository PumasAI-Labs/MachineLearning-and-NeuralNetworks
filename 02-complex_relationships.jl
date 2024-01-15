# Note: `1-linear_regression.jl` needs to be executed first.

#
# 2. CAPTURING COMPLEX RELATIONSHIPS
#
# 2.1. Sample subjects from a more complex `true_function`
# 2.2. Exercise: Reason about using a linear regression to model the current `true_function`
# 2.3. Use a neural network (NN) to model `true_function`
#

# 2.1. Sample subjects from a more complex `true_function`

true_function = x -> x^2
x = rand(uniform, 1, num_samples)
ϵ = rand(normal, 1, num_samples)
y = true_function.(x) + σ * ϵ

fig = scatter(vec(x), vec(y); axis = (xlabel = "x", ylabel = "y"), label = "data");
lines!(-1..1, true_function; color = :gray, label = "true");
axislegend();
fig

# 2.2. Exercise: Reason about using a linear regression to model `true_function`

target = preprocess(x, y)
fitted_linreg =
    fit(linreg, target; optim_alg = DeepPumas.BFGS(), optim_options = (; iterations = 50))
coef(fitted_linreg)

ŷ_ex22_50iter = fitted_linreg(x)

fig = scatter(vec(x), vec(y); axis = (xlabel = "x", ylabel = "y"), label = "data");
scatter!(vec(x), vec(ŷ_ex22_50iter); label = "prediction");
lines!(-1..1, true_function; color = :gray, label = "true");
axislegend();
fig

# 2.3. Use a neural network (NN) to model `true_function`

nn = MLPDomain(1, (8, tanh), (1, identity); bias = true)
fitted_nn =
    fit(nn, target; optim_alg = DeepPumas.BFGS(), optim_options = (; iterations = 50))
coef(fitted_nn) # try to make sense of the parameters in the NN

ŷ = fitted_nn(x)

fig = scatter(vec(x), vec(y); axis = (xlabel = "x", ylabel = "y"), label = "data");
scatter!(vec(x), vec(ŷ), label = "prediction");
lines!(-1..1, true_function; color = :gray, label = "true");
axislegend();
fig
