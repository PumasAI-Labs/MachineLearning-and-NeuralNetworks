using DeepPumas
using CairoMakie
using Distributions
using Random
set_theme!(deep_light())

#
# 1. A SIMPLE MACHINE LEARNING (ML) MODEL
#
# 1.1. Sample subjects from an obvious `true_function`
# 1.2. Model `true_function` with a linear regression model
#

# 1.1. Sample subjects from an obvious `true_function`

true_function = x -> x
num_samples = 100
uniform = Uniform(-1, 1)
normal = Normal(0, 1)
σ = 0.25

x = rand(uniform, 1, num_samples)  # samples stored columnwise
ϵ = rand(normal, 1, num_samples)  # samples stored columnwise
y = true_function.(x) + σ * ϵ

fig = scatter(vec(x), vec(y); axis = (xlabel = "x", ylabel = "y"), label = "data");
lines!(-1 .. 1, true_function; color = :gray, label = "true");
axislegend(; position = :rb);
fig

# 1.2. Model `true_function` with a linear regression model

target = preprocess(x, y)  # DeepPumas `target`
linreg = MLPDomain(1, (1, identity); bias = true)  # DeepPumas multilayer perceptron
# y = a * x + b

fitted_linreg = fit(linreg, target; optim_alg = DeepPumas.BFGS())
coef(fitted_linreg)  # `true_function` is y = x + noise (that is, a = 1 b = 0)

ŷ = fitted_linreg(x)

fig = scatter(vec(x), vec(y); axis = (xlabel = "x", ylabel = "y"), label = "data");
scatter!(vec(x), vec(ŷ); label = "prediction");
lines!(-1 .. 1, true_function; color = :gray, label = "true");
axislegend(; position = :rb);
fig
