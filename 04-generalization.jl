# Note: `1-linear_regression.jl`, `2-complex_relationships.jl` and 
# `3-bias-variance_tradeoff.jl` need to be executed first and in that order.

#
# 4. GENERALIZATION
#
# 4.1. Withheld (or unseen) data
# 4.2. Validation loss as a proxy for generalization performance
# 4.3. Regularization to prevent overfitting
# 4.4. Programmatic hyperparameter tuning
# 

# 4.1. Withheld (or unseen) data

x_train, y_train = x, y
target_train = target

ϵ_valid = rand(normal, 1, num_samples)
x_valid = rand(uniform, 1, num_samples)
y_valid = true_function.(x_valid) + σ * ϵ_valid
target_valid = preprocess(x_valid, y_valid)

fig = scatter(vec(x_train), vec(y_train); axis = (xlabel = "x", ylabel = "y"), label = "training data");
scatter!(vec(x_valid), vec(y_valid); label = "validation data");
lines!(-1..1, true_function; color = :gray, label = "true");
axislegend();
fig

# 4.2. Validation loss as a proxy for generalization performance

loss_train_l, loss_valid_l = [], []

fitted_nn = fit(
    nn,
    target_train;
    optim_alg = DeepPumas.BFGS(),
    optim_options = (; iterations = 10),
)
push!(loss_train_l, sum((fitted_nn(x_train) .- y_train) .^ 2))
push!(loss_valid_l, sum((fitted_nn(x_valid) .- y_valid) .^ 2))

iteration_blocks = 100
for _ = 2:iteration_blocks
    global fitted_nn = fit(
        nn,
        target_train,
        coef(fitted_nn);
        optim_alg = DeepPumas.BFGS(),
        optim_options = (; iterations = 10),
    )
    push!(loss_train_l, sum((fitted_nn(x_train) .- y_train) .^ 2))
    push!(loss_valid_l, sum((fitted_nn(x_valid) .- y_valid) .^ 2))
end

iteration = 10 .* (1:iteration_blocks)
fig, ax = scatterlines(
    iteration,
    Float32.(loss_train_l);
    label = "training",
    axis = (; xlabel = "Iteration", ylabel = "Mean squared loss"),
);
scatterlines!(iteration, Float32.(loss_valid_l); label = "validation");
axislegend();
fig

# 4.3. Regularization to prevent overfitting

reg_nn = MLPDomain(1, (32, tanh), (32, tanh), (1, identity); bias = true, reg = L2(0.1))

reg_loss_train_l, reg_loss_valid_l = [], []

fitted_reg_nn = fit(
    reg_nn,
    target_train;
    optim_alg = DeepPumas.BFGS(),
    optim_options = (; iterations = 10),
)
push!(reg_loss_train_l, sum((fitted_reg_nn(x_train) .- y_train) .^ 2))
push!(reg_loss_valid_l, sum((fitted_reg_nn(x_valid) .- y_valid) .^ 2))

iteration_blocks = 100
for _ = 2:iteration_blocks
    global fitted_reg_nn = fit(
        reg_nn,
        target_train,
        coef(fitted_reg_nn);
        optim_alg = DeepPumas.BFGS(),
        optim_options = (; iterations = 10),
    )
    push!(reg_loss_train_l, sum((fitted_reg_nn(x_train) .- y_train) .^ 2))
    push!(reg_loss_valid_l, sum((fitted_reg_nn(x_valid) .- y_valid) .^ 2))
end

iteration = 10 .* (1:iteration_blocks)
fig, ax = scatterlines(
    iteration,
    Float32.(loss_train_l);
    label = "training",
    axis = (; xlabel = "Blocks of 10 iterations", ylabel = "Mean squared loss"),
);
scatterlines!(iteration, Float32.(loss_valid_l); label = "validation");
scatterlines!(iteration, Float32.(reg_loss_train_l); label = "training (L2)");
scatterlines!(iteration, Float32.(reg_loss_valid_l); label = "validation (L2)");
axislegend();
fig

# 4.4. Programmatic hyperparameter tuning

nn_ho = hyperopt(reg_nn, target_train)
nn_ho.best_hyperparameters
ŷ_ho = nn_ho(x_valid)

fig = scatter(vec(x_valid), vec(y_valid); label = "validation data");
scatter!(vec(x_valid), vec(ŷ_ho), label = "prediction (hyperparam opt.)");
lines!(-1..1, true_function; color = :gray, label = "true");
axislegend(; position=:ct);
fig
