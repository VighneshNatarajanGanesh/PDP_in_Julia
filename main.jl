include("pdp.jl")

Pkg.activate("PDP", shared=true)
Pkg.add("RDatasets")
Pkg.add("MLJDecisionTreeInterface")
Pkg.add("DataFrames")
Pkg.add("MLJ")
Pkg.add("Plots")

using MLJ
using DataFrames
using Plots

# For more details on the model, visit: https://alan-turing-institute.github.io/MLJ.jl/dev/getting_started/

#load iris dataset
import RDatasets
iris = RDatasets.dataset("datasets", "iris");
y, X = unpack(iris, ==(:Species), colname -> true);
train, test = partition(eachindex(y), 0.7, shuffle=true); #train test split

# load, initialize model type and make the machine
Tree = @load DecisionTreeClassifier
tree = Tree()
evaluate(tree, X, y, resampling=CV(shuffle=true), measure=cross_entropy)
mach = machine(tree, X, y)

#train
fit!(mach, rows=train);

#evaluate
yhat = predict(mach, X[test,:]);
log_loss(yhat, y[test]) |> mean

# call the pdp function
PDP(mach, X[test,:], "SepalLength", ["setosa", "virginica", "versicolor"])
