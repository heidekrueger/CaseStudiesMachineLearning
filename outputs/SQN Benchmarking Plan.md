# SQN Benchmarking Plan

1. Do initial run on HIGGS Data set with as many iterations as possible, but such that runtime should be limited to an overnight run. We estimate that `add number` iterations should be good.
2. Analyze Results, draw graphs (see below) collecting results for batch sizes.
3. Based on most most promising parameter combination, test additional variable parameters in subsequent runs. (ceteris paribus)

##### Restuls for finding optimal iteration number:

###### 1000 iterations:
* 100 g, 0 H, 148 s
* 100 g, 100 H, 160 s
* 100 g, 1000 H, 1470 s

## Initial Run

### Fixed Parameters
* Fixed step length: `beta/k`, `beta = 5`
* `L=20`,
* `M=10`
* `1` SG update per batch
* Training set size: `5M`

### Variable Parameters: 
 * SG batch size `[100, 1000, 10000]`
 * Hessian batch size `[0, 100, 1000, 4000]`, where 0 corresponds to SGD for benchmarking

### Collected Data for each run:
* Iteration
* cumulative ADP
* cumulative number of objective function calls
* Objective value on gradient sample
* Cumulative cpu-time
* Parameters `w`
* Norm of stochastic gradient 

## Graphs that we can draw after initial run:

### Objective vs Iterations et al.
Due to the size of the dataset, the actual objective value cannot be calculated or plotted in reasonable time for many iterations. We therefore will try the following as substitutes:

#### Objective on Samples
Expectation: Plotting f_S vs.  `[ADP, iter, cpu]` should produce a graph that looks similar to a random walk. Plotting a moving average should ideally reveal a steady decrease over time (when seen over the course of several epochs, allthough we're not sure that will be realistic with the size of our dataset in one night).

#### Objective on fixed Subset of Training data
Calculating F for the first, say, `50000` or `500000` points in the dataset should
- be feasible
- look more similar to a 'normal' f vs iter graph.

#### Objective on full dataset
 This might be possible for selected iterations, such as each 1000th iteration?

### Other types of graphs/tables that might be insightful:
* Avg. cpu/iter for each combination of batch sizes

## Subsequent runs
Based on the insights from the intitial run, we want to test changes in the following ceteris paribus (if time permits):

* Changing the step-size rule to `max("Armijo", beta/k)`
* Using multiple gradient steps per batch (`1` vs `5`)
* Choices of `M`
* Choices of `L`.
* Choice of `beta`
* Choosing samples "intelligently"



