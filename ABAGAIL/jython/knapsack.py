import sys
import os
import time

import java.io.FileReader as FileReader
import java.io.File as File
import java.lang.String as String
import java.lang.StringBuffer as StringBuffer
import java.lang.Boolean as Boolean
import java.util.Random as Random

import dist.DiscreteDependencyTree as DiscreteDependencyTree
import dist.DiscreteUniformDistribution as DiscreteUniformDistribution
import dist.Distribution as Distribution
import opt.DiscreteChangeOneNeighbor as DiscreteChangeOneNeighbor
import opt.EvaluationFunction as EvaluationFunction
import opt.GenericHillClimbingProblem as GenericHillClimbingProblem
import opt.HillClimbingProblem as HillClimbingProblem
import opt.NeighborFunction as NeighborFunction
import opt.RandomizedHillClimbing as RandomizedHillClimbing
import opt.SimulatedAnnealing as SimulatedAnnealing
import opt.example.FourPeaksEvaluationFunction as FourPeaksEvaluationFunction
import opt.ga.CrossoverFunction as CrossoverFunction
import opt.ga.SingleCrossOver as SingleCrossOver
import opt.ga.DiscreteChangeOneMutation as DiscreteChangeOneMutation
import opt.ga.GenericGeneticAlgorithmProblem as GenericGeneticAlgorithmProblem
import opt.ga.GeneticAlgorithmProblem as GeneticAlgorithmProblem
import opt.ga.MutationFunction as MutationFunction
import opt.ga.StandardGeneticAlgorithm as StandardGeneticAlgorithm
import opt.ga.UniformCrossOver as UniformCrossOver
import opt.prob.GenericProbabilisticOptimizationProblem as GenericProbabilisticOptimizationProblem
import opt.prob.MIMIC as MIMIC
import opt.prob.ProbabilisticOptimizationProblem as ProbabilisticOptimizationProblem
import shared.FixedIterationTrainer as FixedIterationTrainer
import opt.example.KnapsackEvaluationFunction as KnapsackEvaluationFunction
from array import array
import opt.example.ContinuousPeaksEvaluationFunction as ContinuousPeaksEvaluationFunction
from array import array
import shared.ConvergenceTrainer as ConvergenceTrainer
import csv


def train(alg_func, exp_name, alg_name, exp_params, ef, iters):
    ef.resetFunctionEvaluationCount()
    fit = ConvergenceTrainer(alg_func)
    FILE_NAME = exp_name + "-" + alg_name + "-" + exp_params + ".csv"
    OUTPUT_FILE = os.path.join("data/" + "knapsack", FILE_NAME)
    with open(OUTPUT_FILE, "wb") as results:
        writer = csv.writer(results, delimiter=',')
        writer.writerow(["iters", "fevals", "fitness", "time"])
        ts = time.time()
        for i in range(iters):
            fit.train()
            te = time.time()
            writer.writerow([i, ef.getFunctionEvaluations() - i, ef.value(alg_func.getOptimal()), (te-ts)])

    print alg_name + ": " + str(ef.value(alg_func.getOptimal()))
    print "Function Evaluations: " + str(ef.getFunctionEvaluations()-iters)
    print "Iters: " + str(iters)
    print "####"

"""
Commandline parameter(s):
    none
"""

# Random number generator */
random = Random()
# The number of items
NUM_ITEMS = 100
# The number of copies each
COPIES_EACH = 4
# The maximum weight for a single element
MAX_WEIGHT = 50
# The maximum volume for a single element
MAX_VOLUME = 50
# The volume of the knapsack 
KNAPSACK_VOLUME = MAX_VOLUME * NUM_ITEMS * COPIES_EACH * .4

# create copies
fill = [COPIES_EACH] * NUM_ITEMS
copies = array('i', fill)

# create weights and volumes
fill = [0] * NUM_ITEMS
weights = array('d', fill)
volumes = array('d', fill)
for i in range(0, NUM_ITEMS):
    weights[i] = random.nextDouble() * MAX_WEIGHT
    volumes[i] = random.nextDouble() * MAX_VOLUME


# create range
fill = [COPIES_EACH + 1] * NUM_ITEMS
ranges = array('i', fill)

ef = KnapsackEvaluationFunction(weights, volumes, KNAPSACK_VOLUME, copies)
odd = DiscreteUniformDistribution(ranges)
nf = DiscreteChangeOneNeighbor(ranges)
mf = DiscreteChangeOneMutation(ranges)
cf = UniformCrossOver()
df = DiscreteDependencyTree(.1, ranges)
hcp = GenericHillClimbingProblem(ef, odd, nf)
gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)
pop = GenericProbabilisticOptimizationProblem(ef, odd, df)

#### Experienet 0 - Tuning Algo Prams for MIMIC ####
if True:
    exp_name = "exp00"

    mimic = MIMIC(500, 25, pop)
    train(mimic, exp_name, "MIMIC", "0", ef, 2000)

    mimic = MIMIC(500, 50, pop)
    train(mimic, exp_name, "MIMIC", "1", ef, 2000)

    mimic = MIMIC(500, 100, pop)
    train(mimic, exp_name, "MIMIC", "2", ef, 2000)

    mimic = MIMIC(500, 25, pop)
    train(mimic, exp_name, "MIMIC", "3", ef, 2000)


    mimic = MIMIC(200, 10, pop)
    train(mimic, exp_name, "MIMIC", "4", ef, 2000)

    mimic = MIMIC(200, 25, pop)
    train(mimic, exp_name, "MIMIC", "5", ef, 2000)

    mimic = MIMIC(200, 50, pop)
    train(mimic, exp_name, "MIMIC", "6", ef, 2000)

    mimic = MIMIC(200, 100, pop)
    train(mimic, exp_name, "MIMIC", "7", ef, 2000)

    mimic = MIMIC(200, 125, pop)
    train(mimic, exp_name, "MIMIC", "8", ef, 2000)

    mimic = MIMIC(150, 10, pop)
    train(mimic, exp_name, "MIMIC", "9", ef, 2000)

    mimic = MIMIC(150, 25, pop)
    train(mimic, exp_name, "MIMIC", "10", ef, 2000)

    mimic = MIMIC(150, 50, pop)
    train(mimic, exp_name, "MIMIC", "11", ef, 2000)

    mimic = MIMIC(150, 75, pop)
    train(mimic, exp_name, "MIMIC", "12", ef, 2000)

    mimic = MIMIC(150, 100, pop)
    train(mimic, exp_name, "MIMIC", "13", ef, 2000)


    mimic = MIMIC(100, 25, pop)
    train(mimic, exp_name, "MIMIC", "14", ef, 2000)

    mimic = MIMIC(100, 50, pop)
    train(mimic, exp_name, "MIMIC", "15", ef, 2000)

    mimic = MIMIC(100, 75, pop)
    train(mimic, exp_name, "MIMIC", "16", ef, 2000)


    mimic = MIMIC(50, 5, pop)
    train(mimic, exp_name, "MIMIC", "17", ef, 2000)


    mimic = MIMIC(50, 10, pop)
    train(mimic, exp_name, "MIMIC", "18", ef, 2000)

    mimic = MIMIC(50, 25, pop)
    train(mimic, exp_name, "MIMIC", "19", ef, 2000)

    mimic = MIMIC(25, 5, pop)
    train(mimic, exp_name, "MIMIC", "20", ef, 2000)

    mimic = MIMIC(25, 10, pop)
    train(mimic, exp_name, "MIMIC", "21", ef, 2000)

    mimic = MIMIC(10, 5, pop)
    train(mimic, exp_name, "MIMIC", "22", ef, 2000)


#### Experienet 0 - Tuning Algo Prams for RHC ####
if True:
    exp_name = "exp01"
    rhc = RandomizedHillClimbing(hcp)
    train(rhc, exp_name, "RHC", "1", ef, 2000)



#### Experienet 0 - Tuning Algo Prams for SA ####
if True:
    exp_name = "exp02"

    sa = SimulatedAnnealing(1E10, .95, hcp)
    train(sa, exp_name, "SA", "0", ef, 2000)

    sa = SimulatedAnnealing(1E5, .95, hcp)
    train(sa, exp_name, "SA", "1", ef, 2000)

    sa = SimulatedAnnealing(1E3, .95, hcp)
    train(sa, exp_name, "SA", "2", ef, 2000)

    sa = SimulatedAnnealing(1E10, .90, hcp)
    train(sa, exp_name, "SA", "3", ef, 2000)

    sa = SimulatedAnnealing(1E5, .90, hcp)
    train(sa, exp_name, "SA", "4", ef, 2000)

    sa = SimulatedAnnealing(1E3, .90, hcp)
    train(sa, exp_name, "SA", "5", ef, 2000)

    sa = SimulatedAnnealing(1E10, .85, hcp)
    train(sa, exp_name, "SA", "6", ef, 2000)

    sa = SimulatedAnnealing(1E5, .85, hcp)
    train(sa, exp_name, "SA", "7", ef, 2000)

    sa = SimulatedAnnealing(1E3, .85, hcp)
    train(sa, exp_name, "SA", "8", ef, 2000)

    sa = SimulatedAnnealing(1E10, .99, hcp)
    train(sa, exp_name, "SA", "9", ef, 2000)

    sa = SimulatedAnnealing(1E5, .99, hcp)
    train(sa, exp_name, "SA", "10", ef, 2000)

    sa = SimulatedAnnealing(1E3, .99, hcp)
    train(sa, exp_name, "SA", "11", ef, 2000)

    sa = SimulatedAnnealing(1E12, .99, hcp)
    train(sa, exp_name, "SA", "12", ef, 2000)


#### Experienet 0 - Tuning Algo Prams for GA ####
if True:
    exp_name = "exp03"

    ga = StandardGeneticAlgorithm(500, 250, 50, gap)
    train(ga, exp_name, "GA", "0", ef, 20000)

    ga = StandardGeneticAlgorithm(500, 250, 100, gap)
    train(ga, exp_name, "GA", "1", ef, 20000)

    ga = StandardGeneticAlgorithm(200, 100, 50, gap)
    train(ga, exp_name, "GA", "2", ef, 20000)

    ga = StandardGeneticAlgorithm(200, 100, 10, gap)
    train(ga, exp_name, "GA", "3", ef, 20000)

    ga = StandardGeneticAlgorithm(200, 50, 25, gap)
    train(ga, exp_name, "GA", "4", ef, 20000)

    ga = StandardGeneticAlgorithm(200, 50, 10, gap)
    train(ga, exp_name, "GA", "5", ef, 20000)

    ga = StandardGeneticAlgorithm(200, 50, 20, gap)
    train(ga, exp_name, "GA", "55", ef, 20000)

    ga = StandardGeneticAlgorithm(200, 50, 5, gap)
    train(ga, exp_name, "GA", "6", ef, 20000)

    ga = StandardGeneticAlgorithm(200, 50, 1, gap)
    train(ga, exp_name, "GA", "66", ef, 20000)


    ga = StandardGeneticAlgorithm(100, 50, 25, gap)
    train(ga, exp_name, "GA", "7", ef, 20000)

    ga = StandardGeneticAlgorithm(100, 25, 10, gap)
    train(ga, exp_name, "GA", "71", ef, 20000)

    ga = StandardGeneticAlgorithm(100, 25, 5, gap)
    train(ga, exp_name, "GA", "72", ef, 20000)

    ga = StandardGeneticAlgorithm(100, 25, 2, gap)
    train(ga, exp_name, "GA", "73", ef, 20000)

    ga = StandardGeneticAlgorithm(100, 50, 10, gap)
    train(ga, exp_name, "GA", "8", ef, 20000)

    ga = StandardGeneticAlgorithm(50, 25, 10, gap)
    train(ga, exp_name, "GA", "0", ef, 20000)

    ga = StandardGeneticAlgorithm(50, 25, 5, gap)
    train(ga, exp_name, "GA", "9", ef, 20000)

    ga = StandardGeneticAlgorithm(50, 10, 5, gap)
    train(ga, exp_name, "GA", "10", ef, 20000)




#### Experienet 4 - Small Problem Size ####

# The number of items
NUM_ITEMS = 50
# The number of copies each
COPIES_EACH = 2
# The maximum weight for a single element
MAX_WEIGHT = 25
# The maximum volume for a single element
MAX_VOLUME = 25
# The volume of the knapsack
KNAPSACK_VOLUME = MAX_VOLUME * NUM_ITEMS * COPIES_EACH * .4

# create copies
fill = [COPIES_EACH] * NUM_ITEMS
copies = array('i', fill)

# create weights and volumes
fill = [0] * NUM_ITEMS
weights = array('d', fill)
volumes = array('d', fill)
for i in range(0, NUM_ITEMS):
    weights[i] = random.nextDouble() * MAX_WEIGHT
    volumes[i] = random.nextDouble() * MAX_VOLUME


# create range
fill = [COPIES_EACH + 1] * NUM_ITEMS
ranges = array('i', fill)

ef = KnapsackEvaluationFunction(weights, volumes, KNAPSACK_VOLUME, copies)
odd = DiscreteUniformDistribution(ranges)
nf = DiscreteChangeOneNeighbor(ranges)
mf = DiscreteChangeOneMutation(ranges)
cf = UniformCrossOver()
df = DiscreteDependencyTree(.1, ranges)
hcp = GenericHillClimbingProblem(ef, odd, nf)
gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)
pop = GenericProbabilisticOptimizationProblem(ef, odd, df)


if True:
    exp_name = "exp04"

    mimic = MIMIC(50, 10, pop)
    train(mimic, exp_name, "MIMIC", "0", ef, 2000)

    rhc = RandomizedHillClimbing(hcp)
    train(rhc, exp_name, "RHC", "0", ef, 2000)

    sa = SimulatedAnnealing(1E3, .99, hcp)
    train(sa, exp_name, "SA", "0", ef, 2000)

    ga = StandardGeneticAlgorithm(200, 50, 25, gap)
    train(ga, exp_name, "GA", "0", ef, 20000)



#### Experienet 4 - Small Problem Size ####

# The number of items
NUM_ITEMS = 200
# The number of copies each
COPIES_EACH = 8
# The maximum weight for a single element
MAX_WEIGHT = 100
# The maximum volume for a single element
MAX_VOLUME = 100
# The volume of the knapsack
KNAPSACK_VOLUME = MAX_VOLUME * NUM_ITEMS * COPIES_EACH * .4

# create copies
fill = [COPIES_EACH] * NUM_ITEMS
copies = array('i', fill)

# create weights and volumes
fill = [0] * NUM_ITEMS
weights = array('d', fill)
volumes = array('d', fill)
for i in range(0, NUM_ITEMS):
    weights[i] = random.nextDouble() * MAX_WEIGHT
    volumes[i] = random.nextDouble() * MAX_VOLUME


# create range
fill = [COPIES_EACH + 1] * NUM_ITEMS
ranges = array('i', fill)

ef = KnapsackEvaluationFunction(weights, volumes, KNAPSACK_VOLUME, copies)
odd = DiscreteUniformDistribution(ranges)
nf = DiscreteChangeOneNeighbor(ranges)
mf = DiscreteChangeOneMutation(ranges)
cf = UniformCrossOver()
df = DiscreteDependencyTree(.1, ranges)
hcp = GenericHillClimbingProblem(ef, odd, nf)
gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)
pop = GenericProbabilisticOptimizationProblem(ef, odd, df)


if True:
    exp_name = "exp05"

    mimic = MIMIC(50, 10, pop)
    train(mimic, exp_name, "MIMIC", "0", ef, 2000)

    rhc = RandomizedHillClimbing(hcp)
    train(rhc, exp_name, "RHC", "0", ef, 2000)

    sa = SimulatedAnnealing(1E3, .99, hcp)
    train(sa, exp_name, "SA", "0", ef, 2000)

    ga = StandardGeneticAlgorithm(200, 50, 25, gap)
    train(ga, exp_name, "GA", "0", ef, 20000)

