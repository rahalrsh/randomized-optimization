# traveling salesman algorithm implementation in jython
# This also prints the index of the points of the shortest route.
# To make a plot of the route, write the points at these indexes 
# to a file and plot them in your favorite tool.
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
import dist.DiscretePermutationDistribution as DiscretePermutationDistribution
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
import opt.example.TravelingSalesmanEvaluationFunction as TravelingSalesmanEvaluationFunction
import opt.example.TravelingSalesmanRouteEvaluationFunction as TravelingSalesmanRouteEvaluationFunction
import opt.SwapNeighbor as SwapNeighbor
import opt.ga.SwapMutation as SwapMutation
import opt.example.TravelingSalesmanCrossOver as TravelingSalesmanCrossOver
import opt.example.TravelingSalesmanSortEvaluationFunction as TravelingSalesmanSortEvaluationFunction
import shared.Instance as Instance
import util.ABAGAILArrays as ABAGAILArrays
import opt.example.ContinuousPeaksEvaluationFunction as ContinuousPeaksEvaluationFunction
from array import array
import shared.ConvergenceTrainer as ConvergenceTrainer
import csv

from array import array


def train(alg_func, exp_name, alg_name, exp_params, ef, iters):
    ef.resetFunctionEvaluationCount()
    fit = ConvergenceTrainer(alg_func)
    FILE_NAME = exp_name + "-" + alg_name + "-" + exp_params + ".csv"
    OUTPUT_FILE = os.path.join("data/" + "tsp", FILE_NAME)
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

# set N value.  This is the number of points
N = 100
random = Random()

points = [[0 for x in xrange(2)] for x in xrange(N)]
for i in range(0, len(points)):
    points[i][0] = random.nextDouble()
    points[i][1] = random.nextDouble()

ef = TravelingSalesmanRouteEvaluationFunction(points)
odd = DiscretePermutationDistribution(N)
nf = SwapNeighbor()
mf = SwapMutation()
cf = TravelingSalesmanCrossOver(ef)
hcp = GenericHillClimbingProblem(ef, odd, nf)
gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)

# rhc = RandomizedHillClimbing(hcp)
# fit = FixedIterationTrainer(rhc, 200000)
# train(rhc, "RHC", ef, 200000)
#
# sa = SimulatedAnnealing(1E12, .999, hcp)
# fit = FixedIterationTrainer(sa, 200000)
# train(sa, "SA", ef, 200000)
#
# ga = StandardGeneticAlgorithm(2000, 1500, 250, gap)
# fit = FixedIterationTrainer(ga, 1000)
# train(ga, "GA", ef, 2000)
#

# for mimic we use a sort encoding
ef = TravelingSalesmanSortEvaluationFunction(points);
fill = [N] * N
ranges = array('i', fill)
odd = DiscreteUniformDistribution(ranges);
df = DiscreteDependencyTree(.1, ranges);
pop = GenericProbabilisticOptimizationProblem(ef, odd, df);

# mimic = MIMIC(500, 100, pop)
# fit = FixedIterationTrainer(mimic, 1000)
# train(mimic, "MIMIC", ef, 2000)


#### Experienet 0 - Tuning Algo Prams for RHC ####
if False:
    exp_name = "exp00"
    rhc = RandomizedHillClimbing(hcp)
    fit = FixedIterationTrainer(rhc, 200000)
    train(rhc, exp_name, "RHC", "0", ef, 200000)

#### Experienet 1 - Tuning Algo Prams for SA ####
if False:
    exp_name = "exp01"

    sa = SimulatedAnnealing(1E10, .95, hcp)
    fit = FixedIterationTrainer(sa, 200000)
    train(sa, exp_name, "SA", "0", ef, 200000)

    sa = SimulatedAnnealing(1E5, .95, hcp)
    fit = FixedIterationTrainer(sa, 200000)
    train(sa, exp_name, "SA", "1", ef, 200000)

    sa = SimulatedAnnealing(1E3, .95, hcp)
    fit = FixedIterationTrainer(sa, 200000)
    train(sa, exp_name, "SA", "2", ef, 200000)

    sa = SimulatedAnnealing(1E10, .90, hcp)
    fit = FixedIterationTrainer(sa, 200000)
    train(sa, exp_name, "SA", "3", ef, 200000)

    sa = SimulatedAnnealing(1E5, .90, hcp)
    fit = FixedIterationTrainer(sa, 200000)
    train(sa, exp_name, "SA", "4", ef, 200000)

    sa = SimulatedAnnealing(1E3, .90, hcp)
    fit = FixedIterationTrainer(sa, 200000)
    train(sa, exp_name, "SA", "5", ef, 200000)

    sa = SimulatedAnnealing(1E10, .85, hcp)
    fit = FixedIterationTrainer(sa, 200000)
    train(sa, exp_name, "SA", "6", ef, 200000)

    sa = SimulatedAnnealing(1E5, .85, hcp)
    fit = FixedIterationTrainer(sa, 200000)
    train(sa, exp_name, "SA", "7", ef, 200000)

    sa = SimulatedAnnealing(1E3, .85, hcp)
    fit = FixedIterationTrainer(sa, 200000)
    train(sa, exp_name, "SA", "8", ef, 200000)

    sa = SimulatedAnnealing(1E10, .99, hcp)
    fit = FixedIterationTrainer(sa, 200000)
    train(sa, exp_name, "SA", "9", ef, 200000)

    sa = SimulatedAnnealing(1E5, .99, hcp)
    fit = FixedIterationTrainer(sa, 200000)
    train(sa, exp_name, "SA", "10", ef, 200000)

    sa = SimulatedAnnealing(1E3, .99, hcp)
    fit = FixedIterationTrainer(sa, 200000)
    train(sa, exp_name, "SA", "11", ef, 200000)

    sa = SimulatedAnnealing(1E12, .99, hcp)
    fit = FixedIterationTrainer(sa, 200000)
    train(sa, exp_name, "SA", "12", ef, 200000)

    sa = SimulatedAnnealing(1E12, .95, hcp)
    fit = FixedIterationTrainer(sa, 200000)
    train(sa, exp_name, "SA", "13", ef, 200000)

    sa = SimulatedAnnealing(1E12, .90, hcp)
    fit = FixedIterationTrainer(sa, 200000)
    train(sa, exp_name, "SA", "14", ef, 200000)

    sa = SimulatedAnnealing(1E12, .80, hcp)
    fit = FixedIterationTrainer(sa, 200000)
    train(sa, exp_name, "SA", "15", ef, 200000)


#### Experienet 2 - Tuning Algo Prams for Mimic ####
if False:
    exp_name = "exp02"

    mimic = MIMIC(500, 25, pop)
    fit = FixedIterationTrainer(mimic, 1000)
    train(mimic, exp_name, "MIMIC", "0", ef, 2000)

    mimic = MIMIC(500, 50, pop)
    fit = FixedIterationTrainer(mimic, 1000)
    train(mimic, exp_name, "MIMIC", "1", ef, 2000)

    mimic = MIMIC(500, 100, pop)
    fit = FixedIterationTrainer(mimic, 1000)
    train(mimic, exp_name, "MIMIC", "2", ef, 2000)

    mimic = MIMIC(200, 10, pop)
    fit = FixedIterationTrainer(mimic, 1000)
    train(mimic, exp_name, "MIMIC", "3", ef, 2000)

    mimic = MIMIC(200, 25, pop)
    fit = FixedIterationTrainer(mimic, 1000)
    train(mimic, exp_name, "MIMIC", "4", ef, 2000)

    mimic = MIMIC(200, 50, pop)
    fit = FixedIterationTrainer(mimic, 1000)
    train(mimic, exp_name, "MIMIC", "5", ef, 2000)

    mimic = MIMIC(100, 25, pop)
    fit = FixedIterationTrainer(mimic, 1000)
    train(mimic, exp_name, "MIMIC", "6", ef, 2000)

    mimic = MIMIC(100, 50, pop)
    fit = FixedIterationTrainer(mimic, 1000)
    train(mimic, exp_name, "MIMIC", "7", ef, 2000)

    mimic = MIMIC(50, 5, pop)
    fit = FixedIterationTrainer(mimic, 1000)
    train(mimic, exp_name, "MIMIC", "8", ef, 2000)

    mimic = MIMIC(50, 10, pop)
    fit = FixedIterationTrainer(mimic, 1000)
    train(mimic, exp_name, "MIMIC", "9", ef, 2000)

    mimic = MIMIC(50, 25, pop)
    fit = FixedIterationTrainer(mimic, 1000)
    train(mimic, exp_name, "MIMIC", "10", ef, 2000)

    mimic = MIMIC(25, 10, pop)
    fit = FixedIterationTrainer(mimic, 1000)
    train(mimic, exp_name, "MIMIC", "11", ef, 2000)

    mimic = MIMIC(10, 5, pop)
    fit = FixedIterationTrainer(mimic, 1000)
    train(mimic, exp_name, "MIMIC", "12", ef, 2000)




#### Experienet 3 - Tuning Algo Prams for GA ####
if True:
    exp_name = "exp03"

    ga = StandardGeneticAlgorithm(500, 250, 50, gap)
    fit = FixedIterationTrainer(ga, 1000)
    train(ga, exp_name, "GA", "0", ef, 20000)

    ga = StandardGeneticAlgorithm(500, 250, 100, gap)
    fit = FixedIterationTrainer(ga, 1000)
    train(ga, exp_name, "GA", "1", ef, 20000)

    ga = StandardGeneticAlgorithm(200, 100, 50, gap)
    fit = FixedIterationTrainer(ga, 1000)
    train(ga, exp_name, "GA", "2", ef, 20000)

    ga = StandardGeneticAlgorithm(200, 100, 10, gap)
    fit = FixedIterationTrainer(ga, 1000)
    train(ga, exp_name, "GA", "3", ef, 20000)

    ga = StandardGeneticAlgorithm(200, 50, 25, gap)
    fit = FixedIterationTrainer(ga, 1000)
    train(ga, exp_name, "GA", "4", ef, 20000)

    ga = StandardGeneticAlgorithm(200, 50, 10, gap)
    fit = FixedIterationTrainer(ga, 1000)
    train(ga, exp_name, "GA", "5", ef, 20000)

    ga = StandardGeneticAlgorithm(200, 50, 5, gap)
    fit = FixedIterationTrainer(ga, 1000)
    train(ga, exp_name, "GA", "6", ef, 20000)

    ga = StandardGeneticAlgorithm(100, 50, 25, gap)
    fit = FixedIterationTrainer(ga, 1000)
    train(ga, exp_name, "GA", "7", ef, 20000)

    ga = StandardGeneticAlgorithm(100, 25, 10, gap)
    fit = FixedIterationTrainer(ga, 1000)
    train(ga, exp_name, "GA", "8", ef, 20000)

    ga = StandardGeneticAlgorithm(100, 25, 5, gap)
    fit = FixedIterationTrainer(ga, 1000)
    train(ga, exp_name, "GA", "9", ef, 20000)

    ga = StandardGeneticAlgorithm(100, 25, 2, gap)
    fit = FixedIterationTrainer(ga, 1000)
    train(ga, exp_name, "GA", "10", ef, 20000)

    ga = StandardGeneticAlgorithm(100, 50, 10, gap)
    fit = FixedIterationTrainer(ga, 1000)
    train(ga, exp_name, "GA", "11", ef, 20000)

    ga = StandardGeneticAlgorithm(50, 25, 10, gap)
    fit = FixedIterationTrainer(ga, 1000)
    train(ga, exp_name, "GA", "12", ef, 20000)

    ga = StandardGeneticAlgorithm(50, 25, 5, gap)
    fit = FixedIterationTrainer(ga, 1000)
    train(ga, exp_name, "GA", "13", ef, 20000)

    ga = StandardGeneticAlgorithm(50, 10, 5, gap)
    fit = FixedIterationTrainer(ga, 1000)
    train(ga, exp_name, "GA", "14", ef, 20000)

    ga = StandardGeneticAlgorithm(2000, 1500, 250, gap)
    fit = FixedIterationTrainer(ga, 1000)
    train(ga, exp_name, "GA", "15", ef, 20000)

    ga = StandardGeneticAlgorithm(1000, 500, 250, gap)
    fit = FixedIterationTrainer(ga, 1000)
    train(ga, exp_name, "GA", "16", ef, 20000)
































