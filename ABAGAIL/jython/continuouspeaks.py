import sys
import os
import time
import csv
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
import opt.example.ContinuousPeaksEvaluationFunction as ContinuousPeaksEvaluationFunction
from array import array
import shared.ConvergenceTrainer as ConvergenceTrainer


"""
Commandline parameter(s):
   none
"""


def train(alg_func, exp_name, alg_name, exp_params, ef, iters):
    ef.resetFunctionEvaluationCount()
    fit = ConvergenceTrainer(alg_func)
    FILE_NAME = exp_name + "-" + alg_name + "-" + exp_params + ".csv"
    OUTPUT_FILE = os.path.join("data/" + "continous-peaks", FILE_NAME)
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

N=100
T=N/50
fill = [2] * N
ranges = array('i', fill)

ef = ContinuousPeaksEvaluationFunction(T)
odd = DiscreteUniformDistribution(ranges)
nf = DiscreteChangeOneNeighbor(ranges)
mf = DiscreteChangeOneMutation(ranges)
cf = SingleCrossOver()
df = DiscreteDependencyTree(.1, ranges)
hcp = GenericHillClimbingProblem(ef, odd, nf)
gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)
pop = GenericProbabilisticOptimizationProblem(ef, odd, df)

#### Experienet 0 - Tuning Algo Prams for SA ####
if True:
    exp_name = "exp00"

    # sa = SimulatedAnnealing(1E11, .95, hcp)
    # train(sa, exp_name, "SA", "0", ef, 30000)
    #
    # sa = SimulatedAnnealing(1E9, .95, hcp)
    # train(sa, exp_name, "SA", "1", ef, 30000)
    #
    # sa = SimulatedAnnealing(1E7, .95, hcp)
    # train(sa, exp_name, "SA", "2", ef, 30000)
    #
    # sa = SimulatedAnnealing(1E5, .95, hcp)
    # train(sa, exp_name, "SA", "3", ef, 30000)
    #
    # sa = SimulatedAnnealing(1E3, .95, hcp)
    # train(sa, exp_name, "SA", "4", ef, 30000)
    #
    # sa = SimulatedAnnealing(1E1, .95, hcp)
    # train(sa, exp_name, "SA", "5", ef, 30000)

    sa = SimulatedAnnealing(1E11, .99, hcp)
    train(sa, exp_name, "SA", "0", ef, 30000)

    sa = SimulatedAnnealing(1E11, .98, hcp)
    train(sa, exp_name, "SA", "1", ef, 30000)

    sa = SimulatedAnnealing(1E11, .97, hcp)
    train(sa, exp_name, "SA", "2", ef, 30000)

    sa = SimulatedAnnealing(1E11, .96, hcp)
    train(sa, exp_name, "SA", "3", ef, 30000)

    sa = SimulatedAnnealing(1E11, .95, hcp)
    train(sa, exp_name, "SA", "4", ef, 30000)

    sa = SimulatedAnnealing(1E11, .80, hcp)
    train(sa, exp_name, "SA", "5", ef, 30000)


    sa = SimulatedAnnealing(1E5, .99, hcp)
    train(sa, exp_name, "SA", "6", ef, 30000)

    sa = SimulatedAnnealing(1E5, .98, hcp)
    train(sa, exp_name, "SA", "7", ef, 30000)

    sa = SimulatedAnnealing(1E5, .97, hcp)
    train(sa, exp_name, "SA", "8", ef, 30000)

    sa = SimulatedAnnealing(1E5, .96, hcp)
    train(sa, exp_name, "SA", "9", ef, 30000)

    sa = SimulatedAnnealing(1E5, .95, hcp)
    train(sa, exp_name, "SA", "10", ef, 30000)

    sa = SimulatedAnnealing(1E5, .90, hcp)
    train(sa, exp_name, "SA", "11", ef, 30000)

    sa = SimulatedAnnealing(1E5, .80, hcp)
    train(sa, exp_name, "SA", "12", ef, 30000)

    # sa = SimulatedAnnealing(1E5, .70, hcp)
    # train(sa, exp_name, "SA", "13", ef, 30000)


#### Experienet 1 - Tuning Algo Prams for RHC ####
if True:
    exp_name = "exp01"
    rhc = RandomizedHillClimbing(hcp)
    train(rhc, exp_name, "RHC", "0", ef, 30000)


#### Experienet 2 - Tuning Algo Prams for GA ####
if True:
    exp_name = "exp02"
    ga = StandardGeneticAlgorithm(500, 250, 25, gap)
    train(ga, exp_name, "GA", "0", ef, 40000)

    ga = StandardGeneticAlgorithm(300, 150, 15, gap)
    train(ga, exp_name, "GA", "1", ef, 40000)

    ga = StandardGeneticAlgorithm(200, 100, 10, gap)
    train(ga, exp_name, "GA", "2", ef, 40000)

    ga = StandardGeneticAlgorithm(100, 50, 5, gap)
    train(ga, exp_name, "GA", "3", ef, 40000)

    ga = StandardGeneticAlgorithm(50, 25, 2, gap)
    train(ga, exp_name, "GA", "4", ef, 40000)


    ga = StandardGeneticAlgorithm(500, 125, 12, gap)
    train(ga, exp_name, "GA", "5", ef, 40000)

    ga = StandardGeneticAlgorithm(300, 75, 7, gap)
    train(ga, exp_name, "GA", "6", ef, 40000)

    ga = StandardGeneticAlgorithm(200, 50, 5, gap)
    train(ga, exp_name, "GA", "7", ef, 40000)

    ga = StandardGeneticAlgorithm(100, 25, 3, gap)
    train(ga, exp_name, "GA", "8", ef, 40000)

    ga = StandardGeneticAlgorithm(50, 12, 2, gap)
    train(ga, exp_name, "GA", "9", ef, 40000)

#### Experienet 3 - Tuning Algo Prams for MIMIC ####
if True:
    exp_name = "exp03"

    mimic = MIMIC(300, 150, pop)
    train(mimic, exp_name, "MIMIC", "0", ef, 2000)

    mimic = MIMIC(300, 100, pop)
    train(mimic, exp_name, "MIMIC", "1", ef, 2000)

    mimic = MIMIC(200, 100, pop)
    train(mimic, exp_name, "MIMIC", "2", ef, 2000)

    mimic = MIMIC(200, 50, pop)
    train(mimic, exp_name, "MIMIC", "3", ef, 2000)

    mimic = MIMIC(100, 50, pop)
    train(mimic, exp_name, "MIMIC", "4", ef, 2000)

    mimic = MIMIC(100, 25, pop)
    train(mimic, exp_name, "MIMIC", "5", ef, 2000)

    mimic = MIMIC(50, 25, pop)
    train(mimic, exp_name, "MIMIC", "6", ef, 2000)




#### Experienet N=100 ####

N=100
T= N/50
fill = [2] * N
ranges = array('i', fill)

ef = ContinuousPeaksEvaluationFunction(T)
odd = DiscreteUniformDistribution(ranges)
nf = DiscreteChangeOneNeighbor(ranges)
mf = DiscreteChangeOneMutation(ranges)
cf = SingleCrossOver()
df = DiscreteDependencyTree(.1, ranges)
hcp = GenericHillClimbingProblem(ef, odd, nf)
gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)
pop = GenericProbabilisticOptimizationProblem(ef, odd, df)

if True:
    exp_name = "exp04"
    rhc = RandomizedHillClimbing(hcp)
    train(rhc, exp_name, "RHC", "N=100", ef, 30000)

    sa = SimulatedAnnealing(1E5, .97, hcp)
    train(sa, exp_name, "SA", "N=100", ef, 30000)

    ga = StandardGeneticAlgorithm(300, 150, 15, gap)
    train(ga, exp_name, "GA", "N=100", ef, 30000)

    mimic = MIMIC(100, 25, pop)
    train(mimic, exp_name, "MIMIC", "N=100", ef, 5000)


#### Experienet Small Problem Size ####

N=50
T= N/50
fill = [2] * N
ranges = array('i', fill)

ef = ContinuousPeaksEvaluationFunction(T)
odd = DiscreteUniformDistribution(ranges)
nf = DiscreteChangeOneNeighbor(ranges)
mf = DiscreteChangeOneMutation(ranges)
cf = SingleCrossOver()
df = DiscreteDependencyTree(.1, ranges)
hcp = GenericHillClimbingProblem(ef, odd, nf)
gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)
pop = GenericProbabilisticOptimizationProblem(ef, odd, df)

if True:
    exp_name = "exp05"
    rhc = RandomizedHillClimbing(hcp)
    train(rhc, exp_name, "RHC", "N=50", ef, 30000)

    sa = SimulatedAnnealing(1E5, .97, hcp)
    train(sa, exp_name, "SA", "N=50", ef, 30000)

    ga = StandardGeneticAlgorithm(300, 150, 15, gap)
    train(ga, exp_name, "GA", "N=50", ef, 30000)

    mimic = MIMIC(100, 25, pop)
    train(mimic, exp_name, "MIMIC", "N=50", ef, 2000)




#### Experienet Large Problem Size ####

N=200
T=N/50
fill = [2] * N
ranges = array('i', fill)

ef = ContinuousPeaksEvaluationFunction(T)
odd = DiscreteUniformDistribution(ranges)
nf = DiscreteChangeOneNeighbor(ranges)
mf = DiscreteChangeOneMutation(ranges)
cf = SingleCrossOver()
df = DiscreteDependencyTree(.1, ranges)
hcp = GenericHillClimbingProblem(ef, odd, nf)
gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)
pop = GenericProbabilisticOptimizationProblem(ef, odd, df)

if True:
    exp_name = "exp06"
    rhc = RandomizedHillClimbing(hcp)
    train(rhc, exp_name, "RHC", "N=200", ef, 30000)

    sa = SimulatedAnnealing(1E5, .97, hcp)
    train(sa, exp_name, "SA", "N=200", ef, 30000)

    ga = StandardGeneticAlgorithm(300, 150, 15, gap)
    train(ga, exp_name, "GA", "N=200", ef, 30000)

    mimic = MIMIC(100, 25, pop)
    train(mimic, exp_name, "MIMIC", "N=200", ef, 15000)
