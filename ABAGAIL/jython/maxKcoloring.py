"""
Max K Coloring Test. Inspired from Java file

-Athira Nair
"""
import random

import opt.ga.MaxKColorFitnessFunction as MaxKColorFitnessFunction
import opt.ga.Vertex as Vertex

import dist.DiscreteDependencyTree as DiscreteDependencyTree
import dist.DiscretePermutationDistribution as DiscretePermutationDistribution
import dist.DiscreteUniformDistribution as DiscreteUniformDistribution
import opt.ga.UniformCrossOver as UniformCrossOver
import dist.Distribution as Distribution

import opt.DiscreteChangeOneNeighbor as DiscreteChangeOneNeighbor
import opt.EvaluationFunction as EvaluationFunction
import opt.GenericHillClimbingProblem as GenericHillClimbingProblem
import opt.HillClimbingProblem as HillClimbingProblem
import opt.NeighborFunction as NeighborFunction
import opt.RandomizedHillClimbing as RandomizedHillClimbing
import opt.SimulatedAnnealing as SimulatedAnnealing
import opt.SwapNeighbor as SwapNeighbor
import opt.ga.CrossoverFunction as CrossoverFunction
import opt.ga.DiscreteChangeOneMutation as DiscreteChangeOneMutation
import opt.ga.SingleCrossOver as SingleCrossOver
import opt.ga.GenericGeneticAlgorithmProblem as GenericGeneticAlgorithmProblem
import opt.ga.GeneticAlgorithmProblem as GeneticAlgorithmProblem
import opt.ga.MutationFunction as MutationFunction
import opt.ga.StandardGeneticAlgorithm as StandardGeneticAlgorithm
import opt.ga.SwapMutation as SwapMutation
import opt.prob.GenericProbabilisticOptimizationProblem as GenericProbabilisticOptimizationProblem
import opt.prob.MIMIC as MIMIC
import opt.prob.ProbabilisticOptimizationProblem as ProbabilisticOptimizationProblem
import shared.FixedIterationTrainer as FixedIterationTrainer
import opt.example.ContinuousPeaksEvaluationFunction as ContinuousPeaksEvaluationFunction
from array import array
import shared.ConvergenceTrainer as ConvergenceTrainer
import csv

def train(alg_func, alg_name, ef, iters):
    ef.resetFunctionEvaluationCount()
    fit = ConvergenceTrainer(alg_func)
    FILE_NAME = alg_name + "-continous-peaks.csv"
    OUTPUT_FILE = os.path.join("data/" + "continous-peaks", FILE_NAME)
    with open(OUTPUT_FILE, "wb") as results:
        writer = csv.writer(results, delimiter=',')
        writer.writerow(["iters", "fevals", "fitness"])
        for i in range(iters):
            fit.train()
            # print str(i) + ", " + str(ef.getFunctionEvaluations()) + ", " + str(ef.value(alg_func.getOptimal()))
            writer.writerow([i, ef.getFunctionEvaluations() - i, ef.value(alg_func.getOptimal())])

    print alg_name + ": " + str(ef.value(alg_func.getOptimal()))
    print "Function Evaluations: " + str(ef.getFunctionEvaluations()-iters)
    print "Iters: " + str(iters)
    print "####"

N = 200 # number of vertices
L = 25 # L adjacent nodes per vertex
K = 15 # K possible colors

random.seed(N*L)

# create the random velocity
vertices = [None] * N
for i in range(0, N):
    vertex = Vertex()
    vertices[i] = vertex
    vertex.setAdjMatrixSize(L)
    for j in range(0, L):
        vertex.getAadjacencyColorMatrix().append(random.randint(1, N * L))

# for i in range(0,N):
#     vertex = vertices[i]
#     print vertex.getAadjacencyColorMatrix()

# for rhc, sa, and ga we use a permutation based encoding
ef = MaxKColorFitnessFunction(vertices)
odd = DiscretePermutationDistribution(K)
nf = SwapNeighbor()
mf = SwapMutation()
cf = SingleCrossOver()
hcp = GenericHillClimbingProblem(ef, odd, nf)
gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)

df = DiscreteDependencyTree(.1)
pop = GenericProbabilisticOptimizationProblem(ef, odd, df)

rhc = RandomizedHillClimbing(hcp)
fit = FixedIterationTrainer(rhc, 20000)
fit.train()
print("RHC: %0.03f" % ef.value(rhc.getOptimal()))
print(ef.foundConflict())

print("============================")

sa = SimulatedAnnealing(1E12, .1, hcp)
fit = FixedIterationTrainer(sa, 20000)
fit.train()
print("SA: %0.03f" % ef.value(sa.getOptimal()))
print(ef.foundConflict())

print("============================")

ga = StandardGeneticAlgorithm(200, 10, 60, gap)
fit = FixedIterationTrainer(ga, 50)
fit.train()
print("GA: %0.03f" % ef.value(ga.getOptimal()))
print(ef.foundConflict())

print("============================")

mimic = MIMIC(200, 100, pop)
fit = FixedIterationTrainer(mimic, 5)
fit.train()
print("MIMIC: %0.03f" % ef.value(mimic.getOptimal()))
print(ef.foundConflict())

