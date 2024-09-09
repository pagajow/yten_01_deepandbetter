import random
import pandas as pd

# Define items with (weight, value) pairs
items = [(50, 5), (94, 76), (31, 88), (71, 84), (91, 15), (19, 47), (12, 44), (1, 49), 
         (10, 7), (16, 19), (3, 78), (57, 69), (27, 14), (99, 40), (52, 58), 
         (93, 79), (50, 48), (39, 20), (26, 39), (40, 14)]
items = pd.DataFrame(items, columns=['weight', 'value'])
items.index.name = "items"

items.T

# Set the total weight limit
totalWeightLimit = 200

# Generate a genome, where each item is included or not
genome = [random.randint(0, 5) == 0 for _ in range(len(items))]
genome

def createPopulation(populationSize: int):
    return [[random.randint(0, 5) == 0 for _ in range(len(items))] for _ in range(populationSize)]

def fitness(genes: list):
    totalWeight = 0
    totalValue = 0
    for idx, gene in enumerate(genes):
        if gene:
            totalWeight += items["weight"][idx]
            totalValue += items["value"][idx]

    if totalWeight > totalWeightLimit:
        return 0
    else:
        return totalValue

def selectWinners(population: list):
    populationValues = []
    for genes in population:
        value = fitness(genes=genes)
        if value > 0:
            populationValues.append((value, genes))

    return [genes for value, genes in sorted(populationValues, key=lambda x: x[0], reverse=True)]

def selectBest(winners: list, population: list, percentage=0.2):
    limit = int(percentage * len(population))
    if len(winners) > limit:
        best = winners[:limit]
    else:
        best = winners
    return best

def crossover(genes1: list, genes2: list):
    crossoverPoint = random.randint(1, len(genes1)-1)
    newGenes = genes1[:crossoverPoint] + genes2[crossoverPoint:]
    return newGenes

def mutate(genes: list):
    newGenes = list(genes)
    idx = random.randint(0, len(genes)-1)
    newGenes[idx] = not bool(genes[idx])
    return newGenes

def mutateRandomly(genes: list, probability: int = 100):
    if random.randint(0, probability) == 0:
        return mutate(genes)
    else:
        return list(genes)

def nextGeneration(population: list):
    newPopulation = []
    winners = selectWinners(population)

    if len(winners) > 0:
        winners = selectBest(winners, population, 0.2)
        for _ in range(len(population)):
            newGenes = crossover(random.choice(winners), random.choice(winners))
            newPopulation.append(mutateRandomly(newGenes))
    else:
        newPopulation = createPopulation(len(population))

    return newPopulation

population = createPopulation(1000)
for i in range(20):
    newPopulation = nextGeneration(population)
    population = newPopulation

results = pd.concat([pd.DataFrame(population).mean() * 100, items], axis=1)
results.columns = ["selection_rate"] + list(items.columns)
results.index.name = items.index.name
results.T

results["selection_rate"].plot.bar()
