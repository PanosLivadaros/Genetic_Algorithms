import random
import numpy


def calculate_fitness(sol):
    fit = 0
    # Calculate fitness of solution, using a penalty of 2 points every time 2 neighbouring nodes have the same colour
    for node in sol:
        for neighbour in node[1]:
            if sol[neighbour - 1][0] == node[0]:
                fit += 1
    return fit


def calculate_exp_reprod_rate(fit):
    return 1 - (fit / total) / (mean / total)


def check_compatibility(parents, probabilities):
    parent_a = random.choices(parents, weights=probabilities, k=1)[0]
    parent_b = random.choices(parents, weights=probabilities, k=1)[0]
    while parent_a == parent_b:
        parent_a = random.choices(parents, weights=probabilities, k=1)[0]
        parent_b = random.choices(parents, weights=probabilities, k=1)[0]
    return parent_a, parent_b


def crossover(parent_a, parent_b, c_point):
    return parent_a[:c_point] + parent_b[-(13 - c_point):]


def mutate(child, probability):
    if random.choices([True, False], weights=[probability, 1 - probability])[0]:
        position = random.randint(0, 12)
        colour = random.choice(colours)
        while child[position][0] == colour:
            position = random.randint(0, 12)
            colour = random.choice(colours)
        child[position][0] = colour
    return child


def sort_solutions(sol, fit):
    paired_list = list(zip(fit, sol))
    paired_list.sort()
    sorted_fit, sorted_sol = zip(*paired_list)
    return list(sorted_sol), list(sorted_fit)


nodes = [["", [2, 3, 12]],
         ["", [1, 3, 4, 11, 12]],
         ["", [1, 2, 4, 5, 6, 7]],
         ["", [2, 3, 7, 9]],
         ["", [3, 6]],
         ["", [3, 5, 7]],
         ["", [3, 4, 6, 8, 9]],
         ["", [7, 9, 10]],
         ["", [4, 7, 8, 10, 11]],
         ["", [8, 9, 11, 13]],
         ["", [2, 9, 10, 12, 13]],
         ["", [1, 2, 11, 13]],
         ["", [10, 11, 12]]]
colours = ["blue", "red", "green", "yellow"]

# Parameters
population = 500
generations = 100
elitism = 1
elite_solutions = 0.01
mutation_rate = 0.001
initial_gen = []

# Create initial population
for _ in range(population):
    initial_gen += [[[random.choice(colours), *node[1:]] for node in nodes]]
current_gen = initial_gen

for i in range(generations):
    # Calculate fitness of each solution
    fitness = []
    for p in range(population):
        fitness += [calculate_fitness(current_gen[p])]

    total = numpy.sum(fitness)
    mean = numpy.mean(fitness)

    # Calculate Expected Reproduction Rate of each solution
    exp_reprod_rate = []
    for f in fitness:
        exp_reprod_rate += [calculate_exp_reprod_rate(f)]

    # Normalize Expected Reproduction Rate probabilities
    min_prob = min(exp_reprod_rate)
    max_prob = max(exp_reprod_rate)
    exp_reprod_rate = (exp_reprod_rate - min_prob) / (max_prob - min_prob)
    exp_reprod_rate /= sum(exp_reprod_rate)

    # Print the characteristics of the current generation
    print("Generation number:", i)
    for k in range(population):
        print("Solution:", current_gen[k], "with score:", fitness[k], "and probability to reproduce:", exp_reprod_rate[k])

    # Create next generation, starting by deciding if the Elites will survive
    next_gen = []
    temp_fitness = fitness
    current_gen, fitness = sort_solutions(current_gen, fitness)
    exp_reprod_rate, temp_fitness = sort_solutions(exp_reprod_rate, temp_fitness)
    for e in range(int(population * elite_solutions)):
        if random.choices([True, False], weights=[elitism, 1-elitism])[0]:
            next_gen += [current_gen[e]]

    # Randomly decide which atoms of the current generation will survive to the next generation
    while len(next_gen) < int(population * 0.4):
        survives = random.choice(current_gen)
        while survives in next_gen:
            survives = random.choice(current_gen)
        next_gen += [survives]

    # Produce offspring
    reserved = len(next_gen)
    for p in range(int((population - reserved) / 2)):
        chosen_parent_a, chosen_parent_b = check_compatibility(current_gen, exp_reprod_rate)
        crossover_point = random.randint(1, 11)
        # Crossover and mutate
        next_gen += [mutate(crossover(chosen_parent_a, chosen_parent_b, crossover_point), mutation_rate)]
        next_gen += [mutate(crossover(chosen_parent_b, chosen_parent_a, crossover_point), mutation_rate)]

    # Next generation now becomes current generation for the next epoch
    current_gen = next_gen

# Calculate fitness for the last generation
fitness = []
for p in range(population):
    fitness += [calculate_fitness(current_gen[p])]

total = numpy.sum(fitness)
mean = numpy.mean(fitness)

# Calculate Expected Reproduction Rate of each solution of the last generation
exp_reprod_rate = []
for f in fitness:
    exp_reprod_rate += [calculate_exp_reprod_rate(f)]

# Normalize the Expected Reproduction Rate probabilities
min_prob = min(exp_reprod_rate)
max_prob = max(exp_reprod_rate)
exp_reprod_rate = (exp_reprod_rate - min_prob) / (max_prob - min_prob)
exp_reprod_rate /= sum(exp_reprod_rate)

# Print the characteristics of the current generation
print("Generation number:", generations)
for k in range(population):
    print("Solution:", current_gen[k], "with score:", fitness[k], "and probability to reproduce:", exp_reprod_rate[k])

# Sort the solutions by their fitness values and get the best solution
sorted_solutions, sorted_fitness = sort_solutions(current_gen, fitness)
print("The best solution is:", sorted_solutions[0], "with a score of:", sorted_fitness[0])
