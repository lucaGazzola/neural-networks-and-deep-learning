import random
import network
import mnist_loader
import time
import sys
import argparse

# script usage: python genetic.py <population size> <generations number> <number of crossovers per generation> <mutaiton probability>

POP_SIZE = 10
GENERATIONS = 10
CROSSOVERS = 2
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
mutation_probability = 0.1


def random_population():

    pop = []
    for i in xrange(POP_SIZE):
        mid_neurons_number = random.randint(10, 40)
        epochs = random.randint(5, 10)
        mini_batch_size = random.randint(10, 30)
        learning_rate = random.uniform(1.0, 4.0)
        pop.append([mid_neurons_number, epochs, mini_batch_size, learning_rate])
    return pop


def fitness(individual):

    mid_neurons_number = individual[0]
    epochs = individual[1]
    mini_batch_size = individual[2]
    learning_rate = individual[3]

    # 784px images describing digits from 0 to 9 (10 possibilities)
    net = network.Network([784, mid_neurons_number, 10])

    start_time = time.time()
    correct_results, n_tests = net.SGD(training_data, epochs, mini_batch_size, learning_rate, test_data=test_data)
    total_time = time.time() - start_time

    fitness = n_tests - correct_results + total_time

    return fitness, correct_results, total_time


def weighted_choice(items):

    weight_total = sum((item[1] for item in items))
    n = random.uniform(0, weight_total)
    for item, weight in items:
        if n < weight:
            return item
        n = n - weight
    return item


def mutate(individual):

    # number of neurons
    if random.uniform(0,1) < mutation_probability:
        if random.uniform(0,1) < 0.5:
            individual[0] += 1
        else:
            individual[0] -= 1

    # number of epochs
    if random.uniform(0,1) < mutation_probability:
        if random.uniform(0,1) < 0.5:
            individual[1] += 1
        else:
            individual[1] -= 1

    # mini batch size
    if random.uniform(0,1) < mutation_probability:
        if random.uniform(0,1) < 0.5:
            individual[2] += 1
        else:
            individual[2] -= 1

    # learning rate
    if random.uniform(0, 1) < mutation_probability:
        if random.uniform(0, 1) < 0.5:
            individual[3] += 0.1
        else:
            individual[3] -= 0.1

    return individual


def crossover(individual1, individual2):

    offspring1 = individual1
    offspring2 = individual2

    offspring1[1] = individual2[1]
    offspring1[3] = individual2[3]
    offspring2[1] = individual1[1]
    offspring2[3] = individual1[3]

    return offspring1, offspring2


def elitism(population, elite):

    sorted_population = sorted(population, key=lambda tup: tup[1])
    if elite[1] < sorted_population[0][1]:
        sorted_population[0] = elite
    return sorted_population


if __name__ == "__main__":

    f = open('report.txt', 'w')

    parser = argparse.ArgumentParser()
    parser.add_argument("--popsize", help="population size")
    parser.add_argument("--gen", help="number of generations")
    parser.add_argument("--cross", help="number of crossovers per generation")
    parser.add_argument("--mut_prob", help="mutation probability")
    args = parser.parse_args()

    if args.popsize:
        POP_SIZE = int(args.popsize)

    if args.gen:
        GENERATIONS = int(args.gen)

    if args.cross:
        CROSSOVERS = int(args.cross)

    if args.mut_prob:
        mutation_probability = float(args.mut_prob)

    population = random_population()

    # Simulate all of the generations.
    for generation in xrange(GENERATIONS):

        start_time = time.time()

        print "Generation %s... Random sample: '%s'" % (generation, population[0])
        f.write("Generation %s... Random sample: '%s'\n" % (generation, population[0]))

        if generation != 0:

            population = new_population
            sorted_evaluated_population = sorted(new_population_evaluated, key=lambda tup: tup[1])
            elite = sorted_evaluated_population[0]

        else:
            
            new_population_evaluated = []

            for individual in population:
                fitness_val = fitness(individual)
                pair = (individual, fitness_val)
                new_population_evaluated.append(pair)

            sorted_evaluated_population = sorted(new_population_evaluated, key=lambda tup: tup[1])
            elite = sorted_evaluated_population[0]
        
        new_population = []

        for i in xrange(CROSSOVERS):

            print "individuals to cross %s and %s: %s , %s" % (i*2, i*2+1, sorted_evaluated_population[i*2], sorted_evaluated_population[i*2+1])
            f.write("individuals to cross %s and %s: %s , %s \n" % (i*2, i*2+1, sorted_evaluated_population[i*2], sorted_evaluated_population[i*2+1]))

            ind2 = sorted_evaluated_population[i*2][0]
            ind1 = sorted_evaluated_population[i*2+1][0]
            
            ind1, ind2 = crossover(ind1, ind2)

            # Mutate and add back into the population.
            new_population.append(mutate(ind1))
            new_population.append(mutate(ind2))

        for i in xrange(len(population)):

            if i not in xrange(CROSSOVERS*2):
                print "mutating and adding individual %s : %s" % (i, population[i])
                f.write("mutating and adding individual %s : %s \n" % (i, population[i]))
                new_population.append(mutate(population[i]))

        new_population_evaluated = []

        for individual in new_population:
            fitness_val, correct_results, training_time = fitness(individual)
            f.write("fitness of individual %s : %s , correct_results: %s , network training time: %s\n" % (individual, fitness_val, correct_results, training_time))
            pair = (individual, fitness_val)
            new_population_evaluated.append(pair) 

        new_population_evaluated = elitism(new_population_evaluated, elite)    

        total_time = time.time() - start_time
        print "Generation Completed in %s" % total_time
        f.write("Generation Completed in %s \n" % total_time)
        for individual in new_population:
            print individual

    fittest_string = population[0]
    minimum_fitness = fitness(population[0])

    for individual in population:
        ind_fitness, correct_results, training_time = fitness(individual)
        if ind_fitness <= minimum_fitness:
            fittest_string = individual
            minimum_fitness = ind_fitness

    print "Fittest: %s" % fittest_string
    f.write("Fittest: %s \n" % fittest_string)
    f.close()
    exit(0)
