import random
import conv_network
import mnist_loader
import time
import argparse

# script usage: python genetic.py <population size> <generations number> <number of crossovers per generation> <mutaiton probability>

'''
genetic algorithm parameters and training/testing dataset loading
'''

# population size
POP_SIZE = 5

# total number of generations
GENERATIONS = 5

# number of crossovers per generation
CROSSOVERS = 2

# mutation probability
mutation_probability = 0.1

# load the dataset
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()


'''
initialize a random population
'''


def random_population():

    pop = []
    for i in xrange(POP_SIZE):

        # randomly initialize the network parameters
        mid_neurons_number = random.randint(50, 100)
        kernel = random.randint(2, 7)
        pool = random.randint(2, 5)

        # add the individual to the population
        pop.append([mid_neurons_number, kernel, pool])
    return pop


'''
calculate the fitness of an individual
'''


def fitness(individual):

    # set up the network parameters
    mid_neurons_number = individual[0]
    kernel = individual[1]
    pool = individual[2]

    # instantiate a network classifying 784px images describing digits from 0 to 9 (10 possibilities)
    net = conv_network.CNNetwork()

    start_time = time.time()

    # use statistical gradient descent algorithm to train the network with the dataset
    accuracy = net.run(kernel,pool,mid_neurons_number)

    total_time = time.time() - start_time

    # the individual fitness is calculated as the wrongly classified examples + the total training time
    fitness = 1 - accuracy

    return fitness


'''
apply mutation to an individual in the population
each individual is mutated with *mutation_probability* probability
'''


def mutate(individual):

    # number of neurons
    if random.uniform(0, 5) < mutation_probability:
        if random.uniform(0, 1) < 0.5:
            individual[0] += 1
        else:
            individual[0] -= 1

    # kernel size
    if random.uniform(0, 1) < mutation_probability:
        if random.uniform(0, 1) < 0.5:
            individual[1] += 1
        else:
            individual[1] -= 1

    # pool size
    if random.uniform(0,1) < mutation_probability:
        if random.uniform(0,1) < 0.5:
            individual[2] += 1
        else:
            individual[2] -= 1

    return individual


'''
apply crossover to two individual to generate two new ones
the crossover type is 3-point
'''


def crossover(individual1, individual2):

    offspring1 = individual1
    offspring2 = individual2

    offspring1[1] = individual2[1]
    offspring2[1] = individual1[1]

    return offspring1, offspring2


'''
apply elitism to the population
if the elite individual is better that the fittest one in the population,
a randomly chosen individual in the population is substituted with the elite
'''


def elitism(population, elite):

    sorted_population = sorted(population, key=lambda tup: tup[1])
    if elite[1] < sorted_population[0][1]:
        sorted_population[random.randint(0, len(population)-1)] = elite
    return sorted_population


if __name__ == "__main__":


    # in the user supplied optional arguments, parse them and assign the GA parameters accordingly
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

    # open the report file
    f = open('report_pop' + str(POP_SIZE) + '_gen' + str(GENERATIONS) + '.txt', 'w')

    # initialize random population
    population = random_population()

    # interate for the number of generations specified
    for generation in xrange(GENERATIONS):

        start_time = time.time()

        print "Generation %s'" % (generation)
        f.write("Generation %s\n" % (generation))

        # sort the populations and store the elite individual
        # if the generation is not the first the fitness is already calculated at the end of the iteration
        if generation != 0:

            population = new_population
            sorted_evaluated_population = sorted(new_population_evaluated, key=lambda tup: tup[1])
            elite = sorted_evaluated_population[0]

        # if this is the first generation, calculate the fitness here
        else:
            
            new_population_evaluated = []

            for individual in population:
                fitness_val = fitness(individual)
                pair = (individual, fitness_val)
                new_population_evaluated.append(pair)

            sorted_evaluated_population = sorted(new_population_evaluated, key=lambda tup: tup[1])
            elite = sorted_evaluated_population[0]
        
        new_population = []

        # apply *CROSSOVERS* crossovers to the population, individuals with the highest fitness are selected for crossover
        for i in xrange(CROSSOVERS):

            print "individuals to cross %s and %s: %s , %s" % (i*2, i*2+1, sorted_evaluated_population[i*2], sorted_evaluated_population[i*2+1])
            f.write("individuals to cross %s and %s: %s , %s \n" % (i*2, i*2+1, sorted_evaluated_population[i*2], sorted_evaluated_population[i*2+1]))

            ind2 = sorted_evaluated_population[i*2][0]
            ind1 = sorted_evaluated_population[i*2+1][0]
            
            ind1, ind2 = crossover(ind1, ind2)

            # mutate the offspring and add back into the population.
            new_population.append(mutate(ind1))
            new_population.append(mutate(ind2))

        # mutate the remaining individuals that were not crossovered
        for i in xrange(len(population)):

            if i not in xrange(CROSSOVERS*2):
                print "mutating and adding individual %s : %s" % (i, population[i])
                f.write("mutating and adding individual %s : %s \n" % (i, population[i]))
                new_population.append(mutate(population[i]))

        new_population_evaluated = []

        # calculate the fitness of the population
        for individual in new_population:
            fitness_val = fitness(individual)
            f.write("fitness of individual %s : %s\n" % (individual, fitness_val))
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
        ind_fitness = fitness(individual)
        if ind_fitness <= minimum_fitness:
            fittest_string = individual
            minimum_fitness = ind_fitness

    print "Fittest: %s" % fittest_string
    f.write("Fittest: %s \n" % fittest_string)
    f.close()
    exit(0)
