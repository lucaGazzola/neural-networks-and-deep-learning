import random
import network
import mnist_loader
import time

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

    return fitness


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

    if random.uniform(0, 1) < 0.5:
        offspring1[0] = individual2[0]
        offspring1[1] = individual2[1]
        offspring2[2] = individual1[2]
        offspring2[3] = individual1[3]
    else:
        offspring2[0] = individual1[0]
        offspring2[1] = individual1[1]
        offspring1[2] = individual2[2]
        offspring1[3] = individual2[3]

    return offspring1, offspring2


if __name__ == "__main__":

    population = random_population()

    # Simulate all of the generations.
    for generation in xrange(GENERATIONS):

        start_time = time.time()

        print "Generation %s... Random sample: '%s'" % (generation, population[0])

        ind_to_cross = random.sample(range(0, POP_SIZE - 1), CROSSOVERS * 2)

        if generation != 0:
            population = new_population
        new_population = []

        for i in xrange(CROSSOVERS):

            print "individuals to cross %s and %s: %s , %s" % (i*2, i*2+1, ind_to_cross[i*2], ind_to_cross[i*2+1])

            # Selection - random choice
            ind2 = population[ind_to_cross[i*2]]
            ind1 = population[ind_to_cross[i*2+1]]

            # Crossover
            print "crossover on %s and %s (individual %s and %s)" % (ind1, ind2, ind_to_cross[i*2+1], ind_to_cross[i*2])
            ind1, ind2 = crossover(ind1, ind2)

            # Mutate and add back into the population.
            new_population.append(mutate(ind1))
            new_population.append(mutate(ind2))

        for i in xrange(len(population)):

            if i not in ind_to_cross:
                print "mutating and adding individual %s : %s" % (i, population[i])
                new_population.append(mutate(population[i]))

        new_population_evaluated = []

        for individual in new_population:
            fitness_val = fitness(individual)
            pair = (individual, fitness_val)
            new_population_evaluated.append(pair)

        total_time = time.time() - start_time
        print "Generation Completed in %s" % total_time
        for individual in new_population:
            print individual

    # Display the highest-ranked string after all generations have been iterated
    # over. This will be the closest string to the OPTIMAL string, meaning it
    # will have the smallest fitness value. Finally, exit the program.
    fittest_string = population[0]
    minimum_fitness = fitness(population[0])

    for individual in population:
        ind_fitness = fitness(individual)
        if ind_fitness <= minimum_fitness:
            fittest_string = individual
            minimum_fitness = ind_fitness

    print "Fittest: %s" % fittest_string
    exit(0)
