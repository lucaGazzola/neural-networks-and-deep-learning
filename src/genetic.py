import random
import network
import mnist_loader
import time

POP_SIZE = 10
GENERATIONS = 50
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
mutation_probability = 0.1


def random_population():

    pop = []
    for i in xrange(POP_SIZE):
        mid_neurons_number = random.randint(10, 40)
        epochs = random.randint(10, 30)
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
    network.Network([784, mid_neurons_number, 10])

    start_time = time.time()
    correct_results, n_tests = individual[0].SGD(training_data, epochs, mini_batch_size, learning_rate, test_data=test_data)
    total_time = time.time() - start_time

    fitness = n_tests - correct_results + total_time

    return fitness


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

if __name__ == "__main__":

    population = random_population()

    # Simulate all of the generations.
    for generation in xrange(GENERATIONS):
        print "Generation %s... Random sample: '%s'" % (generation, population[0])
        weighted_population = []

        # Add individuals and their respective fitness levels to the weighted
        # population list. This will be used to pull out individuals via certain
        # probabilities during the selection phase. Then, reset the population list
        # so we can repopulate it after selection.
        for individual in population:
            fitness_val = fitness(individual)

            # Generate the (individual,fitness) pair, taking in account whether or
            # not we will accidently divide by zero.
            if fitness_val == 0:
                pair = (individual, 1.0)
            else:
                pair = (individual, 1.0 / fitness_val)

            weighted_population.append(pair)

        population = []

        # Select two random individuals, based on their fitness probabilites, cross
        # their genes over at a random point, mutate them, and add them back to the
        # population for the next iteration.
        for _ in xrange(POP_SIZE / 2):
            # Selection
            ind1 = weighted_choice(weighted_population)
            ind2 = weighted_choice(weighted_population)

            # Crossover
            ind1, ind2 = crossover(ind1, ind2)

            # Mutate and add back into the population.
            population.append(mutate(ind1))
            population.append(mutate(ind2))

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

    print "Fittest String: %s" % fittest_string
    exit(0)
