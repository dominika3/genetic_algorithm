import random
from statistics import mean
import numpy as np


global distance_matrix


def load_data_from_file(filename: str) -> None:

    with open(filename, 'r') as data:

        lines = data.readlines()
        num_of_cities = int(lines[0].strip())
        matrix = [[0 for _ in range(num_of_cities)] for _ in range(num_of_cities)]

        for idx, line in enumerate(lines[1:]):
            row = list(map(int, line.split()))
            for i in range(len(row)):
                matrix[idx][i] = row[i]
                matrix[i][idx] = row[i]

        global distance_matrix
        distance_matrix = matrix


def random_route(num_of_cities: int) -> list[int]:
    individual = [i for i in range(num_of_cities)]
    random.shuffle(individual)
    return individual


def random_population(population_size: int, num_of_cities: int) -> list[list[int]]:
    return [random_route(num_of_cities) for _ in range(population_size)]


def calculate_distance_to_next(position: int, route: list[int]) -> int:
    if position == len(route) - 1:
        return distance_matrix[route[-1]][route[0]]
    else:
        return distance_matrix[route[position]][route[position + 1]]


def calculate_individual_distance(route: list[int]) -> int:
    return sum(list(map(lambda city_index: calculate_distance_to_next(city_index, route), range(len(route)))))


def evaluate_population(population: list[list[int]]) -> list[list[int]]:
    return list(map(lambda route: [route, calculate_individual_distance(route)], population))


def print_shortest_and_average_distance(evaluated_population: list[list, int]) -> None:

    distances = list(map(lambda i: i[1], evaluated_population))
    shortest_distance = min(distances)
    average_distance = mean(distances)

    print('Shortest distance in the initial generation : ' + str(shortest_distance) + '\n' +
          'Average distance in the initial generation: ' + str(average_distance))


def skewed_distribution(num_of_elements, sigma_scale, mean_value_position):
    sigma = num_of_elements * sigma_scale
    mean_value = num_of_elements * mean_value_position
    return abs(np.random.normal(mean_value, sigma, num_of_elements).astype(int)) % (num_of_elements - 1)


def generate_new_population(old_population: list[list[int], int]) -> list[list[int]]:

    old_population_sorted = list(map(lambda x: x[0], sorted(old_population, key=lambda x: x[1])))
    population_size = len(old_population)
    indexes = skewed_distribution(population_size, 0.25, 0.15)

    return list(map(lambda i: list(old_population_sorted[i]), indexes))


def crossover_population(population: list[list[int]], crossover_probability: float) -> list[list[int]]:
    even_indexes = range(0, len(population), 2)
    samples_count = int(crossover_probability * len(population) / 2)
    random_even_indexes_to_crossover = random.sample(even_indexes, samples_count)

    for i in random_even_indexes_to_crossover:
        population[i:i+2][:] = pmx(population[i], population[i+1])

    return population


def pmx(mom_chromosome, dad_chromosome):

    cut_point_1 = random.randint(0, len(mom_chromosome) - 1)
    cut_point_2 = random.randint(cut_point_1, len(mom_chromosome) - 1)

    child1 = mom_chromosome[:]
    child2 = dad_chromosome[:]

    for idx in range(cut_point_1, cut_point_2):
        swap_gene(child1, dad_chromosome, idx)
        swap_gene(child2, mom_chromosome, idx)

    return child1, child2


def swap_gene(child, parent, idx):
    to_be_replaced_index = child.index(parent[idx])
    child[to_be_replaced_index] = child[idx]
    child[idx] = parent[idx]


def mutate_population(population: list[list[int]], mutation_probability: float) -> list[list[int]]:
    chromosomes_to_mutate = random.sample(population, int(mutation_probability * len(population)))

    for chromosome in chromosomes_to_mutate:
        cut_point = random.randint(0, len(chromosome) - 1)
        part_to_inverse = chromosome[cut_point:]
        chromosome[len(chromosome) - len(part_to_inverse):] = part_to_inverse[::-1]

    return population


def __main__():
    load_data_from_file('a10.txt')
    num_of_cities = len(distance_matrix)
    population_size = 10
    mutation_probability = 0.4
    crossover_probability = 0.2
    population = random_population(population_size, num_of_cities)
    evaluated_population = evaluate_population(population)
    print_shortest_and_average_distance(evaluated_population)
    for _ in range(10):
        new_population = generate_new_population(evaluated_population)
        new_population = mutate_population(new_population, mutation_probability)
        new_population = crossover_population(new_population, crossover_probability)

        evaluated_population = evaluate_population(new_population)

    print_shortest_and_average_distance(evaluated_population)


if __name__ == "__main__":
    __main__()

