import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
import dwave.inspector

A = 1
C = 3.0
D = 5.5

alpha = 0.2
beta = 4.5


def main():
    # ------- Set up our graph -------

    # Create empty graph
    G = nx.Graph()

    # Add edges to the graph (also adds nodes)
    G.add_edges_from([(1, 2), (1, 4), (1, 5), (1, 6), (2, 3), (2, 4), (3, 4), (4, 5), (5, 6)])

    # ------- Set up our QUBO dictionary -------

    # Initialize our Q matrix
    Q = defaultdict(int)

    # Update Q matrix for every edge in the graph
    for i, j in G.edges:
        Q[(i, i)] += C * alpha - D
        Q[(j, j)] += A * beta + C * alpha - D
        Q[(i, j)] += -2 * C * alpha + D

    # ------- Run our QUBO on the QPU -------
    # Set up QPU parameters
    chainstrength = 8
    numruns = 10

    nx.draw(G, with_labels=True)
    plt.show()

    # Run the QUBO on the solver from your config file
    sampler = EmbeddingComposite(DWaveSampler())
    response = sampler.sample_qubo(Q,
                                   chain_strength=chainstrength,
                                   num_reads=numruns,
                                   label='Example - MUCGVCP')

    # ------- Print results to user -------
    print('-' * 60)
    print('{:>30s}{:^30s}'.format('Subset', 'Energy'))
    print('-' * 60)
    for sample, E in response.data(fields=['sample', 'energy']):
        S1 = [k for k, v in sample.items() if v == 1]
        print('{:>30s}{:^30s}'.format(str(S1), str(E)))

    dwave.inspector.show(response)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
