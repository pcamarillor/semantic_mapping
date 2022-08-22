from multiprocessing import Pool
from multiprocessing import Array
from multiprocessing import shared_memory

import numpy as np
import json
import time

from sematch.semantic.similarity import EntitySimilarity
from rdflib import Graph
from rdflib.namespace import RDF
from tqdm import tqdm


class Exemplar:
    def __init__(self, resourceURI, datanum, children):
        self.id = resourceURI
        self.datum = datanum
        self.children = children

    def toJSON(self):
        return json.dump(self, default=lambda o: o.__dict__)


def find_exemplar(centers, items):
    '''
    Returns the exemplar's index based on the centers and items. Exemplar should be
    an index in items
    '''
    for item in items:
        if item in centers:
            return item
    return -1


class MultiExample:

    def __init__(self):
        self.b = None

    def compute_sim(self, concept_a, concept_b, i, j):
        sim = EntitySimilarity()
        sim = sim.similarity(concept_a, concept_b)
        self.b[i][j] = sim
        print(self.b[i][j])


    def get_name(self, concept_url):
        items = concept_url.split("/")
        return items[-1]

    def build_similarity_matrix(self, dataset_name):
        g = Graph()
        sim = EntitySimilarity()
        g.parse("datasets/{0}.nt".format(dataset_name))

        # Print number of triples in document
        print('Number of n-triples {0}'.format(len(g)))

        rdf_instances = []
        # find all subjects of RDF.type
        for person in g.subjects(RDF.type, None):
            rdf_instances.append(person)

        num_rdf_instances = len(rdf_instances)
        print("Number of fictional chars:{0}".format(num_rdf_instances))

        # Write list of subject names
        out_names_filename = "similarity_matrices/names_{0}.txt".format(dataset_name)
        with open(out_names_filename, 'w') as fp:
            fp.write('\n'.join(rdf_instances))

        # Initialize the matrix with 0
        sim_matrix = np.zeros((num_rdf_instances, num_rdf_instances))
        #sim_matrix = np.matrix([[0 for i in range(num_rdf_instances)], [0 for i in range(num_rdf_instances)]])
        shm = shared_memory.SharedMemory(create=True, size=sim_matrix.nbytes)
        self.b = np.ndarray(sim_matrix.shape, dtype=sim_matrix.dtype, buffer=shm.buf)

        print("Computing similarity matrix")
        print(num_rdf_instances)
        # Compute the similarity matrix for the RDF molecules
        with Pool(processes=10) as pool:
            multi_res = []
            for i in tqdm(range(0, num_rdf_instances)):
                for j in tqdm(range(i, num_rdf_instances)):
                    if i == j:
                        sim_matrix.itemset((i, j), 1)
                    else:
                        multi_res.append(pool.apply_async(self.compute_sim, (rdf_instances[i], rdf_instances[j], i, j)))

            # make a single worker sleep for 10 secs
            [res.get(timeout=60) for res in multi_res]
            #print(self.b)

        #            sim_matrix.itemset((i, j), res.get())
        #            sim_matrix.itemset((j, i), sim_matrix.item((i, j)))

        matrix_output = "similarity_matrices/{0}.txt".format(dataset_name)
        np.savetxt(matrix_output, self.b)
        print("Adjacency matrix computed successfully!")
        print("Min similarity found:{0}".format(self.b.min()))
        print("Max similarity found:{0}".format(self.b.max()))
        del self.b
        shm.close()
        shm.unlink()


def load_names(dataset_name):
    lst_names = []
    with open("similarity_matrices/names_{0}.txt".format(dataset_name), 'r') as fd:
        lst_names = fd.read().split('\n')

    for i in range(0, len(lst_names)):
        lst_names[i] = get_name(lst_names[i])

    return lst_names
