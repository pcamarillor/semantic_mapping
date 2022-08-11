import numpy as np
import json

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


def compute_sim(concept_a, concept_b):
    name_a = get_name(concept_a)
    name_b = get_name(concept_b)
    sim = EntitySimilarity()
    sim = sim.similarity(concept_a, concept_b)
    print("Similarity between {0} and {1} is: {2}".format(name_a, name_b, sim))


def get_name(concept_url):
    items = concept_url.split("/")
    return items[-1]


def build_similarity_matrix(dataset_name):
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

    print("Computing similarity matrix")
    # Compute the similarity matrix for the RDF molecules
    for i in tqdm(range(0, num_rdf_instances)):
        for j in tqdm(range(i, num_rdf_instances)):
            if i == j:
                sim_matrix.itemset((i, j), 1)
            else:
                sim_matrix.itemset((i, j), sim.similarity(rdf_instances[i], rdf_instances[j]))
                sim_matrix.itemset((j, i), sim_matrix.item((i, j)))

    matrix_output = "similarity_matrices/{0}.txt".format(dataset_name)
    np.savetxt(matrix_output, sim_matrix)
    print("Adjacency matrix computed successfully!")
    print("Min similarity found:{0}".format(sim_matrix.min()))
    print("Max similarity found:{0}".format(sim_matrix.max()))


def load_names(dataset_name):
    lst_names = []
    with open("similarity_matrices/names_{0}.txt".format(dataset_name), 'r') as fd:
        lst_names = fd.read().split('\n')

    for i in range(0, len(lst_names)):
        lst_names[i] = get_name(lst_names[i])

    return lst_names
