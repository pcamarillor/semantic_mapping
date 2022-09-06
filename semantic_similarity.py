from multiprocessing import Pool
import SharedArray as sa

import numpy as np
import json

from sematch.semantic.similarity import EntitySimilarity
from sematch.semantic.sparql import EntityFeatures
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
    """
    Returns the exemplar's index based on the centers and items. Exemplar should be
    an index in items
    """
    for item in items:
        if item in centers:
            return item
    return -1


class SemanticMap:

    def __init__(self):
        self.origin_array = None
        self.origin_concepts = None
        self._SHM_SIMILARITIES = "similarities"
        self._SHM_CONCEPTS = "concepts"
        self.num_rdf_instances = 0

    def compute_sim(self, entity_a, entity_b, i, j):
        entity_similarity = EntitySimilarity()
        entity_features = EntityFeatures()
        concepts_entity_a = [c for c in entity_features.type(entity_a)]
        concepts_entity_b = [c for c in entity_features.type(entity_b)]

        entity_similarity = entity_similarity.similarity(entity_a, entity_b)
        shared_similarities = sa.attach("shm://{0}".format(self._SHM_SIMILARITIES))
        shared_similarities[self.num_rdf_instances * i + j] = entity_similarity

        shared_concepts = sa.attach("shm://{0}".format(self._SHM_CONCEPTS))
        shared_concepts[i] = concepts_entity_a[0]

    def get_name(self, concept_url):
        items = concept_url.split("/")
        return items[-1]

    def compute_matrix_indices(self, index):
        j = index % self.num_rdf_instances
        i = int(index / self.num_rdf_instances)
        return i, j

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

        self.num_rdf_instances = len(rdf_instances)
        print("Number of fictional chars:{0}".format(self.num_rdf_instances))

        # Write list of subject names
        out_names_filename = "similarity_matrices/names_{0}.txt".format(dataset_name)
        with open(out_names_filename, 'w') as fp:
            fp.write('\n'.join(rdf_instances))

        # Initialize the matrix with 0
        sim_matrix = np.zeros((self.num_rdf_instances, self.num_rdf_instances))

        # Initialize shared array that will contain similarity results
        self.origin_array = sa.create("shm://{0}".format(self._SHM_SIMILARITIES), self.num_rdf_instances ** 2)
        self.origin_concepts = sa.create("shm://{0}".format(self._SHM_CONCEPTS), self.num_rdf_instances)

        print("Computing similarity matrix")
        print(self.num_rdf_instances)
        # Compute the similarity matrix for the RDF molecules
        with Pool(processes=10) as pool:
            multi_res = []
            for i in range(0, self.num_rdf_instances):
                for j in range(i, self.num_rdf_instances):
                    if i != j:
                        multi_res.append(pool.apply_async(self.compute_sim, (rdf_instances[i], rdf_instances[j], i, j)))

            # make a single worker sleep for 60 secs
            [res.get(timeout=60) for res in multi_res]
            for i in range(0, self.num_rdf_instances):
                for j in range(i, self.num_rdf_instances):
                    if i == j:
                        sim_matrix.itemset((i, j), 1.0)
                    elif i != j and sim_matrix.item((i, j)) == 0:
                        sim_matrix.itemset((i, j), self.origin_array[self.num_rdf_instances * i + j])
                        sim_matrix.itemset((j, i), sim_matrix.item((i, j)))

        matrix_output = "similarity_matrices/{0}.txt".format(dataset_name)
        np.savetxt(matrix_output, sim_matrix)
        print("Adjacency matrix computed successfully!")
        print("Min similarity found:{0}".format(sim_matrix.min()))
        print("Max similarity found:{0}".format(sim_matrix.max()))
        sa.delete(self._SHM_SIMILARITIES)

    def load_names(self, dataset_name):
        lst_names = []
        with open("similarity_matrices/names_{0}.txt".format(dataset_name), 'r') as fd:
            lst_names = fd.read().split('\n')

        for i in range(0, len(lst_names)):
            lst_names[i] = self.get_name(lst_names[i])

        return lst_names
