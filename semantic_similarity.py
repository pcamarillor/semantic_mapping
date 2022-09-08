from multiprocessing import Pool
import SharedArray as sa

import numpy as np
import json
import logging
import operator

from sematch.semantic.similarity import EntitySimilarity
from sematch.semantic.sparql import EntityFeatures
from sematch.semantic.graph import DBpediaDataTransform, Taxonomy
from sematch.semantic.similarity import ConceptSimilarity
from rdflib import Graph
from rdflib.namespace import RDF
from sklearn.cluster import AffinityPropagation
from sklearn import metrics
from sklearn.metrics import davies_bouldin_score
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
        self._sim_matrix = None
        self.similarity_shared_array = None
        self._SHM_SIMILARITIES = "similarities"
        self.num_rdf_instances = 0
        self._rdf_instances = []
        self._centroids = []
        self._alpha = None

    def compute_entity_similarity(self, entity_a, entity_b, i, j):
        entity_similarity = EntitySimilarity()
        entity_similarity = entity_similarity.similarity(entity_a, entity_b)
        shared_similarities = sa.attach("shm://{0}".format(self._SHM_SIMILARITIES))
        shared_similarities[self.num_rdf_instances * i + j] = entity_similarity

    def get_name(self, concept_url):
        items = concept_url.split("/")
        return items[-1]

    def compute_matrix_indices(self, index):
        j = index % self.num_rdf_instances
        i = int(index / self.num_rdf_instances)
        return i, j

    def build_similarity_matrix(self, dataset_name):
        g = Graph()
        g.parse("datasets/{0}.nt".format(dataset_name))

        # Print number of triples in document
        logging.info('Number of n-triples {0}'.format(len(g)))

        self._rdf_instances = []
        # find all subjects of RDF.type
        for person in g.subjects(RDF.type, None):
            self._rdf_instances.append(person)

        self.num_rdf_instances = len(self._rdf_instances)
        logging.info("Number of fictional chars:{0}".format(self.num_rdf_instances))

        # Write list of subject names
        out_names_filename = "similarity_matrices/names_{0}.txt".format(dataset_name)
        with open(out_names_filename, 'w') as fp:
            fp.write('\n'.join(self._rdf_instances))

        # Initialize the matrix with 0
        self._sim_matrix = np.zeros((self.num_rdf_instances, self.num_rdf_instances))

        # Initialize shared array that will contain similarity results
        self.similarity_shared_array = sa.create("shm://{0}".format(self._SHM_SIMILARITIES),
                                                 self.num_rdf_instances ** 2)

        logging.info("Computing similarity matrix")
        logging.info(self.num_rdf_instances)
        # Compute the similarity matrix for the RDF molecules
        with Pool(processes=10) as pool:
            multi_res = []
            for i in range(0, self.num_rdf_instances):
                for j in range(i, self.num_rdf_instances):
                    if i != j:
                        multi_res.append(pool.apply_async(self.compute_entity_similarity,
                                                          (self._rdf_instances[i], self._rdf_instances[j], i, j)))

            # make a single worker sleep for 60 secs
            try:
                [res.get(timeout=60) for res in multi_res]
            except RuntimeError:
                logging.error("Something unexpected occurred")
                sa.delete(self._SHM_SIMILARITIES)

            for i in range(0, self.num_rdf_instances):
                for j in range(i, self.num_rdf_instances):
                    if i == j:
                        self._sim_matrix.itemset((i, j), 1.0)
                    elif i != j and self._sim_matrix.item((i, j)) == 0:
                        self._sim_matrix.itemset((i, j), self.similarity_shared_array[self.num_rdf_instances * i + j])
                        self._sim_matrix.itemset((j, i), self._sim_matrix.item((i, j)))

        matrix_output = "similarity_matrices/{0}.txt".format(dataset_name)
        np.savetxt(matrix_output, self._sim_matrix)
        logging.info("Adjacency matrix computed successfully!")
        logging.info("Min similarity found:{0}".format(self._sim_matrix.min()))
        logging.info("Max similarity found:{0}".format(self._sim_matrix.max()))
        sa.delete(self._SHM_SIMILARITIES)

    def compute_centroids(self):
        # Compute clusters using Affinity Propagation algorithm
        logging.info("Compute affinity propagation clustering...")
        clustering = AffinityPropagation(random_state=10, max_iter=800).fit(self._sim_matrix)
        for centroid in clustering.cluster_centers_indices_:
            self._centroids.append(self._rdf_instances[centroid])

        # Compute Intrinsic Measures to evaluate the cluster quality for Affinity Propagation
        logging.info("Silhouette index: %0.3f" % metrics.silhouette_score(self._sim_matrix, clustering.labels_,
                                                                          metric="sqeuclidean"))
        logging.info("Davies Bouldin score: %0.3f" % davies_bouldin_score(self._sim_matrix, clustering.labels_))
        logging.info(
            "Calinski Harabasz score: %0.3f" % metrics.calinski_harabasz_score(self._sim_matrix, clustering.labels_))

    def infer_central_term(self):
        entity_features_tool = EntityFeatures()
        concept_tool = ConceptSimilarity(Taxonomy(DBpediaDataTransform()), 'models/dbpedia_type_ic.txt')
        concepts = {}
        shared_concepts_among_centroids = set()

        def add_centroid_concepts(centroid, centroid_type):
            concepts[self.get_name(centroid)].add(centroid_type)

        def add_ic(target_dict, c):
            target_dict[c] = concept_tool.concept_ic(c)

        # Obtain the list of types (RDF.RDF_TYPE) for each centroid
        for centroid in self._centroids:
            if centroid not in concepts.keys():
                concepts[self.get_name(centroid)] = set()
            [add_centroid_concepts(centroid, centroid_type) for centroid_type in entity_features_tool.type(centroid)]

        # Get the intersection of centroid types
        for centroid, centroid_types in concepts.items():
            if len(shared_concepts_among_centroids) == 0:
                shared_concepts_among_centroids = centroid_types
            else:
                # Get the intersection for all shared types among centroids
                shared_concepts_among_centroids = shared_concepts_among_centroids & centroid_types

        # Once obtained the intersection, proceed to obtain the IC for each shared type and get the max
        shared_concepts_dict = dict()
        [add_ic(shared_concepts_dict, c) for c in shared_concepts_among_centroids]

        self._alpha = max(shared_concepts_dict.items(), key=operator.itemgetter(1))[0]
        logging.info("Central term of the semantic map:{0}".format(self._alpha))

    def load_names(self, dataset_name):
        lst_names = []
        with open("similarity_matrices/names_{0}.txt".format(dataset_name), 'r') as fd:
            lst_names = fd.read().split('\n')

        for i in range(0, len(lst_names)):
            lst_names[i] = self.get_name(lst_names[i])

        return lst_names
