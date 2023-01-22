from multiprocessing import Pool
from multiprocessing import cpu_count
from multiprocessing import shared_memory
import SharedArray as sa

import numpy as np
import json
import logging
import operator
import array

from sematch.semantic.similarity import EntitySimilarity
from sematch.semantic.sparql import EntityFeatures
from sematch.semantic.graph import DBpediaDataTransform, Taxonomy
from sematch.semantic.similarity import ConceptSimilarity
from rdflib import Graph
from rdflib.namespace import RDF
from sklearn.cluster import AffinityPropagation
from sklearn import metrics
from sklearn.metrics import davies_bouldin_score
from pyvis.network import Network
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
        self.similarity_np_array = None
        self._sim_matrix = None
        self.similarity_shared_array = None
        self._SHM_SIMILARITIES = "similarities"
        self.num_rdf_instances = 0
        self._rdf_instances = []
        self._centroids = []
        self._alpha = None
        self._semantic_map = {}
        self._centroids_map = {}
        self._graphic_semantic_map = Network()
        self._entity_similarity = EntitySimilarity()
        self._dataset_name = None

    def compute_entity_similarity(self, entity_a, entity_b, i, j):
        entity_similarity = self._entity_similarity.similarity(entity_a, entity_b)
        shared_similarities = shared_memory.SharedMemory(name=self.similarity_shared_array.name)
        shared_similarities.buf[self.num_rdf_instances * i + j] = entity_similarity
        shared_similarities.close()
        # shared_similarities = sa.attach("shm://{0}".format(self._SHM_SIMILARITIES))
        # shared_similarities[self.num_rdf_instances * i + j] = entity_similarity

    def get_name(self, concept_url):
        items = concept_url.split("/")
        return items[-1]

    def compute_matrix_indices(self, index):
        j = index % self.num_rdf_instances
        i = int(index / self.num_rdf_instances)
        return i, j

    def build_similarity_matrix(self, dataset_name, read_matrix=False):
        # Load RDF Graph
        g = Graph()
        g.parse("datasets/{0}.nt".format(dataset_name))
        self._dataset_name = dataset_name

        # Print number of triples in document
        logging.info('Number of n-triples {0}'.format(len(g)))

        self._rdf_instances = []
        # find all subjects of RDF.type
        for person in g.subjects(RDF.type, None):
            self._rdf_instances.append(person)

        self.num_rdf_instances = len(self._rdf_instances)
        logging.info("Number of named individuals:{0}".format(self.num_rdf_instances))

        # Write list of subject names
        out_names_filename = "similarity_matrices/names_{0}.txt".format(dataset_name)
        with open(out_names_filename, 'w') as fp:
            fp.write('\n'.join(self._rdf_instances))

        if read_matrix:
            matrix_output = "similarity_matrices/{0}.txt".format(dataset_name)
            self._sim_matrix = np.loadtxt(matrix_output, self._sim_matrix)
        else:
            # Initialize the matrix with 0
            self._sim_matrix = np.zeros((self.num_rdf_instances, self.num_rdf_instances))
            self.similarity_np_array = np.zeros(self.num_rdf_instances ** 2)
            self.similarity_shared_array = shared_memory.SharedMemory(create=True, size=self.similarity_np_array.nbytes)

            # Initialize shared array that will contain similarity results
            #try:
            #    self.similarity_shared_array = sa.create("shm://{0}".format(self._SHM_SIMILARITIES),
            #                                             self.num_rdf_instances ** 2)
            #except FileExistsError:
            #    sa.delete(self._SHM_SIMILARITIES)
            #    self.similarity_shared_array = sa.create("shm://{0}".format(self._SHM_SIMILARITIES),
            #                                             self.num_rdf_instances ** 2)

            logging.info("Computing similarity matrix")
            logging.info(self.num_rdf_instances)
            # Compute the similarity matrix for the RDF molecules
            with Pool(cpu_count()) as pool:
                multi_res = []
                for i in range(0, self.num_rdf_instances):
                    for j in range(i, self.num_rdf_instances):
                        if i != j:
                            multi_res.append(pool.apply_async(self.compute_entity_similarity,
                                                              (self._rdf_instances[i], self._rdf_instances[j], i, j)))

                # make a single worker sleep for 60 secs
                try:
                    [res.get(timeout=60) for res in multi_res]
                except TimeoutError:
                    logging.error("Something unexpected occurred")
                    # sa.delete(self._SHM_SIMILARITIES)

                for i in range(0, self.num_rdf_instances):
                    for j in range(i, self.num_rdf_instances):
                        if i == j:
                            self._sim_matrix.itemset((i, j), 1.0)
                        elif i != j and self._sim_matrix.item((i, j)) == 0:
                            self._sim_matrix.itemset((i, j),
                                                     self.similarity_shared_array.buf[self.num_rdf_instances * i + j])
                            self._sim_matrix.itemset((j, i), self._sim_matrix.item((i, j)))

            matrix_output = "similarity_matrices/{0}.txt".format(dataset_name)
            np.savetxt(matrix_output, self._sim_matrix)
            self.similarity_shared_array.close()
            self.similarity_shared_array.unlink()
            # sa.delete(self._SHM_SIMILARITIES)

        logging.info("Adjacency matrix computed successfully!")
        logging.info("Min similarity found:{0}".format(self._sim_matrix.min()))
        logging.info("Max similarity found:{0}".format(self._sim_matrix.max()))

    def compute_centroids(self):
        # Compute clusters using Affinity Propagation algorithm
        logging.info("Compute affinity propagation clustering...")
        centroid_label = 0
        clustering = AffinityPropagation(random_state=50, max_iter=800, preference=1, affinity='precomputed').fit(
            self._sim_matrix)
        # clustering = AffinityPropagation(random_state=50, max_iter=800).fit(self._sim_matrix)
        for centroid_index in clustering.cluster_centers_indices_:
            self._centroids.append(self._rdf_instances[centroid_index])
            self._centroids_map[centroid_label] = centroid_index
            centroid_label += 1

        for indx in range(0, len(clustering.labels_)):
            label = clustering.labels_[indx]
            if label not in self._semantic_map.keys():
                self._semantic_map[label] = []
            self._semantic_map[label].append(indx)

        try:
            # Compute Intrinsic Measures to evaluate the cluster quality for Affinity Propagation
            silhouette = metrics.silhouette_score(self._sim_matrix, clustering.labels_, metric="sqeuclidean")
            dwvies = davies_bouldin_score(self._sim_matrix, clustering.labels_)
            calinski = metrics.calinski_harabasz_score(self._sim_matrix, clustering.labels_)
            logging.info("Silhouette index: %0.3f" % silhouette)
            logging.info("Davies Bouldin score: %0.3f" % dwvies)
            logging.info("Calinski Harabasz score: %0.3f" % calinski)
        except ValueError:
            logging.error("Unexcepted error")

    def infer_central_term(self):
        entity_features_tool = EntityFeatures()
        concept_tool = ConceptSimilarity(Taxonomy(DBpediaDataTransform()), 'models/dbpedia_type_ic.txt')
        concepts = {}
        shared_concepts_among_centroids = set()

        def add_centroid_concepts(centroid, centroid_type):
            if "wikidata" not in centroid_type and "entity" not in centroid_type:
                concepts[self.get_name(centroid)].add(centroid_type)

        def add_ic(target_dict, c):
            target_dict[c] = concept_tool.concept_ic(c)

        # Obtain the list of types (RDF.RDF_TYPE) for each centroid
        for centroid in self._centroids:
            if centroid not in concepts.keys():
                concepts[self.get_name(centroid)] = set()
            _ = [add_centroid_concepts(centroid, centroid_type) for centroid_type in
                 entity_features_tool.type(centroid)]

        # Get the intersection of centroid types
        for centroid, centroid_types in concepts.items():
            if len(shared_concepts_among_centroids) == 0:
                shared_concepts_among_centroids = centroid_types
            else:
                # Get the intersection for all shared types among centroids
                shared_concepts_among_centroids = shared_concepts_among_centroids & centroid_types

        # Once obtained the intersection, proceed to obtain the IC for each shared type and get the max
        shared_concepts_dict = dict()
        _ = [add_ic(shared_concepts_dict, c) for c in shared_concepts_among_centroids]

        self._alpha = max(shared_concepts_dict.items(), key=operator.itemgetter(1))[0]
        logging.info("Central term of the semantic map:{0}".format(self._alpha))

    def assemble_semantic_map(self):
        # First, add main term:
        alpha_indx = self.num_rdf_instances
        # Alpha concept in gray
        self._graphic_semantic_map.add_node(alpha_indx, label=self.get_name(self._alpha), size=20, color='#808080')

        # Second, add all centroids
        for k, v in self._centroids_map.items():
            idx = int(v)
            # Centroids in blue
            self._graphic_semantic_map.add_node(idx, label=self.get_name(self._rdf_instances[idx]), size=15,
                                                color='#4da6ff')
            self._graphic_semantic_map.add_edge(alpha_indx, idx)

        # Finally, connect add the rest of entity instances and connect them with its centroid
        for centroid, instance_lst in self._semantic_map.items():
            centroid_index = int(self._centroids_map[centroid])
            for instance in instance_lst:
                if instance != centroid_index:  # Avoid adding the centroid itself
                    # Non-centroids in white
                    self._graphic_semantic_map.add_node(instance, label=self.get_name(self._rdf_instances[instance]),
                                                        size=15, color='#ffffff')
                    self._graphic_semantic_map.add_edge(centroid_index, instance)

        # self._graphic_semantic_map.show_buttons(filter_=["nodes"])
        self._graphic_semantic_map.toggle_physics(True)
        self._graphic_semantic_map.show("{0}.html".format(self._dataset_name))

    def load_names(self, dataset_name):
        lst_names = []
        with open("similarity_matrices/names_{0}.txt".format(dataset_name), 'r') as fd:
            lst_names = fd.read().split('\n')

        for i in range(0, len(lst_names)):
            lst_names[i] = self.get_name(lst_names[i])

        return lst_names
