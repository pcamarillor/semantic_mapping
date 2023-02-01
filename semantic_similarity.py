import time

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
from pyvis.network import Network
from tqdm import tqdm
from queue import Queue
from threading import Thread
from threading import Lock
lock = Lock()
from threading import Event
from reprint import output

entity_similarity = EntitySimilarity()

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
        self._finish_build_matrix = None
        self._sim_matrix = None
        self._sim_matrix_boolean = None
        self.num_rdf_instances = 0
        self._rdf_instances = []
        self._centroids = []
        self._alpha = None
        self._semantic_map = {}
        self._centroids_map = {}
        self._graphic_semantic_map = Network()
        self._dataset_name = None
        self._pending = []

    def compute_entity_similarity(self, lst_indexes, queue_out):
        q_in = Queue()
        _ = [q_in.put((xx, y)) for (xx, y) in lst_indexes]
        logging.info("q_in:{}".format(list(q_in.queue)))

        while not q_in.empty():
            (i, j) = q_in.get()
            try:
                with lock:
                    x = entity_similarity.similarity(self._rdf_instances[i], self._rdf_instances[j])
                    logging.info("processing ({0}, {1})".format(i, j))
                    # self._sim_matrix_boolean[i][j] = self._sim_matrix_boolean[j][i] = 0x01
                    queue_out.put((i, j, x))
            except RuntimeError as ue:
                logging.error("Error getting semantic similarity for {0}-{1}, {2}".format(
                    self.get_name(self._rdf_instances[i]),
                    self.get_name(self._rdf_instances[i]),
                    ue))

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
        n = self.num_rdf_instances
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
            self._sim_matrix = np.zeros((n, n))
            self._sim_matrix_boolean = [[0x00 for j in range(0, n)] for i in range(0, n)]
            for i in range(0, n):
                self._sim_matrix.itemset((i, i), 1.0)
                self._sim_matrix_boolean[i][i] = 0x01

            # Compute the similarity matrix for the RDF molecules
            logging.info("Computing similarity matrix")
            logging.info("Size of dataset:{}".format(self.num_rdf_instances))

            # Generate the map of tuples containing the indexes that should be split
            n_threads = 10
            indexes_lst = []
            for i in range(0, n):
                for j in range(i, n):
                    if i != j:
                        indexes_lst.append((i, j))

            start = 0
            delta = len(indexes_lst) // n_threads
            end = start + delta
            thread_list = []
            m = {}
            q_outs = []
            #self._finish_build_matrix = Event()
            #visualize_thread = Thread(target=self.print_sim_matrix)
            #visualize_thread.start()
            for x in range(0, n_threads):
                q_out = Queue()
                m[x] = indexes_lst[start:end]
                t = Thread(target=self.compute_entity_similarity, args=(indexes_lst[start:end], q_out,))
                thread_list.append(t)
                t.start()
                q_outs.append(q_out)
                start = end
                if x >= n_threads - 2:
                    end = len(indexes_lst)
                else:
                    end = start + delta

            # timeout = 60 * delta  # 60 seconds per call
            # time.sleep(15)
            for index, thread in enumerate(thread_list):
                logging.info("Main    : before joining thread %d.", index)
                thread.join()
                logging.info("Main    : thread %d done", index)

            #self._finish_build_matrix.set()
            #visualize_thread.join()

            n_similarities_computed = 0
            for q_o in q_outs:
                while not q_o.empty():
                    (i, j, x) = q_o.get()
                    n_similarities_computed += 1
                    self._sim_matrix.itemset((i, j), x)
                    self._sim_matrix.itemset((j, i), x)

            matrix_output = "similarity_matrices/{0}.txt".format(dataset_name)
            np.savetxt(matrix_output, self._sim_matrix)

        logging.info("Adjacency matrix computed successfully!")
        logging.info("Min similarity found:{0}".format(self._sim_matrix.min()))
        logging.info("Max similarity found:{0}".format(self._sim_matrix.max()))
        logging.info("Number of similarities computed:{0}".format(n_similarities_computed))
        logging.info("Number of expected similarities:{0}".format(len(indexes_lst)))
        return len(indexes_lst) == n_similarities_computed

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
        except ValueError as ve:
            logging.error("Unable to compute clustery quality metrics:{}".format(ve))

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

    def print_sim_matrix(self):
        with output(initial_len=self.num_rdf_instances, interval=0) as output_lines:
            while True:
                for i in range(0, self.num_rdf_instances):
                    output_lines[i] = "{}".format(self._sim_matrix_boolean[i])
                if self._finish_build_matrix.is_set():
                    break
                time.sleep(1)
