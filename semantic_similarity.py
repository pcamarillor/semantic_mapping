import csv
import os.path
import time

import numpy as np
import json
import logging
import operator

import pandas as pd
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
        self._partial_filename = None
        self._queue_out = Queue()
        self._similarities_computed = 0

    def compute_entity_similarity(self, lst_indexes):
        q_in = Queue()
        _ = [q_in.put((xx, y)) for (xx, y) in lst_indexes]
        # logging.info("q_in:{}".format(list(q_in.queue)))
        while not q_in.empty():
            (i, j) = q_in.get()
            try:
                #with lock:
                x = entity_similarity.similarity(self._rdf_instances[i], self._rdf_instances[j])
                # logging.info("processing ({0}, {1})".format(i, j))
                self._sim_matrix_boolean[i][j] = self._sim_matrix_boolean[j][i] = 0x01
                self._queue_out.put((i, j, x))
                #time.sleep(2)
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

    def build_similarity_matrix(self, dataset_name, read_matrix=False, progress=False):
        # Load RDF Graph
        g = Graph()
        g.parse("datasets/{0}.nt".format(dataset_name))
        self._dataset_name = dataset_name
        self._partial_filename = "similarity_matrices/partial_{}.csv".format(self._dataset_name)
        self.create_partial_file_if_not_exists()

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
            n_threads = 80
            logging.info("Number of threads: {}".format(n_threads))
            indexes_lst = []
            if os.path.isfile(self._partial_filename):
                existing_indexes = pd.read_csv(self._partial_filename)
            expected_similarities = 0

            # Exclude already computed similarities from indexes to request
            for i in range(0, n):
                for j in range(i, n):
                    if i != j:
                        indexes_lst.append((i, j))
                        expected_similarities += 1
            df = existing_indexes.reset_index()
            logging.info("Removing already computed similarities")
            for index, row in tqdm(df.iterrows(), total=df.shape[0]):
                i = int(row["i"])
                j = int(row["j"])
                if (i, j) in indexes_lst:
                    indexes_lst.remove((i, j))

            if len(indexes_lst) > 0:
                logging.info("Number of request to do:{}".format(len(indexes_lst)))
                start = 0
                delta = len(indexes_lst) // n_threads
                end = start + delta
                thread_list = []
                if progress:
                    self._finish_build_matrix = Event()
                    visualize_thread = Thread(target=self.print_sim_matrix)
                    visualize_thread.start()

                for x in range(0, n_threads):
                    t = Thread(target=self.compute_entity_similarity, args=(indexes_lst[start:end],))
                    thread_list.append(t)
                    start = end
                    if x >= n_threads - 2:
                        end = len(indexes_lst)
                    else:
                        end = start + delta

                stop_event = Event()
                monitor_thread = Thread(target=self.progress_monitor, args=(stop_event,))
                monitor_thread.start()
                _ = [t.start() for t in thread_list]
                _ = [t.join() for t in thread_list]
                stop_event.set()
                monitor_thread.join()

                if progress:
                    self._finish_build_matrix.set()
                    visualize_thread.join()
            else:
                logging.info("All similarities already computed")
                # Iterate data frame and populate similarity matrix
                df = existing_indexes.reset_index()
                for index, row in df.iterrows():
                    i = int(row["i"])
                    j = int(row["j"])
                    x = row["sim"]
                    self._sim_matrix.itemset((i, j), x)
                    self._sim_matrix.itemset((j, i), x)

                matrix_output = "similarity_matrices/{0}.txt".format(dataset_name)
                np.savetxt(matrix_output, self._sim_matrix)
                logging.info("Adjacency matrix computed successfully!")
                logging.info("Min similarity found:{0}".format(self._sim_matrix.min()))
                logging.info("Max similarity found:{0}".format(self._sim_matrix.max()))
                logging.info("Number of similarities computed:{0}".format(self._similarities_computed))
                logging.info("Number of expected similarities:{0}".format(len(indexes_lst)))
                return True
        return expected_similarities == self._similarities_computed

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
            n = self.num_rdf_instances
            while True:
                with lock:
                    for i in range(0, n):
                        output_lines[i] = "{:.2%}".format((self._sim_matrix_boolean[i].count(0x01) / n))
                if self._finish_build_matrix.is_set():
                    break
                time.sleep(0.5)

    def create_partial_file_if_not_exists(self):
        if not os.path.isfile(self._partial_filename):
            with open(self._partial_filename, 'w+', newline='') as partial_matrix:
                writer = csv.writer(partial_matrix)
                writer.writerow(["i", "j", "sim"])
        else:
            data = pd.read_csv(self._partial_filename)
            self._similarities_computed = len(data)
            logging.info("Partial file already exists and it contains:{}".format(self._similarities_computed))

    def progress_monitor(self, stop_event):
        while True:
            with lock:
                with open(self._partial_filename, 'a', newline='') as tmp_matrix:
                    writer = csv.writer(tmp_matrix)
                    while not self._queue_out.empty():
                        (i, j, x) = self._queue_out.get()
                        self._similarities_computed += 1
                        writer.writerow([i, j, x])
            logging.info("Similarities computed so far:{}".format(self._similarities_computed))
            if stop_event.is_set():
                break
            time.sleep(120)  # every 2 min

