import csv
import os.path
import time
from operator import attrgetter

import numpy as np
import json
import logging
import operator

import pandas as pd
from kneed import KneeLocator
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
from sklearn_extra.cluster import KMedoids
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

    def __init__(self, dataset):

        self.num_rdf_instances = 0

        self._finish_build_matrix = None
        self._sim_matrix = None
        self._rdf_instances = []
        self._pending = []
        self._queue_out = Queue()
        self._similarities_computed = 0
        self._clustering_map = {}

        self._dataset_name = dataset
        self._partial_filename = "similarity_matrices/partial_{0}.csv".format(self._dataset_name)
        self.create_partial_file_if_not_exists()

        # Load pre-existing clustering configuration
        self._cls_map_filename = "semantic_maps/semantic_map_{0}.json".format(self._dataset_name)
        self.load_clustering_maps()

        # Load RDF Graph
        g = Graph()
        dataset_filename = "datasets/{0}.nt".format(self._dataset_name)
        g.parse(dataset_filename)
        logging.info("Loading Knowledge graph from: {0}".format(dataset_filename))

        # Print number of triples in document
        logging.info('Number of n-triples {0}'.format(len(g)))

        self._rdf_instances = []
        # find all subjects of RDF.type
        for person in g.subjects(RDF.type, None):
            self._rdf_instances.append(person)

        self.num_rdf_instances = len(self._rdf_instances)

    def compute_entity_similarity(self, lst_indexes):
        q_in = Queue()
        _ = [q_in.put((xx, y)) for (xx, y) in lst_indexes]
        while not q_in.empty():
            (i, j) = q_in.get()
            with lock:
                x = entity_similarity.similarity(self._rdf_instances[i], self._rdf_instances[j])
                self._queue_out.put((i, j, x))
            #    logging.info("Found:{}".format(x))
            #    time.sleep(2)

    def get_name(self, concept_url):
        items = concept_url.split("/")
        return items[-1]

    def compute_matrix_indices(self, index):
        j = index % self.num_rdf_instances
        i = int(index / self.num_rdf_instances)
        return i, j

    def build_semantic_distance_matrix(self, recompute_sim_matrix=False):
        n = self.num_rdf_instances
        logging.info("Number of named individuals:{0}".format(self.num_rdf_instances))

        # Write list of subject names
        out_names_filename = "similarity_matrices/names_{0}.txt".format(self._dataset_name)
        with open(out_names_filename, 'w') as fp:
            fp.write('\n'.join(self._rdf_instances))

        matrix_output = "similarity_matrices/{0}.txt".format(self._dataset_name)
        if os.path.isfile(matrix_output) and not recompute_sim_matrix:
            logging.info("Reading pre-computed semantic similarity matrix {}".format(matrix_output))
            self._sim_matrix = np.loadtxt(matrix_output, self._sim_matrix)
        else:
            # Initialize the matrix with 0
            self._sim_matrix = np.zeros((n, n))
            np.fill_diagonal(self._sim_matrix, 1)

            # Compute the similarity matrix for the RDF molecules
            logging.info("Computing similarity matrix")
            logging.info("Size of dataset:{}".format(self.num_rdf_instances))

            # Generate the map of tuples containing the indexes that should be split
            n_threads = 5
            logging.info("Number of threads: {}".format(n_threads))
            indexes_lst = []
            if os.path.isfile(self._partial_filename):
                existing_indexes = pd.read_csv(self._partial_filename)
            else:
                existing_indexes = pd.DataFrame()

            # Getting list of expected pair of similarities to compute
            expected_similarities = 0
            for i in range(0, n):
                for j in range(i, n):
                    if i != j:
                        indexes_lst.append((i, j))
                        expected_similarities += 1

            # Exclude already computed similarities from indexes to request
            logging.info("Removing already computed similarities")
            if expected_similarities != len(existing_indexes):
                df = existing_indexes.reset_index()
                for index, row in tqdm(df.iterrows(), total=df.shape[0]):
                    i = int(row["i"])
                    j = int(row["j"])
                    if (i, j) in indexes_lst:
                        indexes_lst.remove((i, j))
            else:
                # Nothing to do
                indexes_lst = []

            if len(indexes_lst) > 0:
                logging.info("Number of request to do:{}".format(len(indexes_lst)))

                # Splitting pending similarities among available threads
                delta = len(indexes_lst) // n_threads
                start = 0
                end = start + delta
                thread_list = []
                for x in range(0, n_threads):
                    t = Thread(target=self.compute_entity_similarity, args=(indexes_lst[start:end],))
                    thread_list.append(t)
                    t.start()
                    start = end
                    if x >= n_threads - 2:
                        end = len(indexes_lst)
                    else:
                        end = start + delta

                stop_event = Event()
                monitor_thread = Thread(target=self.progress_monitor, args=(stop_event,))
                monitor_thread.start()
                # _ = [t.start() for t in thread_list]
                _ = [t.join() for t in thread_list]
                stop_event.set()
                monitor_thread.join()

                existing_indexes = pd.read_csv(self._partial_filename)

            logging.info("All similarities already computed")
            # Iterate data frame and populate similarity matrix
            df = existing_indexes.reset_index()
            for index, row in df.iterrows():
                i = int(row["i"])
                j = int(row["j"])
                x = row["sim"]
                self._sim_matrix.itemset((i, j), x)
                self._sim_matrix.itemset((j, i), x)

            logging.info("Number of similarities computed:{0}".format(self._similarities_computed))
            logging.info("Number of expected similarities:{0}".format(len(indexes_lst)))

        np.savetxt(matrix_output, self._sim_matrix)
        logging.info("Adjacency matrix computed successfully!")
        logging.info("Min similarity found:{0}".format(self._sim_matrix.min()))
        logging.info("Max similarity found:{0}".format(self._sim_matrix.max()))
        return True

    def compute_centroids(self):
        # Compute clusters using centroid-based and distance-based clustering algorithms
        logging.info("Generating centroids with multiple clustering algorithms...")
        if self._sim_matrix is None:
            logging.error("Similarity matrix does not exist")
            self.build_semantic_distance_matrix()

        self.compute_affinity_propagation()
        self.compute_kmedoids()
        self.save_semantic_maps()
        logging.info("Centroids from multiple clustering algorithms already analyzed")

    def compute_affinity_propagation(self):
        if 'affinity' in self._clustering_map.keys():
            logging.info("Affinity propagation clustering already pre-computed for dataset {0}".format(self._dataset_name))
            return

        _semantic_map = {}
        self._clustering_map["affinity"] = {}
        logging.info("Computing affinity propagation clustering")
        tmp = np.ones(self.num_rdf_instances ** 2).reshape(self.num_rdf_instances, self.num_rdf_instances)
        distance_matrix = np.subtract(tmp, self._sim_matrix)
        np.fill_diagonal(distance_matrix, 0)
        # ap_cls_selection_lst contains a list or triples (clustering instance, silhouette index, preference)
        ap_cls_selection_lst = []
        # Inferring the optimal preference value based on the number of generated clusters based on silhouette index
        for preference in np.arange(0, 1, 0.1):
            logging.info("Inferring optimal preference value for affinity propagation")
            clustering = AffinityPropagation(random_state=50,
                                max_iter=800,
                                preference=preference,
                                affinity='precomputed').fit(distance_matrix)
            n_clusters = len(clustering.cluster_centers_indices_)

            # Valid values are 2 to n_samples - 1 (inclusive)
            if 2 < n_clusters < self.num_rdf_instances:
                ap_cls_selection_lst.append((clustering,
                                             metrics.silhouette_score(self._sim_matrix, clustering.labels_),
                                             preference))

        # Select cluster that maximizes silhouette index (2nd element)
        clustering_triple = max(ap_cls_selection_lst, key=lambda x: x[1])
        clustering = clustering_triple[0] # 1st element contains optimal clustering details
        logging.info("Analysing centroid information")
        for centroid_index in range(len(clustering.cluster_centers_indices_)):
            if centroid_index not in _semantic_map.keys():
                _semantic_map[int(clustering.cluster_centers_indices_[centroid_index])] = []

        logging.info("Generating semantic map")
        for i in range(0, len(clustering.labels_)):
            label = int(clustering.labels_[i])
            centroid_key = int(clustering.cluster_centers_indices_[label])
            if centroid_key != i:
                _semantic_map[centroid_key].append(i)

        # 3rd element in the triple contains the selected preference value for Affinity Propagation
        self._clustering_map["affinity"]["preference"] = clustering_triple[2]
        self._clustering_map["affinity"]["k"] = len(clustering.cluster_centers_indices_)
        self._clustering_map["affinity"]["quality"] = {}
        try:
            # Compute Intrinsic Measures to evaluate the cluster quality for Affinity Propagation
            silhouette = metrics.silhouette_score(self._sim_matrix, clustering.labels_)
            dwvies = davies_bouldin_score(self._sim_matrix, clustering.labels_)
            calinski = metrics.calinski_harabasz_score(self._sim_matrix, clustering.labels_)
            self._clustering_map["affinity"]["quality"]["silhouette"] = float(silhouette)
            self._clustering_map["affinity"]["quality"]["dwvies"] = float(dwvies)
            self._clustering_map["affinity"]["quality"]["calinski"] = float(calinski)
            logging.info("[Affinity Propagation] - Silhouette index: %0.3f" % silhouette)
            logging.info("[Affinity Propagation] - Davies Bouldin score: %0.3f" % dwvies)
            logging.info("[Affinity Propagation] - Calinski Harabasz score: %0.3f" % calinski)
        except ValueError as ve:
            self._clustering_map["affinity"]["quality"]["silhouette"] = -1
            self._clustering_map["affinity"]["quality"]["dwvies"] = 0.0
            self._clustering_map["affinity"]["quality"]["calinski"] = 0.0
            logging.error("Unable to compute clustery quality metrics:{}".format(ve))

        self._clustering_map["affinity"]["semantic_map"] = _semantic_map
        logging.info("Affinity propagation clustering finished")

    def compute_kmedoids(self):
        _semantic_map = {}
        _centroids = []
        logging.info("Computing kmedoids clustering")

        tmp = np.ones(self.num_rdf_instances ** 2).reshape(self.num_rdf_instances, self.num_rdf_instances)
        distance_matrix = np.subtract(tmp, self._sim_matrix)
        np.fill_diagonal(distance_matrix, 0)

        # Check if the number of clusters has been already computed
        # for the working dataset
        if 'kmedoids' not in self._clustering_map.keys() or 'k' not in self._clustering_map['kmedoids']:
            logging.info("Getting the best number of clusters using the Elbow Curve")
            nc = range(2, len(self._rdf_instances) // 2)
            kmedoids_lst_lst = [
                KMedoids(n_clusters=i, metric='precomputed', random_state=0, method='pam', init='k-medoids++') for i in
                nc]
            score = [kmedoids_lst_lst[i].fit(distance_matrix).inertia_ for i in range(0, len(kmedoids_lst_lst))]
            kl = KneeLocator(nc, score, curve="convex", direction="decreasing")
            logging.info("Number optimal of clusters found:{}".format(kl.knee))

            self._clustering_map["kmedoids"] = {}
            self._clustering_map["kmedoids"]["k"] = int(kl.knee)
            kmedoid = kmedoids_lst_lst[kl.knee - 2]
            self._clustering_map["kmedoids"]["inertia"] = float(kmedoid.inertia_)
        elif 'k' in self._clustering_map["kmedoids"].keys():
            k = int(self._clustering_map["kmedoids"]["k"])
            logging.info("Using pre-existing number of clusters {0} for dataset {1}".format(k, self._dataset_name))
            kmedoid = KMedoids(n_clusters=k, metric='precomputed', random_state=0, method='pam', init='k-medoids++').fit(distance_matrix)
        elif 'kmedoids' in self._clustering_map.keys():
            logging.info("Using pre-computed kmedoids clustering info")

        logging.info("Analysing centroid information")
        for i in range(0, len(kmedoid.medoid_indices_)):
            indx = int(kmedoid.medoid_indices_[i])
            if indx not in _semantic_map.keys():
                _semantic_map[indx] = []

        logging.info("Generating semantic map")
        for i in range(0, len(kmedoid.labels_)):
            centroid_key = int(kmedoid.medoid_indices_[kmedoid.labels_[i]])
            if centroid_key != i:
                _semantic_map[centroid_key].append(i)

        try:
            # Compute Intrinsic Measures to evaluate the cluster quality for Kmeans
            silhouette = metrics.silhouette_score(self._sim_matrix, kmedoid.labels_, metric="sqeuclidean")
            dwvies = davies_bouldin_score(self._sim_matrix, kmedoid.labels_)
            calinski = metrics.calinski_harabasz_score(self._sim_matrix, kmedoid.labels_)
            self._clustering_map["kmedoids"]["quality"] = {}
            self._clustering_map["kmedoids"]["quality"]["silhouette"] = silhouette
            self._clustering_map["kmedoids"]["quality"]["dwvies"] = dwvies
            self._clustering_map["kmedoids"]["quality"]["calinski"] = calinski
            logging.info("[kmedoid] - Silhouette index: %0.3f" % silhouette)
            logging.info("[kmedoid] - Davies Bouldin score: %0.3f" % dwvies)
            logging.info("[kmedoid] - Calinski Harabasz score: %0.3f" % calinski)
        except ValueError as ve:
            logging.error("Unable to compute clustery quality metrics:{}".format(ve))
            self._clustering_map["kmedoids"]["quality"]["dwvies"] = -1

        self._clustering_map["kmedoids"]["semantic_map"] = _semantic_map
        logging.info("Kmeans clustering finished")

    def infer_central_term(self):
        entity_features_tool = EntityFeatures()
        concept_tool = ConceptSimilarity(Taxonomy(DBpediaDataTransform()), 'models/dbpedia_type_ic.txt')

        for clustering_algorithm, semantic_map in self._clustering_map.items():
            # Obtain the list of types (RDF.RDF_TYPE) for each centroid
            logging.info("Inferring central term for {0} clustering algorithm".format(clustering_algorithm))
            concepts = {}
            shared_concepts_among_centroids = set()

            def add_centroid_concepts(centroid, centroid_type):
                if "wikidata" not in centroid_type and "entity" not in centroid_type:
                    concepts[self.get_name(centroid)].add(centroid_type)

            def add_ic(target_dict, c):
                target_dict[c] = concept_tool.concept_ic(c)

            for centroid, items in semantic_map['semantic_map'].items():
                if centroid not in concepts.keys():
                    c_url = self._rdf_instances[int(centroid)]
                    concepts[self.get_name(c_url)] = set()
                _ = [add_centroid_concepts(c_url, centroid_type) for centroid_type in entity_features_tool.type(c_url)]

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

            semantic_map['alpha'] = max(shared_concepts_dict.items(), key=operator.itemgetter(1))[0]
            logging.info("Central term of the semantic map:{0}".format(semantic_map['alpha']))
            self.save_semantic_maps()

    def assemble_semantic_map(self):
        if self._clustering_map is None:
            logging.error("Unable to read semantic map information")
            return

        for cls_algorithm, semantic_map in self._clustering_map.items():

            # First, add main term:
            alpha_indx = self.num_rdf_instances
            # Alpha concept in gray
            alpha = semantic_map['alpha']
            _graphic_semantic_map = Network('1024px', '870px')
            _graphic_semantic_map.add_node(alpha_indx, label=self.get_name(alpha), size=20, color='#808080')

            # Second, add all centroids
            for k, v in semantic_map['semantic_map'].items():
                idx = int(k)
                # Centroids in blue
                _graphic_semantic_map.add_node(idx, label=self.get_name(self._rdf_instances[idx]), size=15,
                                                color='#4da6ff')
                _graphic_semantic_map.add_edge(alpha_indx, idx)

            # Finally, connect add the rest of entity instances and connect them with its centroid
            for centroid, instance_lst in semantic_map['semantic_map'].items():
                centroid_index = int(centroid)
                for instance in instance_lst:
                    if instance != centroid_index:  # Avoid adding the centroid itself
                        # Non-centroids in white
                        _graphic_semantic_map.add_node(instance, label=self.get_name(self._rdf_instances[instance]),
                                                        size=15, color='#ffffff')
                        _graphic_semantic_map.add_edge(centroid_index, instance)

            _graphic_semantic_map.toggle_physics(True)
            _graphic_semantic_map.show("results/{0}_{1}.html".format(self._dataset_name, cls_algorithm))

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
        """
        Verifies if the CSV file that contains the already computed
        similarities exists or not. In case it does not exist, this method
        creates the file and establishes the schema (i, j, sim)
        """
        if not os.path.isfile(self._partial_filename):
            with open(self._partial_filename, 'w+', newline='') as partial_matrix:
                writer = csv.writer(partial_matrix)
                writer.writerow(["i", "j", "sim"])
        else:
            data = pd.read_csv(self._partial_filename)
            self._similarities_computed = len(data)
            logging.info("Partial file already exists and it contains:{}".format(self._similarities_computed))

    def progress_monitor(self, stop_event):
        """
        Monitors the progress and for safety every 2 min checks the whole
        progress of getting semantic similarities and flush the content of
        the queue ´_queue_out´ to the ´_partial_filename´
        """
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

    def load_clustering_maps(self):
        if not os.path.isfile(self._cls_map_filename):
            logging.warning("Unable to load pre-existing clustering semantic_maps for dataset {0}".format(self._dataset_name))
            return
        else:
            with open(self._cls_map_filename, 'r') as cls_settings_file:
                self._clustering_map = json.load(cls_settings_file)
                logging.info("Clustering semantic_maps loaded successfully for dataset {0}".format(self._dataset_name))
                logging.debug(self._cls_map_filename)

    def save_semantic_maps(self):
        with open(self._cls_map_filename, 'w') as cls_settings_file:
            json.dump(self._clustering_map, cls_settings_file)
            logging.info("Clustering semantic_maps for dataset {0} saved successfully".format(self._dataset_name))
