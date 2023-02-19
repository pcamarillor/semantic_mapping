import os
import sys

sys.path.insert(0, os.path.abspath('../../semantic_mapping'))

import semantic_similarity
from memory_profiler import profile


@profile
def call_sim():
    os.chdir("../")
    test_instance = semantic_similarity.SemanticMap()
    test_instance.build_semantic_distance_matrix("movies_scifi", False)
    test_instance.compute_centroids()
    test_instance.infer_central_term()
    test_instance.assemble_semantic_map()
    os.chdir("./test")


if __name__ == '__main__':
    call_sim()
