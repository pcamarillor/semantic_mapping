import os
import sys
import unittest

sys.path.insert(0, os.path.abspath('../../semantic_mapping'))

import semantic_similarity


class SemanticSimilarityTest(unittest.TestCase):
    def test_build_similarity_star_wars(self):
        os.chdir("../")
        test_instance = semantic_similarity.SemanticMap()
        test_instance.build_similarity_matrix("star_wars", True)
        test_instance.compute_centroids()
        test_instance.infer_central_term()
        test_instance.assemble_semantic_map()
        os.chdir("./test")

    def test_build_similarity_got(self):
        os.chdir("../")
        test_instance = semantic_similarity.SemanticMap()
        test_instance.build_similarity_matrix("Got_Characters", True)
        test_instance.compute_centroids()
        test_instance.infer_central_term()
        test_instance.assemble_semantic_map()
        os.chdir("./test")

    def test_build_similarity_books(self):
        os.chdir("../")
        test_instance = semantic_similarity.SemanticMap()
        test_instance.build_similarity_matrix("books", True)
        test_instance.compute_centroids()
        test_instance.infer_central_term()
        test_instance.assemble_semantic_map()
        os.chdir("./test")

    def test_build_similarity_movies(self):
        os.chdir("../")
        test_instance = semantic_similarity.SemanticMap()
        test_instance.build_similarity_matrix("movies_scifi", False)
        test_instance.compute_centroids()
        test_instance.infer_central_term()
        test_instance.assemble_semantic_map()
        os.chdir("./test")

    def test_build_similarity_fict_chars(self):
        os.chdir("../")
        test_instance = semantic_similarity.SemanticMap()
        test_instance.build_similarity_matrix("fictional_chars", False)
        test_instance.compute_centroids()
        test_instance.infer_central_term()
        test_instance.assemble_semantic_map()
        os.chdir("./test")



if __name__ == '__main__':
    unittest.main()
