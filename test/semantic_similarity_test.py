import os
import sys
import unittest

sys.path.insert(0, os.path.abspath('../../semantic_mapping'))
import semantic_similarity


class SemanticSimilarityTest(unittest.TestCase):
    def test_build_similarity(self):
        os.chdir("../")
        test_instance = semantic_similarity.MultiExample()
        test_instance.build_similarity_matrix("star_wars")


if __name__ == '__main__':
    unittest.main()
