import os
import sys
import unittest

sys.path.insert(0, os.path.abspath('../../semantic_mapping'))
import semantic_similarity


class SemanticSimilarityTest(unittest.TestCase):
    def test_build_similarity(self):
        os.chdir("../")
        semantic_similarity.build_similarity_matrix("fictional_chars")


if __name__ == '__main__':
    unittest.main()
