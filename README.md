# Semantic mapping for Knowledge Graph visualization

The main goal of this project is to automate the process of creating semanitc maps to visualize knowledge graphs.

# Sematch installation

In order to compute the semantic similarity for each node in the KG, we will use the [sematch](https://github.com/gsi-upm/sematch) python library.

## Sematch installation

The current released version of `sematch` library does not support Python 3.9. To use a python 3+ compatible version, we install the code from branch ´py3compat´.


    git clone https://github.com/gsi-upm/sematch.git
    cd sematch
    git checkout py3compat
    git pull origin py3compat
    python setup.py install

# Unit Test execution

In order to test the semantic mapping process, the test suite
under `test` folder contains a set of Unit Tests that read some
pre-computed sematic similarity matrices and generates the semantic
map for them.

To run this suite simply go to `test` folder and run the following
command:

    python -m pytest

It will run all UTs and generate the corresponding semantic maps.

## Example of UTs output

    ========== test session starts =====================================================
    platform darwin -- Python 3.9.7, pytest-6.2.4, py-1.10.0, pluggy-0.13.1
    rootdir: /Users/pcamarillor/Code/owl_python/semantic_mapping/test
    plugins: anyio-2.2.0
    collected 4 items                                                                                                            

    semantic_similarity_test.py ....                                                                                       [100%]

    ================================================ 4 passed in 91.32s (0:01:31) ================================================