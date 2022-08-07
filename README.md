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
