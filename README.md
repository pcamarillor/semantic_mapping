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

# Dataset generation

In order to test these modules, we need to provide a Knowledge Graph encoded as a list of N-triples (.nt files). To generate
these datasets, we used and recommend to use the [DBPedia endpoint](https://dbpedia.org/sparql/) to run SPARQL queries. This
endpoint offers the capability to generate N-triples files.

In the following subsections, we present the SPARQL queiries used to generate the datasets used to validate the functionality
of the summarization process to visualize Knwoledge Graphs.


## SCI-FI-MOVIES.NT



    PREFIX yago: <http://dbpedia.org/class/yago/>
    PREFIX dbp: <https://dbpedia.org/property/>
    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
    CONSTRUCT { 
        ?movie rdf:type  yago:Movie106613686 .
    }
    WHERE {
        ?movie rdf:type yago:Movie106613686 ;
                   dbo:wikiPageWikiLink  dbc:American_science_fiction_action_films ;
                   dbo:gross ?gross .
        FILTER(?gross > 8E8)
    }


## FANTASY-NOVELS.NT


    PREFIX yago: <http://dbpedia.org/class/yago/>
    PREFIX dbp: <https://dbpedia.org/property/>
    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
    CONSTRUCT { 
        ?book rdf:type  yago:FWikicatFantasyNovels .
    }
    WHERE {
        ?book rdf:type yago:WikicatFantasyNovels .
        FILTER ( dbp:published > "2000-1-1"^^xsd:date  )
    }

## CITIES.NT


    CONSTRUCT { 
    ?city rdf:type  yago:City108524735 .
    }
    WHERE {
    ?city rdf:type yago:City108524735 ;
                dbo:populationTotal ?populationTotal .
        FILTER(?populationTotal > 5E6 )
    }

## DISEASES.NT


    PREFIX yago: <http://dbpedia.org/class/yago/>
    CONSTRUCT { 
        ?disease rdf:type  yago:Disease114070360 .
    }
    WHERE {
       ?disease  rdf:type yago:Disease114070360 ;
                dbo:wikiPageWikiLink dbr:Infectious_disease .
    }

## DRUGS.NT


    PREFIX yago: <http://dbpedia.org/class/yago/>
    CONSTRUCT { 
        ?med rdf:type yago:Compound114818238 .
    }
    WHERE {
       ?disease  rdf:type yago:Disease114070360 ;
                dbo:medication ?med .
       ?med rdf:type yago:Compound114818238 ;
                           rdf:type dbo:Drug .
    }


## ACTORS.NT


    PREFIX dbo: <http://dbpedia.org/ontology/>
    PREFIX dbp: <http://dbpedia.org/property/>
    CONSTRUCT {
        ?actor rdf:type yago:Actor109765278 .
    }
    WHERE {
        ?movie rdf:type yago:Movie106613686 ;
            dbo:wikiPageWikiLink  dbc:American_science_fiction_action_films ;
            dbo:starring ?actor .
    ?actor rdf:type yago:Actor109765278 .
    } LIMIT 400s

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