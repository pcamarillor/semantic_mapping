import math
from sematch.semantic.similarity import EntitySimilarity


def compute_ic():
    N = 5356142
    # actress -> freq = 1926503
    freq = 158505
    ic = -math.log(freq / N)

    print(ic)


sim = EntitySimilarity()
print(sim.similarity('http://dbpedia.org/resource/Joffrey_Baratheon', 'http://dbpedia.org/resource/Cersei_Lannister'))
