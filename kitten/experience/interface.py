from collections import namedtuple

# A class representing a standard Markov Decision Process Transition
Transition = namedtuple('Transition', ["s_0", "a", "r", "s_1", "d"])

# Auxiliary information contained on retrieval from memory
AuxiliaryMemoryData = namedtuple('AuxiliaryMemoryData', [
    "weights", # Recommended importance weighting to match memory data distribution
    "random",  # Random number associated with sample - eg. for bootstrapping split
    "indices", # Index of sample within memory 
])