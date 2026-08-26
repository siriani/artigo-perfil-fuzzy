"""fuzzyprofile — transparent fuzzy clustering of psychometric (IPIP-50) profiles.

Public API
----------
FuzzyCMeansEngine   : Bezdek Fuzzy C-Means with Euclidean / Mahalanobis metric (Module 2.1)
validity            : FPC, PE, MPC, Xie-Beni, Fukuyama-Sugeno, Kwon (Module 2.1)
distances           : squared-distance kernels + covariance regularization (Module 3)
data                : IPIP-50 loader + synthetic generator
likert_fuzzify      : triangular fuzzy number / IFS encoding of Likert responses (Module 1.4)

Not yet implemented (documented stubs): gridsearch (Module 2.2), benchmarks (Module 2.3).
"""
from .engine import FuzzyCMeansEngine
from . import validity, distances, data, likert_fuzzify

__all__ = ["FuzzyCMeansEngine", "validity", "distances", "data", "likert_fuzzify"]
__version__ = "0.1.0"
