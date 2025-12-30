from . import config
from . import preprocess
from . import eda
from . import visualize
from . import model

__version__ = "1.0.0"
__author__ = "23122001_23122004_23122033"

from .preprocess import (
    DataProcessor
)

from .eda import (
    OutlierAnalyzer, CorrelationAnalyzer
)
from .visualize import (
    Plotter, SHAPPlotter
)
from .model import (
    Clusterer, RandomForestPredictor, LinearRegressionPredictor
)

