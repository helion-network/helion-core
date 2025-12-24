import warnings

warnings.warn(
    "helion.dht_utils has been moved to helion.utils.dht. This alias will be removed in Helion 2.2.0+",
    DeprecationWarning,
    stacklevel=2,
)

from helion.utils.dht import *
