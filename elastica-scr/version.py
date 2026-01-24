import importlib.metadata

try:
    VERSION = importlib.metadata.version("pyelastica-SCR")
except importlib.metadata.PackageNotFoundError:
    VERSION = "unknown"
