import logging, sys

_FMT = "%(asctime)s %(levelname)-8s | %(message)s"

def get_logger(name: str = "crypto") -> logging.Logger:
    logging.basicConfig(stream=sys.stdout,
                        level=logging.INFO,
                        format=_FMT)
    return logging.getLogger(name)
