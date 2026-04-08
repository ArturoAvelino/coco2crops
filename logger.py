import logging
from tqdm import tqdm

class TqdmHandler(logging.Handler):
    def emit(self, record):
        msg = self.format(record)
        tqdm.write(msg)

def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            TqdmHandler()
        ]
    )