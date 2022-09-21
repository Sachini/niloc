from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np

"""
We use two levels of hierarchy for flexible data loading pipeline:
  - Sequence: Read the sequence (e.g. single trajectory) from file of a given format and compute per-frame feature and target.
  - Dataset: subclasses of PyTorch's Dataset class. It has three roles:
      1. Create a Sequence instance internally to load data and compute feature/target.
      2. Apply post processing, e.g. smoothing or truncating, to the loaded sequence.
      3. Define how to extract samples from the sequence.
To define a new dataset for training/testing:
  1. Subclass ProcessedSequence class. Load data and compute feature/target in "load()" function.
  2. Subclass the PyTorch Dataset. In the constructor, use the custom ProcessedSequence class to load data. You can also
     apply additional processing to the raw sequence, e.g. smoothing or truncating. Define how to extract samples from
     the sequences by overriding "__getitem()__" function.

"""


class ProcessedSequence(ABC):
    """
    An abstract interface to read a sequence (e.r. single trajectory) from file of a given format
    and compute per-frame (i.e values for a time step) feature and target.
    This class is to decouple preprocessing specific to the file format from dataset.
    """

    @abstractmethod
    def load(self, path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Read a single sequence from file and compute per-frame feature and target.
        args:
            - path: path to file
        returns:
            - preprocessed features, targets and auxillary fields (such as time, orientation)
        """
        pass

    @abstractmethod
    def load_length(self, path: str) -> int:
        """
        Read a single sequence from file and return length of processed target w/o running all the
        preprocessing steps.
        args:
            - path: path to file
        returns:
            - length of preprocessed targets
        """
        pass
