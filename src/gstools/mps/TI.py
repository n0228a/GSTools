import numpy as np

__all__ = ["TrainingImage"]


class TrainingImage:
    """Training image for multiple point statistics simulation.

    The MPS analogue of a covariance model: encapsulates the training data
    and the variable type used to compare data events.

    Parameters
    ----------
    data : numpy.ndarray
        Training image data (n-d array).
    categorical : bool, optional
        Whether the variable is categorical. Default: True.
    """

    def __init__(self, data, categorical: bool = True):
        self._data = np.asarray(data)
        self._categorical = bool(categorical)

    @property
    def data(self) -> np.ndarray:
        """ndarray: Raw training image data."""
        return self._data

    @property
    def ndim(self) -> int:
        """int: Number of spatial dimensions."""
        return self._data.ndim

    @property
    def shape(self) -> tuple:
        """tuple: Shape of the training image."""
        return self._data.shape

    @property
    def categorical(self) -> bool:
        """bool: Whether the variable is categorical."""
        return self._categorical

    def __repr__(self):
        return f"TrainingImage(shape={self.shape}, categorical={self._categorical})"
