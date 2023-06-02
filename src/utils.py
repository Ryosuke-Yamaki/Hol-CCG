from torch.nn.functional import normalize
import pickle
import random
import numpy as np
import torch
from torch import conj
from torch.fft import fft, ifft
from typing import Any


def circular_correlation(a: torch.Tensor, b: torch.Tensor, vector_norm: float) -> torch.Tensor:
    """Compute circular correlation between two vectors.

    Parameters
    ----------
    a : torch.Tensor
        first vector
    b : torch.Tensor
        second vector
    vector_norm : float
        max norm of vector

    Returns
    -------
    torch.Tensor
        circular correlation between two vectors
    """
    a_ = fft(a)
    b_ = fft(b)
    c_ = conj(a_) * b_
    c = ifft(c_).real
    if vector_norm is not None:
        c = vector_norm * normalize(c, dim=-1)
    return c


def inverse_circular_correlation(
        p: torch.Tensor,
        c1: torch.Tensor,
        vector_norm: float,
        child_is_left: bool = True) -> torch.Tensor:
    """Compute inverse circular correlation between two vectors.

    Parameters
    ----------
    p : torch.Tensor
        parent vector
    c1 : torch.Tensor
        child vector
    vector_norm : float
        max norm of vector
    child_is_left : bool
        whether child is left or right child

    Returns
    -------
    torch.Tensor
        inverse circular correlation between two vectors
    """
    p_ = fft(p)
    c1_ = fft(c1)
    if child_is_left:
        c2_ = p_ / (conj(c1_) + 1e-12)
    else:
        c2_ = conj(p_ / (c1_ + 1e-12))
    c2 = ifft(c2_).real
    if vector_norm is not None:
        c2 = vector_norm * normalize(c2, dim=-1)
    return c2


def circular_convolution(a: torch.Tensor, b: torch.Tensor, vector_norm: float) -> torch.Tensor:
    """Compute circular convolution between two vectors.

    Parameters
    ----------
    a : torch.Tensor
        first vector
    b : torch.Tensor
        second vector
    vector_norm : float
        max norm of vector

    Returns
    -------
    torch.Tensor
        circular convolution between two vectors
    """
    a_ = fft(a)
    b_ = fft(b)
    c_ = a_ * b_
    c = ifft(c_).real
    if vector_norm is not None:
        c = vector_norm * normalize(c, dim=-1)
    return c


def inverse_circular_convolution(p: torch.Tensor, c1: torch.Tensor) -> torch.Tensor:
    """Compute inverse circular convolution between two vectors.

    Parameters
    ----------
    p : torch.Tensor
        parent vector
    c1 : torch.Tensor
        child vector

    Returns
    -------
    torch.Tensor
        inverse circular convolution between two vectors
    """
    p_ = fft(p)
    c1_ = conj(fft(c1))
    c2_ = p_ / (c1_ + 1e-6)
    c2 = ifft(c2_).real
    return c2


def shuffled_circular_convolution(
    a: torch.Tensor,
    b: torch.Tensor,
    P: torch.Tensor,
        vector_norm: float) -> torch.Tensor:
    """Compute circular convolution between two vectors.

    Parameters
    ----------
    a : torch.Tensor
        first vector
    b : torch.Tensor
        second vector
    P : torch.Tensor
        permutation matrix
    vector_norm : float
        max norm of vector

    Returns
    -------
    torch.Tensor
        circular convolution between two vectors
    """
    a_ = fft(a)
    b_ = fft(torch.index_select(b, -1, P))
    c_ = a_ * b_
    c = ifft(c_).real
    if vector_norm is not None:
        c = vector_norm * normalize(c, dim=-1)
    return c


def complex_normalize(v: torch.Tensor) -> torch.Tensor:
    """Normalize the vector in complex space.

    Parameters
    ----------
    v : torch.Tensor
        vector to be normalized

    Returns
    -------
    torch.Tensor
        normalized vector
    """
    v_ = fft(v)
    v_ = v_ / (torch.abs(v_) + 1e-12)
    v = ifft(v_).real
    return v


def set_random_seed(seed: int) -> None:
    """Set random seed.

    Parameters
    ----------
    seed : int
        random seed
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def dump(object: Any, path: str) -> None:
    """Dump object to path.

    Parameters
    ----------
    object : Any
        object to be dumped
    path : str
        path to dump object
    """
    with open(path, mode='wb') as f:
        pickle.dump(object, f)


def load(path: str) -> Any:
    """Load object from path.

    Parameters
    ----------
    path : str
        path to load object

    Returns
    -------
    Any
        loaded object
    """
    with open(path, mode='rb') as f:
        data = pickle.load(f)
        return data
