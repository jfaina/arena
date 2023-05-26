from __future__ import annotations
import os
import sys
import numpy as np
from collections import namedtuple
import einops
from typing import Union, Optional, Tuple
import torch as t
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, jaxtyped
import functools
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from tqdm.notebook import tqdm
import typeguard
# Make sure exercises are in the path
chapter = r"chapter0_fundamentals"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "part2_cnns"
if str(exercises_dir) not in sys.path: sys.path.append(str(exercises_dir))

from plotly_utils import imshow, line, bar
from part2_cnns.utils import display_array_as_img, display_soln_array_as_img
import part2_cnns.tests as tests

MAIN = __name__ == "__main__"


if MAIN and False:
    arr = np.load(section_dir / "numbers.npy")
    for array in [
        einops.rearrange(arr, 'b c h w -> c h (b w)'),
        einops.repeat(arr[0], 'c h w -> c (copies h) w', copies=2),
        einops.repeat(arr[:2], 'b c h w -> c (b h) (copies w)', copies=2),
        einops.repeat(arr[0], 'c h w -> c (h copies) w', copies=2),
        einops.rearrange(arr[0], 'c h w -> h (c w)'),
        einops.rearrange(arr, '(b1 b2) c h w -> c (b1 h) (b2 w)', b1=2),
        einops.reduce(arr, 'b c h w -> h (b w)', 'max'),
        einops.reduce(arr, 'b c h w -> h (b w)', 'min'),
        einops.reduce(arr, 'b c h w -> c h w', 'min'),
        ]:
        display_array_as_img(array)

def einsum_trace(mat: np.ndarray):
    '''
    Returns the same as `np.trace`.
    '''
    return einops.einsum(mat, 'i i -> ')

def einsum_mv(mat: np.ndarray, vec: np.ndarray):
    '''
    Returns the same as `np.matmul`, when `mat` is a 2D array and `vec` is 1D.
    '''
    return einops.einsum(mat, vec, 'i j , j -> i')


def einsum_mm(mat1: np.ndarray, mat2: np.ndarray):
    '''
    Returns the same as `np.matmul`, when `mat1` and `mat2` are both 2D arrays.
    '''
    return einops.einsum(mat1, mat2, 'i j , j k -> i k')

def einsum_inner(vec1: np.ndarray, vec2: np.ndarray):
    '''
    Returns the same as `np.inner`.
    '''
    return einops.einsum(vec1, vec2, 'i , i -> ')

def einsum_outer(vec1: np.ndarray, vec2: np.ndarray):
    '''
    Returns the same as `np.outer`.
    '''
    return einops.einsum(vec1, vec2, 'i , j -> i j')


if MAIN:
    tests.test_einsum_trace(einsum_trace)
    tests.test_einsum_mv(einsum_mv)
    tests.test_einsum_mm(einsum_mm)
    tests.test_einsum_inner(einsum_inner)
    tests.test_einsum_outer(einsum_outer)


if MAIN:
    test_input = t.tensor(
        [[0, 1, 2, 3, 4], 
        [5, 6, 7, 8, 9], 
        [10, 11, 12, 13, 14], 
        [15, 16, 17, 18, 19]], dtype=t.float
    )

    TestCase = namedtuple("TestCase", ["output", "size", "stride"])

    test_cases = [
        TestCase(
            output=t.tensor([0, 1, 2, 3]), 
            size=(4,),
            stride=(1,),
        ),
        TestCase(
            output=t.tensor([[0, 2], [5, 7]]), 
            size=(2, 2),
            stride=(5, 2),
        ),

        TestCase(
            output=t.tensor([0, 1, 2, 3, 4]),
            size=(5, ),
            stride=(1, ),
        ),

        TestCase(
            output=t.tensor([0, 5, 10, 15]),
            size=(4, ),
            stride=(5, ),
        ),

        TestCase(
            output=t.tensor([
                [0, 1, 2], 
                [5, 6, 7]
            ]), 
            size=(2, 3),
            stride=(5, 1),
        ),

        TestCase(
            output=t.tensor([
                [0, 1, 2], 
                [10, 11, 12]
            ]), 
            size=(2, 3),
            stride=(10, 1),
        ),

        TestCase(
            output=t.tensor([
                [0, 0, 0], 
                [11, 11, 11]
            ]), 
            size=(2, 3),
            stride=(11, 0),
        ),

        TestCase(
            output=t.tensor([0, 6, 12, 18]), 
            size=(4, ),
            stride=(6, ),
        ),
    ]

    for (i, test_case) in enumerate(test_cases):
        if (test_case.size is None) or (test_case.stride is None):
            print(f"Test {i} failed: attempt missing.")
        else:
            actual = test_input.as_strided(size=test_case.size, stride=test_case.stride)
            if (test_case.output != actual).any():
                print(f"Test {i} failed:")
                print(f"Expected: {test_case.output}")
                print(f"Actual: {actual}\n")
            else:
                print(f"Test {i} passed!\n")

@jaxtyped
@typeguard.typechecked
def as_strided_trace(mat: Float[t.Tensor, "i j"]) -> Float[t.Tensor, ""]:
    '''
    Returns the same as `torch.trace`, using only `as_strided` and `sum` methods.
    '''
    return mat.as_strided(size=(min(mat.shape), ), stride=(sum(mat.stride()), )).sum()


if MAIN:
    tests.test_trace(as_strided_trace)

@jaxtyped
@typeguard.typechecked
def as_strided_mv(mat: Float[Tensor, "i j"], vec: Float[Tensor, "j"]) -> Float[Tensor, "i"]:
    '''
    Returns the same as `torch.matmul`, using only `as_strided` and `sum` methods.
    '''
    A = mat * vec.as_strided(size=mat.shape, stride=(0, vec.stride()[0]))
    return A.sum(dim=1)


if MAIN:
    tests.test_mv(as_strided_mv)
    tests.test_mv2(as_strided_mv)

@jaxtyped
@typeguard.typechecked
def as_strided_mm(matA: Float[Tensor, "i j"], matB: Float[Tensor, "j k"]) -> Float[Tensor, "i k"]:
    '''
    Returns the same as `torch.matmul`, using only `as_strided` and `sum` methods.
    '''
    A = matA.as_strided((matA.shape[0], matA.shape[1], matB.shape[1]), (matA.stride()[0], matA.stride()[1], 0))
    B = matB.as_strided((matA.shape[0], matB.shape[0], matB.shape[1]), (0, matB.stride()[0], matB.stride()[1]))
    C = A * B
    return C.sum(dim=1)

if MAIN:
    tests.test_mm(as_strided_mm)
    tests.test_mm2(as_strided_mm)

@jaxtyped
@typeguard.typechecked
def conv1d_minimal_simple(x: Float[Tensor, "w"], weights: Float[Tensor, "kw"]) -> Float[Tensor, "ow"]:
    '''
    Like torch's conv1d using bias=False and all other keyword arguments left at their default values.

    Simplifications: batch = input channels = output channels = 1.

    x: shape (width,)
    weights: shape (kernel_width,)

    Returns: shape (output_width,)
    '''
    stride = min(x.stride())
    out_width = x.shape[0] - weights.shape[0] + 1
    x_sym = x.as_strided(size=(out_width, weights.shape[0]), stride=(stride, stride))
    return einops.einsum(x_sym, weights, 'ow kw, kw-> ow')


if MAIN:
    tests.test_conv1d_minimal_simple(conv1d_minimal_simple)

@jaxtyped
@typeguard.typechecked
def conv1d_minimal(x: Float[Tensor, "b ic w"], weights: Float[Tensor, "oc ic kw"]) -> Float[Tensor, "b oc ow"]:
    '''
    Like torch's conv1d using bias=False and all other keyword arguments left at their default values.

    x: shape (batch, in_channels, width)
    weights: shape (out_channels, in_channels, kernel_width)

    Returns: shape (batch, out_channels, output_width)
    '''
    out_width = x.shape[2] - weights.shape[2] + 1
    x_sym = x.as_strided(size=(x.shape[0], x.shape[1], out_width, weights.shape[2]),
                         stride=(x.stride()[0], x.stride()[1], x.stride()[2], x.stride()[2]))
    return einops.einsum(x_sym, weights, 'b ic ow kw, oc ic kw -> b oc ow')


if MAIN:
    tests.test_conv1d_minimal(conv1d_minimal)

@jaxtyped
@typeguard.typechecked
def conv2d_minimal(x: Float[Tensor, "b ic h w"], weights: Float[Tensor, "oc ic kh kw"]) -> Float[Tensor, "b oc oh ow"]:
    '''
    Like torch's conv2d using bias=False and all other keyword arguments left at their default values.

    x: shape (batch, in_channels, height, width)
    weights: shape (out_channels, in_channels, kernel_height, kernel_width)

    Returns: shape (batch, out_channels, output_height, output_width)
    '''
    out_height = x.shape[2] - weights.shape[2] + 1
    out_width = x.shape[3] - weights.shape[3] + 1
    x_sym = x.as_strided(size=(x.shape[0], x.shape[1], weights.shape[2], out_height, weights.shape[3], out_width),
                         stride=(x.stride()[0], x.stride()[1], x.stride()[2], x.stride()[2], x.stride()[3], x.stride()[3]))
    return einops.einsum(x_sym, weights, 'b ic kh oh kw ow, oc ic kh kw -> b oc oh ow')
  

if MAIN:
    tests.test_conv2d_minimal(conv2d_minimal)


@jaxtyped
@typeguard.typechecked
def pad1d(x: t.Tensor, left: int, right: int, pad_value: float) -> t.Tensor:
    '''Return a new tensor with padding applied to the edges.

    x: shape (batch, in_channels, width), dtype float32

    Return: shape (batch, in_channels, left + right + width)
    '''
    assert left >= 0 and right >= 0
    ret = x.new_full(size=(x.shape[0], x.shape[1], x.shape[2] + left + right), fill_value=pad_value)
    ret[..., left:ret.shape[2]-right] = x
    return ret


if MAIN:
    tests.test_pad1d(pad1d)
    tests.test_pad1d_multi_channel(pad1d)

@jaxtyped
@typeguard.typechecked
def pad2d(x: t.Tensor, left: int, right: int, top: int, bottom: int, pad_value: float) -> t.Tensor:
    '''Return a new tensor with padding applied to the edges.

    x: shape (batch, in_channels, height, width), dtype float32

    Return: shape (batch, in_channels, top + height + bottom, left + width + right)
    '''
    assert left >= 0 and right >= 0 and top >= 0 and bottom >= 0
    ret = x.new_full(size=(x.shape[0], x.shape[1], x.shape[2] + top + bottom, x.shape[3] + left + right),
                     fill_value=pad_value)
    ret[..., top:ret.shape[2]-bottom, left:ret.shape[3]-right] = x
    return ret


if MAIN:
    tests.test_pad2d(pad2d)
    tests.test_pad2d_multi_channel(pad2d)


@jaxtyped
@typeguard.typechecked
def conv1d(
    x: Float[Tensor, "b ic w"], 
    weights: Float[Tensor, "oc ic kw"], 
    stride: int = 1, 
    padding: int = 0
) -> Float[Tensor, "b oc ow"]:
    '''
    Like torch's conv1d using bias=False.

    x: shape (batch, in_channels, width)
    weights: shape (out_channels, in_channels, kernel_width)

    Returns: shape (batch, out_channels, output_width)
    '''
    assert stride > 0
    if padding:
        x_pad = pad1d(x, left=padding, right=padding, pad_value=0)
    else:
        x_pad = x
    out_width = (x_pad.shape[2] - weights.shape[2]) // stride + 1
    x_sym = x_pad.as_strided(size=(x_pad.shape[0], x_pad.shape[1], out_width, weights.shape[2]),
                stride=(x_pad.stride()[0], x_pad.stride()[1], x_pad.stride()[2] * stride, x_pad.stride()[2]))
    return einops.einsum(x_sym, weights, 'b ic ow kw, oc ic kw -> b oc ow')


if MAIN:
    tests.test_conv1d(conv1d)

IntLike = Union[int, np.int64]
IntPair = Tuple[IntLike, IntLike]
IntOrPair = Union[IntLike, IntPair]

def force_pair(v: IntOrPair) -> IntPair:
    '''Convert v to a pair of int, if it isn't already.'''
    if isinstance(v, tuple):
        if len(v) != 2:
            raise ValueError(v)
        return (int(v[0]), int(v[1]))
    elif isinstance(v, int):
        return (v, v)
    raise ValueError(v)

@jaxtyped
@typeguard.typechecked
def conv2d(
    x: Float[Tensor, "b ic h w"], 
    weights: Float[Tensor, "oc ic kh kw"], 
    stride: IntOrPair = 1, 
    padding: IntOrPair = 0
) -> Float[Tensor, "b oc oh ow"]:
    '''
    Like torch's conv2d using bias=False

    x: shape (batch, in_channels, height, width)
    weights: shape (out_channels, in_channels, kernel_height, kernel_width)

    Returns: shape (batch, out_channels, output_height, output_width)
    '''
    padding_height, padding_width = force_pair(padding)
    assert padding_height >= 0 and padding_width >= 0
    stride_height, stride_width = force_pair(stride)
    assert stride_height > 0 and stride_width > 0
    if padding_height or padding_width:
        x_pad = pad2d(x, left=padding_width, right=padding_width, top=padding_height, bottom=padding_height, pad_value=0)
    else:
        x_pad = x
    out_height = (x_pad.shape[2] - weights.shape[2]) // stride_height + 1
    out_width = (x_pad.shape[3] - weights.shape[3]) // stride_width + 1
    x_sym = x_pad.as_strided(size=(x_pad.shape[0], x_pad.shape[1], weights.shape[2], out_height, weights.shape[3], out_width),
                             stride=(x_pad.stride()[0], x_pad.stride()[1], x_pad.stride()[2], stride_height * x_pad.stride()[2],
                                     x_pad.stride()[3], stride_width * x_pad.stride()[3]))
    return einops.einsum(x_sym, weights, 'b ic kh oh kw ow, oc ic kh kw -> b oc oh ow')



if MAIN:
    tests.test_conv2d(conv2d)

@jaxtyped
@typeguard.typechecked
def maxpool2d(
    x: Float[Tensor, "b ic h w"], 
    kernel_size: IntOrPair, 
    stride: Optional[IntOrPair] = None, 
    padding: IntOrPair = 0
) -> Float[Tensor, "b ic oh ow"]:
    '''
    Like PyTorch's maxpool2d.

    x: shape (batch, channels, height, width)
    stride: if None, should be equal to the kernel size

    Return: (batch, channels, output_height, output_width)
    '''
    padding_height, padding_width = force_pair(padding)
    assert padding_height >= 0 and padding_width >= 0
    stride_height, stride_width = force_pair(stride if stride is not None else kernel_size)
    assert stride_height > 0 and stride_width > 0
    kernel_height, kernel_width = force_pair(kernel_size)
    assert kernel_height > 0 and kernel_width > 0
    if padding_height or padding_width:
        x_pad = pad2d(x, left=padding_width, right=padding_width, top=padding_height, bottom=padding_height, pad_value=-t.inf)
    else:
        x_pad = x
    out_height = (x_pad.shape[2] - kernel_height) // stride_height + 1
    out_width = (x_pad.shape[3] - kernel_width) // stride_width + 1
    x_sym = x_pad.as_strided(size=(x_pad.shape[0], x_pad.shape[1], kernel_height, out_height, kernel_width, out_width),
                             stride=(x_pad.stride()[0], x_pad.stride()[1], x_pad.stride()[2], stride_height * x_pad.stride()[2],
                                     x_pad.stride()[3], stride_width * x_pad.stride()[3]))
    # equivalent to einops.reduce(x_sym, 'b ic kh oh kw ow -> b ic oh ow', 'max')
    return t.amax(x_sym, dim=(2, 4))



if MAIN:
    tests.test_maxpool2d(maxpool2d, n_tests=100)