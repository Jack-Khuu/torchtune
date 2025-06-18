# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Optional

import torch
from torch import Tensor

from torchtune.modules.attention_utils import (
    _MaskType,
    _SUPPORTS_FLEX_ATTENTION,
    causal_mask_flex,
)

if _SUPPORTS_FLEX_ATTENTION:
    from torch.nn.attention.flex_attention import create_block_mask

    def _get_flex_sliding_attention_mask(
        mask: Optional[_MaskType],
        sliding_window_size: int,
        bsz: int,
        seq_len: int,
        device: Optional[torch.device] = None,
    ) -> _MaskType:
        """
        Inspired by https://pytorch.org/blog/flexattention/.

        Args:
            mask (Optional[_MaskType]): Mask to apply to the attention scores.
            sliding_window_size (int): Sliding window size to apply to the attention mask.
            bsz (int): Batch size.
            seq_len (int): Sequence length.
            device (Optional[torch.device]): Device to use for the mask. Argument is unused
                but listed for consistency. Defaults to None.

        Returns:
            A flex attention block mask that applies sliding window masking.

        Raises:
            ValueError: If the input mask is not a BlockMask (flex).
        """
        # Flex attention BlockMask
        if mask is None:
            mask_mod = causal_mask_flex
            q_seq_len, kv_seq_len = seq_len, seq_len
        elif isinstance(mask, BlockMask):
            mask_mod = mask.mask_mod
            q_seq_len, kv_seq_len = mask.seq_lengths
        else:
            raise ValueError("Unsupported mask type")

        def sliding_mask_mod(b: Tensor, h: Tensor, q: Tensor, kv_idx: Tensor) -> Tensor:
            sliding_mask = q - kv_idx < sliding_window_size
            # Apply the original mask mod
            return sliding_mask & mask_mod(b, h, q, kv_idx)

        return create_block_mask(
            sliding_mask_mod,
            bsz,
            None,
            q_seq_len,
            kv_seq_len,
        )


def get_softcap_score_mod(softcapping: float) -> Callable:
    """
    Inspired by https://pytorch.org/blog/flexattention/

    Args:
        softcapping (float): Softcapping value to apply to the attention scores.

    Returns:
        A flex attention score mod that applies softcapping.
        (Specific Callable type is _score_mod_signature)
    """

    def softcap_score_mod(
        score: Tensor, _b: Tensor, _h: Tensor, _q: Tensor, _kv_idx: Tensor
    ) -> Tensor:
        score = score / softcapping
        score = torch.tanh(score)
        return score * softcapping

    return softcap_score_mod


def get_sliding_attention_mask(
    mask: Optional[_MaskType],
    sliding_window_size: int,
    bsz: int,
    seq_len: int,
    device: Optional[torch.device] = None,
) -> _MaskType:
    """
    Args:
        mask (Optional[_MaskType]): Mask to apply to the attention scores.
        sliding_window_size (int): Sliding window size to apply to the attention mask.
        bsz (int): Batch size. Argument is unused, but listed for consistency.
        seq_len (int): Sequence length.
        device (Optional[torch.device]): Device to use for the mask. Defaults to None.

    Returns:
        A tensor mask that applies sliding window masking.
        - If flex attention is supported, returns a flex attention block mask instead.

    Raises:
        ValueError: If the input mask is not a Tensor
    """

    # If flex attention is supported, use the flex attention mask
    if _SUPPORTS_FLEX_ATTENTION:
        return _get_flex_sliding_attention_mask(
            mask, sliding_window_size, bsz, seq_len, device
        )

    if mask is None:
        mask = torch.tril(
            torch.ones(size=(bsz, seq_len, seq_len), dtype=torch.bool).to(device)
        )

    if not isinstance(mask, torch.Tensor):
        raise ValueError(
            f"For non-flex attention, mask must be a Tensor. Got: {type(mask)}"
        )

    if mask.dtype == torch.bool:
        mask = torch.where(mask.logical_not(), -2.3819763e38, 0)
        all_ones = torch.ones_like(mask)
        sliding_mask = torch.triu(all_ones, -1 * sliding_window_size + 1) * torch.tril(
            all_ones, sliding_window_size - 1
        )
        mask = torch.where(sliding_mask == 1, mask, -2.3819763e38).to(torch.bfloat16)

    return mask
