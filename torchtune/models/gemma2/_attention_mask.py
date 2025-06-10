# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

from torchtune.modules.attention_utils import (
    _MaskType,
    _SUPPORTS_FLEX_ATTENTION,
    causal_mask_flex,
)

if _SUPPORTS_FLEX_ATTENTION:
    import torch
    from torch import Tensor
    from torch.nn.attention.flex_attention import (
        _score_mod_signature,
        create_block_mask,
    )

    def get_softcap_score_mod(softcapping: float) -> _score_mod_signature:
        """
        Inspired by https://pytorch.org/blog/flexattention/

        Args:
            softcapping (float): Softcapping value to apply to the attention scores.

        Returns:
            A flex attention score mod that applies softcapping.
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
    ) -> _MaskType:
        """
        Inspired by https://pytorch.org/blog/flexattention/

        Args:
            sliding_window (int): Sliding window size to apply to the attention mask.

        Returns:
            A flex attention mask mod that applies sliding window masking.
        """
        if not _SUPPORTS_FLEX_ATTENTION:
            raise ValueError("Local attention is only supported with flex attention.")
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
