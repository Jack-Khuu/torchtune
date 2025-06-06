# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtune.utils._import_guard import _SUPPORTS_FLEX_ATTENTION

if _SUPPORTS_FLEX_ATTENTION:
    import torch
    from torch import Tensor
    from torch.nn.attention.flex_attention import (
        _mask_mod_signature,
        _score_mod_signature,
    )

    def get_scaled_softcap_score_mod(softcapping: float) -> _score_mod_signature:
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

    def get_sliding_mask_mod(sliding_window: int) -> _mask_mod_signature:
        """
        Inspired by https://pytorch.org/blog/flexattention/

        Args:
            sliding_window (int): Sliding window size to apply to the attention mask.

        Returns:
            A flex attention mask mod that applies sliding window masking.
        """

        def sliding_mask_mod(
            _b: Tensor, _h: Tensor, q: Tensor, kv_idx: Tensor
        ) -> Tensor:
            return q - kv_idx < sliding_window

        return sliding_mask_mod
