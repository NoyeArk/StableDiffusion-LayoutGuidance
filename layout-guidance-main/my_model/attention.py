# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import torch
from torch import nn
from typing import Optional
import torch.nn.functional as F
from dataclasses import dataclass

from diffusers.utils import BaseOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils.import_utils import is_xformers_available
from diffusers.models.embeddings import ImagePositionalEmbeddings
from diffusers.configuration_utils import ConfigMixin, register_to_config


@dataclass
class Transformer2DModelOutput(BaseOutput):
    """
    Args:
        sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` or `(batch size, num_vector_embeds - 1, num_latent_pixels)` if [`Transformer2DModel`] is discrete):
            Hidden states conditioned on `encoder_hidden_states` input. If discrete, returns probability distributions
            for the unnoised latent pixels.
    """

    sample: torch.FloatTensor


if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None


class Transformer2DModel(ModelMixin, ConfigMixin):
    """
    用于类图像数据的 Transformer 模型。采用离散（向量嵌入的类）或连续的（实际
    嵌入）inputs_coarse。

    当输入是连续的时：首先，投影输入（又名嵌入）并调整为 b、t、d。然后应用标准
    变压器动作。最后，重塑图像。

    当输入是离散的时：首先，输入（潜在像素的类别）被转换为嵌入并具有位置
    应用了嵌入，请参阅“ImagePositionalEmbeddings”。然后应用标准变压器动作。最后，预测
    无噪声图像的类别。

    请注意，假设其中一个输入类是被屏蔽的潜在像素。未噪声的预测类别
    图像不包含对屏蔽像素的预测，因为无法屏蔽未加噪的图像。
    Parameters:
        num_attention_heads （'int'， *可选*， 默认为 16）： 用于多头注意力的头部数量。
        attention_head_dim （'int'， *optional*， 默认为 88）：每个磁头中的通道数。
        in_channels （'int'， *可选*）：
            如果输入是连续的，则传递。输入和输出中的通道数。
        num_layers （'int'， *optional*， 默认为 1）： 要使用的 Transformer 模块的层数。
        dropout （'float'， *optional*， 默认为 0.1）：要使用的退出概率。
        cross_attention_dim （'int'， *可选*）：要使用的上下文维度的数量。
        sample_size （'int'， *可选*）：如果输入是离散的，则传递。潜在图像的宽度。
            请注意，这在训练时是固定的，因为它用于学习许多位置嵌入。看
            'ImagePositionalEmbeddings'。
        num_vector_embeds （'int'， *可选*）：
            如果输入是离散的，则传递。潜在像素的向量嵌入的类数。
            包括被屏蔽的潜在像素的类。
        activation_fn （'str'， *可选*， 默认为 '“geglu”'）： 用于前馈的激活函数。
        num_embeds_ada_norm （ 'int'， *optional*）： 如果norm_layers中至少有一个是 'AdaLayerNorm'，则传递。
            训练期间使用的扩散步骤数。请注意，这在使用时在训练时是固定的
            了解添加到隐藏状态中的许多嵌入。在推理过程中，您可以对
            不超过但不超过“num_embeds_ada_norm”的步数。
        attention_bias （'bool'， *可选*）：
            配置 TransformerBlocks 的注意力是否应包含偏置参数。
    """

    @register_to_config
    def __init__(
            self,
            num_attention_heads: int = 16,
            attention_head_dim: int = 88,
            in_channels: Optional[int] = None,
            num_layers: int = 1,
            dropout: float = 0.0,
            norm_num_groups: int = 32,
            cross_attention_dim: Optional[int] = None,
            attention_bias: bool = False,
            sample_size: Optional[int] = None,
            num_vector_embeds: Optional[int] = None,
            activation_fn: str = "geglu",
            num_embeds_ada_norm: Optional[int] = None,
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim

        # 1. Transformer2DModel can process both standard continous images of shape `(batch_size, num_channels, width, height)` as well as quantized image embeddings of shape `(batch_size, num_image_vectors)`
        # Define whether input is continuous or discrete depending on configuration
        self.is_input_continuous = in_channels is not None
        self.is_input_vectorized = num_vector_embeds is not None

        if self.is_input_continuous and self.is_input_vectorized:
            raise ValueError(
                f"Cannot define both `in_channels`: {in_channels} and `num_vector_embeds`: {num_vector_embeds}. Make"
                " sure that either `in_channels` or `num_vector_embeds` is None."
            )
        elif not self.is_input_continuous and not self.is_input_vectorized:
            raise ValueError(
                f"Has to define either `in_channels`: {in_channels} or `num_vector_embeds`: {num_vector_embeds}. Make"
                " sure that either `in_channels` or `num_vector_embeds` is not None."
            )

        # 2. Define input layers
        if self.is_input_continuous:
            self.in_channels = in_channels

            self.norm = torch.nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True)
            self.proj_in = nn.Conv2d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)
        elif self.is_input_vectorized:
            assert sample_size is not None, "Transformer2DModel over discrete input must provide sample_size"
            assert num_vector_embeds is not None, "Transformer2DModel over discrete input must provide num_embed"

            self.height = sample_size
            self.width = sample_size
            self.num_vector_embeds = num_vector_embeds
            self.num_latent_pixels = self.height * self.width

            self.latent_image_embedding = ImagePositionalEmbeddings(
                num_embed=num_vector_embeds, embed_dim=inner_dim, height=self.height, width=self.width
            )

        # 3. Define transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    num_embeds_ada_norm=num_embeds_ada_norm,
                    attention_bias=attention_bias,
                )
                for d in range(num_layers)
            ]
        )

        # 4. Define output layers
        if self.is_input_continuous:
            self.proj_out = nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)
        elif self.is_input_vectorized:
            self.norm_out = nn.LayerNorm(inner_dim)
            self.out = nn.Linear(inner_dim, self.num_vector_embeds - 1)

    def _set_attention_slice(self, slice_size):
        for block in self.transformer_blocks:
            block._set_attention_slice(slice_size)

    def forward(self, hidden_states, encoder_hidden_states=None, timestep=None, attn_map=None, attn_shift=False,
                obj_ids=None, relationship=None, return_dict: bool = True):
        """
        Args:
            hidden_states ( When discrete, `torch.LongTensor` of shape `(batch size, num latent pixels)`.
                When continous, `torch.FloatTensor` of shape `(batch size, channel, height, width)`): Input
                hidden_states
            encoder_hidden_states ( `torch.LongTensor` of shape `(batch size, context dim)`, *optional*):
                Conditional embeddings for cross attention layer. If not given, cross-attention defaults to
                self-attention.
            timestep ( `torch.long`, *optional*):
                Optional timestep to be applied as an embedding in AdaLayerNorm's. Used to indicate denoising step.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`models.unet_2d_condition.UNet2DConditionOutput`] instead of a plain tuple.

        Returns:
            [`~models.attention.Transformer2DModelOutput`] or `tuple`: [`~models.attention.Transformer2DModelOutput`]
            if `return_dict` is True, otherwise a `tuple`. When returning a tuple, the first element is the sample
            tensor.
        """
        # 1. Input
        if self.is_input_continuous:
            batch, channel, height, weight = hidden_states.shape
            residual = hidden_states
            hidden_states = self.norm(hidden_states)
            hidden_states = self.proj_in(hidden_states)
            inner_dim = hidden_states.shape[1]
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * weight, inner_dim)
        elif self.is_input_vectorized:
            hidden_states = self.latent_image_embedding(hidden_states)

        # 2. Blocks
        for block in self.transformer_blocks:
            hidden_states, cross_attn_prob = block(hidden_states, context=encoder_hidden_states, timestep=timestep)

        # 3. Output
        if self.is_input_continuous:
            hidden_states = hidden_states.reshape(batch, height, weight, inner_dim).permute(0, 3, 1, 2)
            hidden_states = self.proj_out(hidden_states)
            output = hidden_states + residual
        elif self.is_input_vectorized:
            hidden_states = self.norm_out(hidden_states)
            logits = self.out(hidden_states)
            # (batch, self.num_vector_embeds - 1, self.num_latent_pixels)
            logits = logits.permute(0, 2, 1)

            # log(p(x_0))
            output = F.log_softmax(logits.double(), dim=1).float()

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output), cross_attn_prob

    def _set_use_memory_efficient_attention_xformers(self, use_memory_efficient_attention_xformers: bool):
        for block in self.transformer_blocks:
            block._set_use_memory_efficient_attention_xformers(use_memory_efficient_attention_xformers)


class AttentionBlock(nn.Module):
    """
    一个注意力块，允许空间位置相互关注。最初是从这里移植过来的，但进行了改编到N-d情况。
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    Uses three q, k, v linear layers to compute attention.

    Parameters:
        channels （'int'）：输入和输出中的通道数。
        num_head_channels （'int'， *可选*）：
            每个磁头中的通道数。如果为 None，则 'num_heads' = 1。
        norm_num_groups （'int'， *可选*， 默认为 32）：用于组范数的组数。
        rescale_output_factor （'float'， *可选*， 默认为 1.0）： 调整输出比例的因子。
        eps （'float'， *可选*， 默认为 1e-5）： 用于组范数的 epsilon 值。
    """

    def __init__(
            self,
            channels: int,
            num_head_channels: Optional[int] = None,
            norm_num_groups: int = 32,
            rescale_output_factor: float = 1.0,
            eps: float = 1e-5,
    ):
        super().__init__()
        self.channels = channels

        self.num_heads = channels // num_head_channels if num_head_channels is not None else 1
        self.num_head_size = num_head_channels
        self.group_norm = nn.GroupNorm(num_channels=channels, num_groups=norm_num_groups, eps=eps, affine=True)

        # define q,k,v as linear layers
        self.query = nn.Linear(channels, channels)
        self.key = nn.Linear(channels, channels)
        self.value = nn.Linear(channels, channels)

        self.rescale_output_factor = rescale_output_factor
        self.proj_attn = nn.Linear(channels, channels, 1)

    def transpose_for_scores(self, projection: torch.Tensor) -> torch.Tensor:
        new_projection_shape = projection.size()[:-1] + (self.num_heads, -1)
        # move heads to 2nd position (B, T, H * D) -> (B, T, H, D) -> (B, H, T, D)
        new_projection = projection.view(new_projection_shape).permute(0, 2, 1, 3)
        return new_projection

    def forward(self, hidden_states):
        residual = hidden_states
        batch, channel, height, width = hidden_states.shape

        # norm
        hidden_states = self.group_norm(hidden_states)

        hidden_states = hidden_states.view(batch, channel, height * width).transpose(1, 2)

        # proj to q, k, v
        query_proj = self.query(hidden_states)
        key_proj = self.key(hidden_states)
        value_proj = self.value(hidden_states)

        # transpose
        query_states = self.transpose_for_scores(query_proj)
        key_states = self.transpose_for_scores(key_proj)
        value_states = self.transpose_for_scores(value_proj)

        # get scores
        scale = 1 / math.sqrt(math.sqrt(self.channels / self.num_heads))
        attention_scores = torch.matmul(query_states * scale, key_states.transpose(-1, -2) * scale)  # TODO: use baddmm
        attention_probs = torch.softmax(attention_scores.float(), dim=-1).type(attention_scores.dtype)

        # compute attention output
        hidden_states = torch.matmul(attention_probs, value_states)

        hidden_states = hidden_states.permute(0, 2, 1, 3).contiguous()
        new_hidden_states_shape = hidden_states.size()[:-2] + (self.channels,)
        hidden_states = hidden_states.view(new_hidden_states_shape)

        # compute next hidden_states
        hidden_states = self.proj_attn(hidden_states)
        hidden_states = hidden_states.transpose(-1, -2).reshape(batch, channel, height, width)

        # res connect and rescale
        hidden_states = (hidden_states + residual) / self.rescale_output_factor
        return hidden_states


class BasicTransformerBlock(nn.Module):
    r"""
    A basic Transformer block.

    Parameters:
        dim （'int'）：输入和输出中的通道数。
        num_attention_heads （'int'）：用于多头注意力的头部数量。
        attention_head_dim （'int'）：每个磁头中的通道数。
        dropout （'float'， *optional*， defaults to 0.0）：要使用的退出概率。
        cross_attention_dim （'int'， *可选*）：交叉注意力的上下文向量的大小。
        activation_fn （'str'， *可选*， 默认为 '“geglu”'）： 用于前馈的激活函数。
        num_embeds_ada_norm （：
            obj： 'int'， *optional*）： 训练期间使用的扩散步骤数。请参阅“Transformer2DModel”。
        attention_bias （：
            obj： 'bool'， *可选*， 默认为 'False'）： 配置注意是否应包含偏差参数。
    """

    def __init__(
            self,
            dim: int,
            num_attention_heads: int,
            attention_head_dim: int,
            dropout=0.0,
            cross_attention_dim: Optional[int] = None,
            activation_fn: str = "geglu",
            num_embeds_ada_norm: Optional[int] = None,
            attention_bias: bool = False,
    ):
        super().__init__()
        self.attn1 = CrossAttention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
        )  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn)
        self.attn2 = CrossAttention(
            query_dim=dim,
            cross_attention_dim=cross_attention_dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
        )  # is self-attn if context is none

        # layer norms
        self.use_ada_layer_norm = num_embeds_ada_norm is not None
        if self.use_ada_layer_norm:
            self.norm1 = AdaLayerNorm(dim, num_embeds_ada_norm)
            self.norm2 = AdaLayerNorm(dim, num_embeds_ada_norm)
        else:
            self.norm1 = nn.LayerNorm(dim)
            self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def _set_attention_slice(self, slice_size):
        self.attn1._slice_size = slice_size
        self.attn2._slice_size = slice_size

    def _set_use_memory_efficient_attention_xformers(self, use_memory_efficient_attention_xformers: bool):
        if not is_xformers_available():
            print("Here is how to install it")
            raise ModuleNotFoundError(
                "Refer to https://github.com/facebookresearch/xformers for more information on how to install"
                " xformers",
                name="xformers",
            )
        elif not torch.cuda.is_available():
            raise ValueError(
                "torch.cuda.is_available() should be True but is False. xformers' memory efficient attention is only"
                " available for GPU "
            )
        else:
            try:
                # Make sure we can run the memory efficient attention
                _ = xformers.ops.memory_efficient_attention(
                    torch.randn((1, 2, 40), device="cuda"),
                    torch.randn((1, 2, 40), device="cuda"),
                    torch.randn((1, 2, 40), device="cuda"),
                )
            except Exception as e:
                raise e
            self.attn1._use_memory_efficient_attention_xformers = use_memory_efficient_attention_xformers
            self.attn2._use_memory_efficient_attention_xformers = use_memory_efficient_attention_xformers

    def forward(self, hidden_states, context=None, timestep=None):
        # 1. Self-Attention
        norm_hidden_states = (
            self.norm1(hidden_states, timestep) if self.use_ada_layer_norm else self.norm1(hidden_states)
        )
        tmp_hidden_states, cross_attn_prob = self.attn1(norm_hidden_states)
        hidden_states = tmp_hidden_states + hidden_states

        # 2. Cross-Attention
        norm_hidden_states = (
            self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
        )
        tmp_hidden_states, cross_attn_prob = self.attn2(norm_hidden_states, context=context)
        hidden_states = tmp_hidden_states + hidden_states

        # 3. Feed-forward
        hidden_states = self.ff(self.norm3(hidden_states)) + hidden_states

        return hidden_states, cross_attn_prob


class CrossAttention(nn.Module):
    r"""
    交叉注意力层

    Parameters:
        query_dim （'int'）：查询中的通道数。
        cross_attention_dim （'int'， *可选*）：
            上下文中的通道数。如果未给出，则默认为“query_dim”。
        heads （'int'， *optional*， 默认为 8）：用于多头注意力的头数。
        dim_head （'int'， *可选*， 默认为 64）： 每个磁头中的通道数。
        dropout （'float'， *optional*， defaults to 0.0）：要使用的退出概率。
        bias （'bool'， *可选*， 默认为 False）：
            将查询、键和值线性层设置为“True”以包含偏差参数。
    """

    def __init__(
            self,
            query_dim: int,
            cross_attention_dim: Optional[int] = None,
            heads: int = 8,
            dim_head: int = 64,
            dropout: float = 0.0,
            bias=False,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim

        self.scale = dim_head ** -0.5
        self.heads = heads
        # for slice_size > 0 the attention score computation
        # is split across the batch axis to save memory
        # You can set slice_size with `set_attention_slice`
        self._slice_size = None
        self._use_memory_efficient_attention_xformers = False

        self.to_q = nn.Linear(query_dim, inner_dim, bias=bias)
        self.to_k = nn.Linear(cross_attention_dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(cross_attention_dim, inner_dim, bias=bias)

        self.to_out = nn.ModuleList([])
        self.to_out.append(nn.Linear(inner_dim, query_dim))
        self.to_out.append(nn.Dropout(dropout))

    def reshape_heads_to_batch_dim(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size * head_size, seq_len, dim // head_size)
        return tensor

    def reshape_batch_dim_to_heads(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len, dim * head_size)
        return tensor

    def forward(self, hidden_states, context=None, mask=None):
        batch_size, sequence_length, _ = hidden_states.shape

        query = self.to_q(hidden_states)
        context = context if context is not None else hidden_states
        key = self.to_k(context)
        value = self.to_v(context)

        dim = query.shape[-1]

        query = self.reshape_heads_to_batch_dim(query)
        key = self.reshape_heads_to_batch_dim(key)
        value = self.reshape_heads_to_batch_dim(value)

        # TODO(PVP) - mask is currently never used. Remember to re-implement when used

        # attention, what we cannot get enough of
        if self._use_memory_efficient_attention_xformers:
            hidden_states = self._memory_efficient_attention_xformers(query, key, value)
        else:
            if self._slice_size is None or query.shape[0] // self._slice_size == 1:
                hidden_states, attention_probs = self._attention(query, key, value)
            else:
                hidden_states = self._sliced_attention(query, key, value, sequence_length, dim)

        # linear proj
        hidden_states = self.to_out[0](hidden_states)
        # dropout
        hidden_states = self.to_out[1](hidden_states)
        return hidden_states, attention_probs

    def _attention(self, query, key, value):
        # TODO: use baddbmm for better performance
        if query.device.type == "mps":
            # Better performance on mps (~20-25%)
            attention_scores = torch.einsum("b i d, b j d -> b i j", query, key) * self.scale
        else:
            attention_scores = torch.matmul(query, key.transpose(-1, -2)) * self.scale
        attention_probs = attention_scores.softmax(dim=-1)
        # compute attention output

        if query.device.type == "mps":
            hidden_states = torch.einsum("b i j, b j d -> b i d", attention_probs, value)
        else:
            hidden_states = torch.matmul(attention_probs, value)

        # reshape hidden_states
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        return hidden_states, attention_probs

    def _sliced_attention(self, query, key, value, sequence_length, dim):
        batch_size_attention = query.shape[0]
        hidden_states = torch.zeros(
            (batch_size_attention, sequence_length, dim // self.heads), device=query.device, dtype=query.dtype
        )
        slice_size = self._slice_size if self._slice_size is not None else hidden_states.shape[0]
        for i in range(hidden_states.shape[0] // slice_size):
            start_idx = i * slice_size
            end_idx = (i + 1) * slice_size
            if query.device.type == "mps":
                # Better performance on mps (~20-25%)
                attn_slice = (
                        torch.einsum("b i d, b j d -> b i j", query[start_idx:end_idx], key[start_idx:end_idx])
                        * self.scale
                )
            else:
                attn_slice = (
                        torch.matmul(query[start_idx:end_idx], key[start_idx:end_idx].transpose(1, 2)) * self.scale
                )  # TODO: use baddbmm for better performance
            attn_slice = attn_slice.softmax(dim=-1)
            if query.device.type == "mps":
                attn_slice = torch.einsum("b i j, b j d -> b i d", attn_slice, value[start_idx:end_idx])
            else:
                attn_slice = torch.matmul(attn_slice, value[start_idx:end_idx])

            hidden_states[start_idx:end_idx] = attn_slice

        # reshape hidden_states
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        return hidden_states

    def _memory_efficient_attention_xformers(self, query, key, value):
        hidden_states = xformers.ops.memory_efficient_attention(query, key, value, attn_bias=None)
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        return hidden_states


class FeedForward(nn.Module):
    r"""
    前馈层

    Parameters:
        dim （'int'）：输入中的通道数。
        dim_out （'int'， *可选*）： 输出中的通道数。如果未给出，则默认为 'dim'。
        mult （'int'， *optional*， defaults to 4）： 用于隐藏维度的乘数。
        dropout （'float'， *optional*， defaults to 0.0）：要使用的退出概率。
        activation_fn （'str'， *可选*， 默认为 '“geglu”'）： 用于前馈的激活函数。 """

    def __init__(
            self,
            dim: int,
            dim_out: Optional[int] = None,
            mult: int = 4,
            dropout: float = 0.0,
            activation_fn: str = "geglu",
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        if activation_fn == "geglu":
            geglu = GEGLU(dim, inner_dim)
        elif activation_fn == "geglu-approximate":
            geglu = ApproximateGELU(dim, inner_dim)

        self.net = nn.ModuleList([])
        # project in
        self.net.append(geglu)
        # project dropout
        self.net.append(nn.Dropout(dropout))
        # project out
        self.net.append(nn.Linear(inner_dim, dim_out))

    def forward(self, hidden_states):
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


# feedforward
class GEGLU(nn.Module):
    r"""
    A variant of the gated linear unit activation function from https://arxiv.org/abs/2002.05202.

    Parameters:
        dim_in (`int`): The number of channels in the input.
        dim_out (`int`): The number of channels in the output.
    """

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def gelu(self, gate):
        if gate.device.type != "mps":
            return F.gelu(gate)
        # mps: gelu is not implemented for float16
        return F.gelu(gate.to(dtype=torch.float32)).to(dtype=gate.dtype)

    def forward(self, hidden_states):
        hidden_states, gate = self.proj(hidden_states).chunk(2, dim=-1)
        return hidden_states * self.gelu(gate)


class ApproximateGELU(nn.Module):
    """
    The approximate form of Gaussian Error Linear Unit (GELU)

    For more details, see section 2: https://arxiv.org/abs/1606.08415
    """

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out)

    def forward(self, x):
        x = self.proj(x)
        return x * torch.sigmoid(1.702 * x)


class AdaLayerNorm(nn.Module):
    """
    Norm layer modified to incorporate timestep embeddings.
    """

    def __init__(self, embedding_dim, num_embeddings):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings, embedding_dim)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, embedding_dim * 2)
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False)

    def forward(self, x, timestep):
        emb = self.linear(self.silu(self.emb(timestep)))
        scale, shift = torch.chunk(emb, 2)
        x = self.norm(x) * (1 + scale) + shift
        return x
