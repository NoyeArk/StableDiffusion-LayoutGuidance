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
import pdb
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.modeling_utils import ModelMixin
from diffusers.utils import BaseOutput, logging
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from .unet_2d_blocks import (
    CrossAttnDownBlock2D,
    CrossAttnUpBlock2D,
    DownBlock2D,
    UNetMidBlock2DCrossAttn,
    UpBlock2D,
    get_down_block,
    get_up_block,
)


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class UNet2DConditionOutput(BaseOutput):
    """
    Args:
        sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Hidden states conditioned on `encoder_hidden_states` input. Output of last layer of model.
    """

    sample: torch.FloatTensor


class UNet2DConditionModel(ModelMixin, ConfigMixin):
    r"""
    UNet2DConditionModel 是一个条件 2D UNet 模型，它接受噪声样本、条件状态和时间步长
    并返回样本形状输出。

    此模型继承自 ['ModelMixin']。检查超类文档，了解泛型方法、库
    适用于所有模型的实现（例如下载或保存等）

    Parameters:
        sample_size （'int'， *可选*）：输入样本的大小。
        in_channels （'int'， *可选*， 默认为 4）： 输入样本中的通道数。
        out_channels （'int'， *optional*， 默认为 4）： 输出中的通道数。
        center_input_sample （'bool'， *可选*， 默认为 'False'）：是否将输入样本居中。
        flip_sin_to_cos （'bool'， *可选*， 默认为 'False'）：
            是否在时间嵌入中将 sin 翻转为 cos。
        freq_shift （'int'， *optional*， 默认为 0）：应用于时间嵌入的频移。
        down_block_types （'Tuple[str]'， *可选*， 默认为 '（“CrossAttnDownBlock2D”， “CrossAttnDownBlock2D”， “CrossAttnDownBlock2D”， “DownBlock2D”）'）：
            要使用的下采样块的元组。
        up_block_types （'Tuple[str]'， *可选*， 默认为 '（“UpBlock2D”， “CrossAttnUpBlock2D”， “CrossAttnUpBlock2D”， “CrossAttnUpBlock2D”，）'）：
            要使用的 upsample 块的元组。
        block_out_channels （'Tuple[int]'， *可选*， 默认为 '（320， 640， 1280， 1280）'）：
            每个块的输出通道元组。
        layers_per_block （'int'， *optional*， 默认为 2）： 每个块的层数。
        downsample_padding （'int'， *optional*， 默认为 1）：用于下采样卷积的填充。
        mid_block_scale_factor （'float'， *可选*， 默认为 1.0）： 用于中间块的比例因子。
        act_fn （'str'， *可选*， 默认为 '“silu”'）： 要使用的激活函数。
        norm_num_groups （'int'， *optional*， 默认为 32）：用于规范化的组数。
        norm_eps （'float'， *optional*， 默认为 1e-5）：用于归一化的 epsilon。
        cross_attention_dim （'int'， *可选*， 默认为 1280）： 交叉注意力特征的维度。
        attention_head_dim （'int'， *可选*， 默认为 8）： 注意头的维度。
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
            self,
            sample_size: Optional[int] = None,
            in_channels: int = 4,
            out_channels: int = 4,
            center_input_sample: bool = False,
            flip_sin_to_cos: bool = True,
            freq_shift: int = 0,
            down_block_types: Tuple[str] = (
                    "CrossAttnDownBlock2D",
                    "CrossAttnDownBlock2D",
                    "CrossAttnDownBlock2D",
                    "DownBlock2D",
            ),
            up_block_types: Tuple[str] = ("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
            block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
            layers_per_block: int = 2,
            downsample_padding: int = 1,
            mid_block_scale_factor: float = 1,
            act_fn: str = "silu",
            norm_num_groups: int = 32,
            norm_eps: float = 1e-5,
            cross_attention_dim: int = 1280,
            attention_head_dim: int = 8,
    ):
        super().__init__()

        self.sample_size = sample_size
        time_embed_dim = block_out_channels[0] * 4

        # input
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, padding=(1, 1))

        # time
        self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
        timestep_input_dim = block_out_channels[0]

        self.time_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)

        self.down_blocks = nn.ModuleList([])
        self.mid_block = None
        self.up_blocks = nn.ModuleList([])

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim,
                attn_num_head_channels=attention_head_dim,
                downsample_padding=downsample_padding,
            )
            self.down_blocks.append(down_block)

        # mid
        self.mid_block = UNetMidBlock2DCrossAttn(
            in_channels=block_out_channels[-1],
            temb_channels=time_embed_dim,
            resnet_eps=norm_eps,
            resnet_act_fn=act_fn,
            output_scale_factor=mid_block_scale_factor,
            resnet_time_scale_shift="default",
            cross_attention_dim=cross_attention_dim,
            attn_num_head_channels=attention_head_dim,
            resnet_groups=norm_num_groups,
        )

        # count how many layers upsample the images
        self.num_upsamplers = 0

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            is_final_block = i == len(block_out_channels) - 1

            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

            # add upsample block for all BUT final layer
            if not is_final_block:
                add_upsample = True
                self.num_upsamplers += 1
            else:
                add_upsample = False

            up_block = get_up_block(
                up_block_type,
                num_layers=layers_per_block + 1,
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=time_embed_dim,
                add_upsample=add_upsample,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim,
                attn_num_head_channels=attention_head_dim,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=norm_eps)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, 3, padding=1)

    def set_attention_slice(self, slice_size):
        if slice_size is not None and self.config.attention_head_dim % slice_size != 0:
            raise ValueError(
                f"Make sure slice_size {slice_size} is a divisor of "
                f"the number of heads used in cross_attention {self.config.attention_head_dim}"
            )
        if slice_size is not None and slice_size > self.config.attention_head_dim:
            raise ValueError(
                f"Chunk_size {slice_size} has to be smaller or equal to "
                f"the number of heads used in cross_attention {self.config.attention_head_dim}"
            )

        for block in self.down_blocks:
            if hasattr(block, "attentions") and block.attentions is not None:
                block.set_attention_slice(slice_size)

        self.mid_block.set_attention_slice(slice_size)

        for block in self.up_blocks:
            if hasattr(block, "attentions") and block.attentions is not None:
                block.set_attention_slice(slice_size)

    def set_use_memory_efficient_attention_xformers(self, use_memory_efficient_attention_xformers: bool):
        for block in self.down_blocks:
            if hasattr(block, "attentions") and block.attentions is not None:
                block.set_use_memory_efficient_attention_xformers(use_memory_efficient_attention_xformers)

        self.mid_block.set_use_memory_efficient_attention_xformers(use_memory_efficient_attention_xformers)

        for block in self.up_blocks:
            if hasattr(block, "attentions") and block.attentions is not None:
                block.set_use_memory_efficient_attention_xformers(use_memory_efficient_attention_xformers)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (CrossAttnDownBlock2D, DownBlock2D, CrossAttnUpBlock2D, UpBlock2D)):
            module.gradient_checkpointing = value

    def forward(
            self,
            sample: torch.FloatTensor,
            timestep: Union[torch.Tensor, float, int],
            encoder_hidden_states: torch.Tensor,
            return_dict: bool = True,
    ) -> Union[UNet2DConditionOutput, Tuple]:
        r"""
        参数：
            样本 （'torch.FloatTensor'）： （batch， channel， height， width） 嘈杂的inputs_coarse张量
            timestep （'torch.FloatTensor' 或 'float' 或 'int'）： （批处理） 时间步长
            encoder_hidden_states（'torch.FloatTensor'）：（批次、通道、高度、宽度）编码器隐藏状态
            return_dict （'bool'， *可选*， 默认为 'True'）：
                是否返回 ['models.unet_2d_condition.UNet2DConditionOutput'] 而不是普通元组。

        返回：
            ['~models.unet_2d_condition.UNet2DConditionOutput'] 或 'tuple'：
            ['~models.unet_2d_condition.UNet2DConditionOutput'] 如果 'return_dict' 为 True，否则为 '元组'。什么时候
            返回一个元组，第一个元素是样本张量。
        """
        # 默认情况下，样本必须至少是总体上采样因子的倍数。
        # 总体上采样因子等于 2 **（# 上采样年数）。
        # 但是，可以强制上采样插值输出大小以适应任何上采样大小
        # 如有必要，即时提供。
        default_overall_up_factor = 2**self.num_upsamplers

        # 当样本不是 'default_overall_up_factor' 的倍数时，应转发 upsample 大小
        forward_upsample_size = False
        upsample_size = None

        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            logger.info("Forward upsample size to force interpolation output size.")
            forward_upsample_size = True

        # 0. center input if necessary
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # 以与 ONNX/Core ML 兼容的方式广播到批处理维度
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)

        # timesteps 不包含任何权重，将始终返回 f32 张量
        # 但是time_embedding实际上可能正在 FP16 中运行。所以我们需要在这里投射。
        # 可能有更好的方法来封装这一点。
        t_emb = t_emb.to(dtype=self.dtype)
        emb = self.time_embedding(t_emb)
        # 2. pre-process
        sample = self.conv_in(sample)
        # 3. down
        attn_down = []
        down_block_res_samples = (sample,)
        for block_idx, downsample_block in enumerate(self.down_blocks):
            if hasattr(downsample_block, "attentions") and downsample_block.attentions is not None:
                sample, res_samples, cross_atten_prob = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states
                )
                attn_down.append(cross_atten_prob)
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

        # 4. mid
        sample, attn_mid = self.mid_block(sample, emb, encoder_hidden_states=encoder_hidden_states)

        # 5. up
        attn_up = []
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "attentions") and upsample_block.attentions is not None:
                sample, cross_atten_prob = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    upsample_size=upsample_size,
                )
                attn_up.append(cross_atten_prob)
            else:
                sample = upsample_block(
                    hidden_states=sample, temb=emb, res_hidden_states_tuple=res_samples, upsample_size=upsample_size
                )
        # 6. post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if not return_dict:
            return (sample,)

        return UNet2DConditionOutput(sample=sample), attn_up, attn_mid, attn_down
