# -*- coding: utf-8 -*-
"""demo model."""
import logging
from collections import namedtuple
from typing import Any
from typing import Dict
from typing import Tuple
from typing import List
from typing import Optional

import numpy as np
import torch
from torch.autograd import Variable
from torch import Tensor
from torch import nn

from unirobot.brain.model.full_model.base_full_model import BaseFullModel
from unirobot.utils.unirobot_slot import ENCODER


logger = logging.getLogger(__name__)


def reparametrize(mu, logvar):
    """Reparameterization trick to sample from N(mu, var) from N(0,1)."""
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps


def get_sinusoid_encoding_table(n_position, d_hid):
    """Sinusoid position encoding table."""

    def get_position_angle_vec(position):
        """Get the angle for a position."""
        return [
            position / np.power(10000, 2 * (hid_j // 2) / d_hid)
            for hid_j in range(d_hid)
        ]

    sinusoid_table = np.array(
        [get_position_angle_vec(pos_i) for pos_i in range(n_position)]
    )
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class ACTModel(BaseFullModel):
    """Define Task Model.

    Args:
        sub_module_cfg (Dict[str, Any]): Config dict of sub module.
        train_mode (bool): Whether to enable training. Default=`True`.
    """

    def __init__(
        self,
        sub_module_cfg: Dict[str, Any],
        camera_names: Optional[List[str]] = None,
        state_dim: int = 14,
        chunk_size: int = 100,
        train_mode: bool = True,
        **kwargs,
    ) -> None:
        """Init Task Model base sub components."""
        super().__init__(
            sub_module_cfg=sub_module_cfg,
            train_mode=train_mode,
            **kwargs,
        )
        self._backbone = ENCODER.build(sub_module_cfg["backbone"])
        self._transformer = ENCODER.build(sub_module_cfg["transformer"])
        self._encoder = ENCODER.build(sub_module_cfg["encoder"])

        num_channels = self._backbone.num_channels
        hidden_dim = self._transformer.d_model
        self.num_queries = chunk_size
        self.camera_names = camera_names

        self.action_head = nn.Linear(hidden_dim, state_dim)
        self.is_pad_head = nn.Linear(hidden_dim, 1)
        self.query_embed = nn.Embedding(self.num_queries, hidden_dim)
        if self._backbone is not None:
            self.input_proj = nn.Conv2d(num_channels, hidden_dim, kernel_size=1)
            self.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)
        else:
            # input_dim = 14 + 7 # robot_state + env_state
            self.input_proj_robot_state = nn.Linear(14, hidden_dim)
            self.input_proj_env_state = nn.Linear(7, hidden_dim)
            self.pos = torch.nn.Embedding(2, hidden_dim)

        # encoder extra parameters
        self.latent_dim = 32  # final size of latent z # TODO tune
        self.cls_embed = nn.Embedding(1, hidden_dim)  # extra cls token embedding
        self.encoder_action_proj = nn.Linear(
            state_dim, hidden_dim
        )  # project action to embedding
        self.encoder_joint_proj = nn.Linear(
            state_dim, hidden_dim
        )  # project qpos to embedding
        self.latent_proj = nn.Linear(
            hidden_dim, self.latent_dim * 2
        )  # project hidden state to latent std, var
        self.register_buffer(
            "pos_table",
            get_sinusoid_encoding_table(1 + 1 + self.num_queries, hidden_dim),
        )  # [CLS], qpos, a_seq

        # decoder extra parameters
        self.latent_out_proj = nn.Linear(
            self.latent_dim, hidden_dim
        )  # project latent sample to embedding
        self.additional_pos_embed = nn.Embedding(
            2, hidden_dim
        )  # learned position embedding for proprio and latent
        self.init_weight()
        # self._latent_sample = torch.zeros([1, self.latent_dim], dtype=torch.float32).cuda()

    def init_weight(self) -> None:
        """Init model weight."""
        self._backbone.init_weight()

    # def parse_inputs_data(self, inputs_data):  # pylint: disable=no-self-use
    #     """Parse inputs data."""
    #     if isinstance(inputs_data, dict):
    #         return [inputs_data["image"]]
    #     return [inputs_data]

    # def wrap_outputs_data(self, outputs_data):  # pylint: disable=no-self-use
    #     """Wrap outputs data."""
    #     return {"model_outputs": outputs_data[0]}

    def infer_forward(self, inputs):
        """Inference forward.

        qpos: batch, qpos_dim
        image: batch, num_cam, channel, height, width
        env_state: None
        actions: batch, seq, action_dim
        """
        env_state = None

        qpos = inputs["qpos"]
        image = inputs["image"]
        # actions = inputs["actions"]
        # is_pad = inputs["is_pad"]

        bs, _ = qpos.shape

        mu = logvar = None
        latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(
            qpos.device
        )
        latent_input = self.latent_out_proj(latent_sample)
       
        

        if self._backbone is not None:
            # Image observation features and position embeddings
            all_cam_features = []
            all_cam_pos = []
            for cam_id, cam_name in enumerate(self.camera_names):
                features, pos = self._backbone(image[:, cam_id])  # HARDCODED
                features = features[-1]  # take the last layer feature
                pos = pos[-1]
                all_cam_features.append(self.input_proj(features))
                all_cam_pos.append(pos)
            # proprioception features
            proprio_input = self.input_proj_robot_state(qpos)
            # fold camera dimension into width dimension
            src = torch.cat(all_cam_features, axis=3)
            pos = torch.cat(all_cam_pos, axis=3)
            hs = self._transformer(
                src,
                None,
                self.query_embed.weight,
                pos,
                latent_input,
                proprio_input,
                self.additional_pos_embed.weight,
            )[0]
        else:
            qpos = self.input_proj_robot_state(qpos)
            env_state = self.input_proj_env_state(env_state)
            transformer_input = torch.cat([qpos, env_state], axis=1)  # seq length = 2
            hs = self._transformer(
                transformer_input, None, self.query_embed.weight, self.pos.weight
            )[0]
        a_hat = self.action_head(hs)
        is_pad_hat = self.is_pad_head(hs)
        return {
            "a_hat": a_hat,
            "is_pad_hat": is_pad_hat,
            "mu": mu,
            "logvar": logvar,
        }

    def train_forward(self, inputs):
        """Training forward.

        qpos: batch, qpos_dim
        image: batch, num_cam, channel, height, width
        env_state: None
        actions: batch, seq, action_dim
        """
        env_state = None
        qpos = inputs["qpos"]
        image = inputs["image"]
        actions = inputs["actions"]
        is_pad = inputs["is_pad"]

        actions = actions[:, : self.num_queries]
        is_pad = is_pad[:, : self.num_queries]
        # print( qpos.shape)
        # print( actions.shape)
        # print( is_pad.shape)
        # breakpoint()
        bs, _ = qpos.shape
        # Obtain latent z from action sequence
        # project action sequence to embedding dim, and concat with a CLS token
        action_embed = self.encoder_action_proj(actions)  # (bs, seq, hidden_dim)
        qpos_embed = self.encoder_joint_proj(qpos)  # (bs, hidden_dim)
        qpos_embed = torch.unsqueeze(qpos_embed, axis=1)  # (bs, 1, hidden_dim)
        cls_embed = self.cls_embed.weight  # (1, hidden_dim)
        cls_embed = torch.unsqueeze(cls_embed, axis=0).repeat(
            bs, 1, 1
        )  # (bs, 1, hidden_dim)
        encoder_input = torch.cat(
            [cls_embed, qpos_embed, action_embed], axis=1
        )  # (bs, seq+1, hidden_dim)
        encoder_input = encoder_input.permute(1, 0, 2)  # (seq+1, bs, hidden_dim)
        # do not mask cls token
        cls_joint_is_pad = torch.full((bs, 2), False).to(
            qpos.device
        )  # False: not a padding
        new_is_pad = torch.cat([cls_joint_is_pad, is_pad], axis=1)  # (bs, seq+1)
        # obtain position embedding
        pos_embed = self.pos_table.clone().detach()
        pos_embed = pos_embed.permute(1, 0, 2)  # (seq+1, 1, hidden_dim)
        # query model
        encoder_output = self._encoder(
            encoder_input, pos=pos_embed, src_key_padding_mask=new_is_pad
        )
        encoder_output = encoder_output[0]  # take cls output only
        latent_info = self.latent_proj(encoder_output)
        mu = latent_info[:, : self.latent_dim]
        logvar = latent_info[:, self.latent_dim :]
        latent_sample = reparametrize(mu, logvar)
        latent_input = self.latent_out_proj(latent_sample)
        if self._backbone is not None:
            # Image observation features and position embeddings
            all_cam_features = []
            all_cam_pos = []
            for cam_id, cam_name in enumerate(self.camera_names):
                features, pos = self._backbone(image[:, cam_id])  # HARDCODED
                features = features[-1]  # take the last layer feature
                pos = pos[-1]
                all_cam_features.append(self.input_proj(features))
                all_cam_pos.append(pos)
            # proprioception features
            proprio_input = self.input_proj_robot_state(qpos)
            # fold camera dimension into width dimension
            src = torch.cat(all_cam_features, axis=3)
            pos = torch.cat(all_cam_pos, axis=3)
            hs = self._transformer(
                src,
                None,
                self.query_embed.weight,
                pos,
                latent_input,
                proprio_input,
                self.additional_pos_embed.weight,
            )[0]
        else:
            qpos = self.input_proj_robot_state(qpos)
            env_state = self.input_proj_env_state(env_state)
            transformer_input = torch.cat([qpos, env_state], axis=1)  # seq length = 2
            hs = self._transformer(
                transformer_input, None, self.query_embed.weight, self.pos.weight
            )[0]

        a_hat = self.action_head(hs)
        is_pad_hat = self.is_pad_head(hs)
        return {
            "a_hat": a_hat,
            "is_pad_hat": is_pad_hat,
            "mu": mu,
            "logvar": logvar,
            "actions": actions,
            "is_pad": is_pad,
        }
