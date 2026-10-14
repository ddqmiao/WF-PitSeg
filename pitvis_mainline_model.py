import math
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from pitvis_mainline_common import BACKBONE_CHUNK_SIZE, FEATURE_DIM, LABELS12, ensure_repo_importable, safe_torch_load
from pitvis_coarse_to_fine import build_fine_to_coarse_index


_LABEL_TO_INDEX = {label: idx for idx, label in enumerate(LABELS12)}
LATE_CHAIN_RAW_LABELS = (9, 10, 12, 14)
LATE_CHAIN_CLASS_INDICES = tuple(_LABEL_TO_INDEX[label] for label in LATE_CHAIN_RAW_LABELS)
LATE_CHAIN_CONTEXT_RAW_LABELS = (8, 9, 10, 12, 14)
LATE_CHAIN_CONTEXT_CLASS_INDICES = tuple(_LABEL_TO_INDEX[label] for label in LATE_CHAIN_CONTEXT_RAW_LABELS)
COARSE_STAGE_INDEX_BY_FINE_CLASS = tuple(build_fine_to_coarse_index(LABELS12))
UNFREEZE_STRATEGIES = ("last_stage", "last_two_stages", "full_finetune")


class LemonFMBackboneWrapper(nn.Module):
    def __init__(self, backbone: nn.Module, channels_last: bool = False):
        super().__init__()
        self.backbone = backbone
        self.channels_last = bool(channels_last)

    @classmethod
    def _build_convnext_backbone(cls, *, weights) -> nn.Module:
        backbone = torchvision.models.convnext_large(weights=weights)
        backbone.classifier[2] = nn.Identity()
        return backbone

    @classmethod
    def from_pretrain_source(
        cls,
        repo_root: Optional[Path],
        weights_path: Optional[Path],
        *,
        pretrain_source: str = "lemonfm",
        channels_last: bool = False,
    ) -> "LemonFMBackboneWrapper":
        if pretrain_source == "none":
            backbone = cls._build_convnext_backbone(weights=None)
            return cls(backbone=backbone, channels_last=channels_last)

        if pretrain_source == "imagenet":
            weights_enum = getattr(torchvision.models, "ConvNeXt_Large_Weights", None)
            if weights_enum is None:
                raise RuntimeError(
                    "This torchvision version has no ConvNeXt_Large_Weights; "
                    "ImageNet initialization is unavailable."
                )
            imagenet_weights = getattr(weights_enum, "DEFAULT", None)
            if imagenet_weights is None:
                imagenet_weights = getattr(weights_enum, "IMAGENET1K_V1", None)
            if imagenet_weights is None:
                raise RuntimeError("No ConvNeXt-Large ImageNet weight enum is available.")
            backbone = cls._build_convnext_backbone(weights=imagenet_weights)
            return cls(backbone=backbone, channels_last=channels_last)

        if pretrain_source != "lemonfm":
            raise ValueError(f"Unsupported pretrain source: {pretrain_source}")

        if repo_root is not None:
            ensure_repo_importable(repo_root)
        if weights_path is None or not Path(weights_path).is_file():
            raise FileNotFoundError(f"Weights file does not exist: {weights_path}")
        checkpoint = safe_torch_load(Path(weights_path))
        if "teacher" not in checkpoint:
            raise KeyError("Weights file has no 'teacher' key for the LemonFM teacher backbone.")
        state_dict = checkpoint["teacher"]
        backbone_state = {
            key.replace("backbone.", ""): value
            for key, value in state_dict.items()
            if key.startswith("backbone.")
        }
        if not backbone_state:
            raise KeyError("No backbone.* parameters were found in the weights file.")
        backbone = cls._build_convnext_backbone(weights=None)
        message = backbone.load_state_dict(backbone_state, strict=False)
        if list(message.unexpected_keys):
            raise RuntimeError(f"Unexpected weight keys: {list(message.unexpected_keys)[:5]}")
        return cls(backbone=backbone, channels_last=channels_last)

    @classmethod
    def from_checkpoint(cls, repo_root: Optional[Path], weights_path: Path, channels_last: bool = False) -> "LemonFMBackboneWrapper":
        return cls.from_pretrain_source(
            repo_root=repo_root,
            weights_path=weights_path,
            pretrain_source="lemonfm",
            channels_last=channels_last,
        )

    def freeze_for_peft(self, strategy: str = "last_stage") -> None:
        if strategy not in UNFREEZE_STRATEGIES:
            raise ValueError(f"Unsupported unfreeze strategy: {strategy}")
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        if strategy == "full_finetune":
            for parameter in self.backbone.parameters():
                parameter.requires_grad = True
            return

        feature_indices = [7]
        if strategy == "last_two_stages":
            feature_indices.insert(0, 6)

        for feature_index in feature_indices:
            if feature_index < len(self.backbone.features):
                for parameter in self.backbone.features[feature_index].parameters():
                    parameter.requires_grad = True

        if len(self.backbone.classifier) > 0:
            for parameter in self.backbone.classifier[0].parameters():
                parameter.requires_grad = True

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # images: [B, L, 3, H, W]
        batch_size, clip_length, channels, height, width = images.shape
        flat = images.reshape(batch_size * clip_length, channels, height, width)
        if self.channels_last:
            flat = flat.contiguous(memory_format=torch.channels_last)
        total_frames = flat.shape[0]
        if total_frames <= BACKBONE_CHUNK_SIZE:
            feats = self.backbone(flat)
        else:
            chunks = []
            for offset in range(0, total_frames, BACKBONE_CHUNK_SIZE):
                chunks.append(self.backbone(flat[offset : offset + BACKBONE_CHUNK_SIZE]))
            feats = torch.cat(chunks, dim=0)
        return feats.reshape(batch_size, clip_length, FEATURE_DIM)


class SpatialAdapter(nn.Module):
    def __init__(self, input_dim: int = FEATURE_DIM, adapter_dim: int = 256):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.down = nn.Linear(input_dim, adapter_dim)
        self.act = nn.GELU()
        self.up = nn.Linear(adapter_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.act(self.down(self.norm(x))))


class TemporalAdapter(nn.Module):
    def __init__(self, input_dim: int = FEATURE_DIM, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        padding = max(0, int(kernel_size) // 2)
        self.norm = nn.LayerNorm(input_dim)
        self.depthwise = nn.Conv1d(
            input_dim,
            input_dim,
            kernel_size=int(kernel_size),
            padding=padding,
            groups=input_dim,
        )
        self.act = nn.GELU()
        self.pointwise = nn.Conv1d(input_dim, input_dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, C]
        normalized = self.norm(x)
        temporal = normalized.transpose(1, 2)
        temporal = self.depthwise(temporal)
        temporal = self.act(temporal)
        temporal = self.pointwise(temporal)
        temporal = self.dropout(temporal)
        return temporal.transpose(1, 2)


class ASRMFeatureAdapter(nn.Module):
    def __init__(
        self,
        input_dim: int = FEATURE_DIM,
        adapter_dim: int = 256,
        temporal_kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.spatial_adapter = SpatialAdapter(input_dim=input_dim, adapter_dim=adapter_dim)
        self.temporal_adapter = TemporalAdapter(input_dim=input_dim, kernel_size=temporal_kernel_size, dropout=dropout)
        self.router_norm = nn.LayerNorm(input_dim)
        self.router = nn.Linear(input_dim, input_dim * 2)

    def forward(self, backbone_feats: torch.Tensor) -> torch.Tensor:
        # backbone_feats: [B, L, 1536]
        x = backbone_feats
        spatial_delta = self.spatial_adapter(x)  # [B, L, 1536]
        temporal_delta = self.temporal_adapter(x)  # [B, L, 1536]
        gates = torch.sigmoid(self.router(self.router_norm(x)))  # [B, L, 3072]
        gs, gt = gates.chunk(2, dim=-1)  # [B, L, 1536], [B, L, 1536]
        adapted_feats = x + gs * spatial_delta + gt * temporal_delta  # [B, L, 1536]
        return adapted_feats


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = int(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D]
        length = x.shape[1]
        device = x.device
        dtype = x.dtype
        position = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, device=device, dtype=dtype) * (-math.log(10000.0) / self.d_model)
        )
        pe = torch.zeros((length, self.d_model), device=device, dtype=dtype)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return self.dropout(x + pe.unsqueeze(0))


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_positional_length: int, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(int(max_positional_length), int(d_model))
        self.dropout = nn.Dropout(dropout)
        self.max_positional_length = int(max_positional_length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.shape[1]
        if length > self.max_positional_length:
            raise ValueError(
                f"sequence length {length} exceeds learned positional length {self.max_positional_length}"
            )
        positions = torch.arange(length, device=x.device, dtype=torch.long)
        return self.dropout(x + self.embedding(positions).unsqueeze(0).to(dtype=x.dtype))


class IdentityPositionalEncoding(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def build_positional_encoding(
    pos_encoding_type: str,
    d_model: int,
    dropout: float,
    max_positional_length: int,
) -> nn.Module:
    pos_encoding_type = str(pos_encoding_type)
    if pos_encoding_type == "learned":
        return LearnedPositionalEncoding(d_model=d_model, max_positional_length=max_positional_length, dropout=dropout)
    if pos_encoding_type == "sinusoidal":
        return SinusoidalPositionalEncoding(d_model=d_model, dropout=dropout)
    if pos_encoding_type == "none":
        return IdentityPositionalEncoding()
    raise ValueError(f"Unknown pos_encoding_type: {pos_encoding_type}")


class LocalTransformerEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = FEATURE_DIM,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 2,
        ffn_dim: int = 512,
        dropout: float = 0.1,
        pos_encoding_type: str = "learned",
        max_positional_length: int = 128,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.positional_encoding = build_positional_encoding(
            pos_encoding_type=pos_encoding_type,
            d_model=d_model,
            dropout=dropout,
            max_positional_length=max_positional_length,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, adapted_feats: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # adapted_feats: [B, L, 1536]
        projected_feats = self.input_proj(adapted_feats)  # [B, L, d_model]
        pos_encoded_feats = self.positional_encoding(projected_feats)  # [B, L, d_model]
        transformer_out = self.encoder(
            pos_encoded_feats,
            src_key_padding_mask=padding_mask if padding_mask is not None else None,
        )
        return self.output_norm(transformer_out)


class ARStepDecoder(nn.Module):
    def __init__(
        self,
        num_classes: int = 12,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 1,
        ffn_dim: int = 512,
        dropout: float = 0.1,
        pos_encoding_type: str = "learned",
        max_positional_length: int = 128,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.bos_token_id = self.num_classes
        self.token_embedding = nn.Embedding(self.num_classes + 1, d_model)
        self.positional_encoding = build_positional_encoding(
            pos_encoding_type=pos_encoding_type,
            d_model=d_model,
            dropout=dropout,
            max_positional_length=max_positional_length,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, self.num_classes)

    def _build_causal_mask(self, length: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones((length, length), device=device, dtype=torch.bool), diagonal=1)

    def _build_teacher_forcing_tokens(
        self,
        step_targets: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len = step_targets.shape
        tokens = torch.full((batch_size, seq_len), self.bos_token_id, dtype=torch.long, device=step_targets.device)
        prev_tokens = torch.full((batch_size,), self.bos_token_id, dtype=torch.long, device=step_targets.device)
        for time_idx in range(seq_len):
            tokens[:, time_idx] = prev_tokens
            current = step_targets[:, time_idx]
            valid_mask = current.ge(0) & current.lt(self.num_classes)
            if padding_mask is not None:
                valid_mask = valid_mask & (~padding_mask[:, time_idx])
            prev_tokens = torch.where(valid_mask, current.long(), prev_tokens)
        return tokens

    def _decode_tokens(
        self,
        decoder_tokens: torch.Tensor,
        encoder_feats: torch.Tensor,
        *,
        tgt_padding_mask: Optional[torch.Tensor],
        memory_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        token_embeds = self.token_embedding(decoder_tokens)
        token_embeds = self.positional_encoding(token_embeds)
        seq_len = decoder_tokens.shape[1]
        causal_mask = self._build_causal_mask(seq_len, encoder_feats.device)
        decoded = self.decoder(
            tgt=token_embeds,
            memory=encoder_feats,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_padding_mask if tgt_padding_mask is not None else None,
            memory_key_padding_mask=memory_padding_mask if memory_padding_mask is not None else None,
        )
        return self.output_proj(decoded)

    def forward(
        self,
        encoder_feats: torch.Tensor,
        *,
        step_targets: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        teacher_forcing: bool = False,
    ) -> torch.Tensor:
        if teacher_forcing:
            if step_targets is None:
                raise ValueError("Teacher-forcing mode requires step_targets.")
            decoder_tokens = self._build_teacher_forcing_tokens(step_targets, padding_mask=padding_mask)
            return self._decode_tokens(
                decoder_tokens,
                encoder_feats,
                tgt_padding_mask=padding_mask,
                memory_padding_mask=padding_mask,
            )
        return self.autoregressive_decode(encoder_feats, padding_mask=padding_mask)

    def autoregressive_decode(
        self,
        encoder_feats: torch.Tensor,
        *,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = encoder_feats.shape
        all_logits = encoder_feats.new_zeros((batch_size, seq_len, self.num_classes))
        prev_tokens = torch.full((batch_size,), self.bos_token_id, dtype=torch.long, device=encoder_feats.device)
        decoder_tokens = torch.full((batch_size, seq_len), self.bos_token_id, dtype=torch.long, device=encoder_feats.device)

        for time_idx in range(seq_len):
            decoder_tokens[:, time_idx] = prev_tokens
            prefix_tokens = decoder_tokens[:, : time_idx + 1]
            prefix_padding = padding_mask[:, : time_idx + 1] if padding_mask is not None else None
            prefix_logits = self._decode_tokens(
                prefix_tokens,
                encoder_feats,
                tgt_padding_mask=prefix_padding,
                memory_padding_mask=padding_mask,
            )
            step_logits = prefix_logits[:, -1, :]
            if padding_mask is not None:
                padded = padding_mask[:, time_idx]
                step_logits = step_logits.masked_fill(padded.unsqueeze(-1), 0.0)
            all_logits[:, time_idx, :] = step_logits
            predicted = step_logits.argmax(dim=-1)
            if padding_mask is None:
                prev_tokens = predicted
            else:
                prev_tokens = torch.where(padding_mask[:, time_idx], prev_tokens, predicted)
        return all_logits


class ParallelStepHead(nn.Module):
    """Parallel classification head: replaces AR decoder to eliminate exposure bias."""

    def __init__(self, d_model: int, num_classes: int = 12, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(d_model, num_classes)

    def forward(self, encoder_feats: torch.Tensor, **kwargs) -> torch.Tensor:
        # encoder_feats: [B, L, d_model] -> [B, L, num_classes]
        return self.proj(self.dropout(self.norm(encoder_feats)))


class DilatedResidualBlock(nn.Module):
    """Single dilated residual block for temporal refinement (MS-TCN style).

    Matches the original MS-TCN paper: no BatchNorm (incompatible with batch_size=1).
    """

    def __init__(self, num_channels: int, dilation: int, dropout: float = 0.1, causal: bool = False):
        super().__init__()
        self.dilation = int(dilation)
        self.causal = bool(causal)
        padding = 0 if self.causal else self.dilation
        self.conv_dilated = nn.Conv1d(
            num_channels, num_channels,
            kernel_size=3, padding=padding, dilation=self.dilation,
        )
        self.conv_1x1 = nn.Conv1d(num_channels, num_channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L]
        conv_input = x
        if self.causal:
            conv_input = F.pad(conv_input, (2 * self.dilation, 0))
        out = torch.relu(self.conv_dilated(conv_input))
        out = self.conv_1x1(self.dropout(out))
        return x + out


class TemporalRefinementTCN(nn.Module):
    """Lightweight 1D-TCN that refines frame-level step logits (MS-TCN++ style).

    Input: initial step_logits [B, L, num_classes]
    Output: refined step_logits [B, L, num_classes]
    """

    def __init__(
        self,
        num_classes: int = 12,
        num_channels: int = 64,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Conv1d(num_classes, num_channels, kernel_size=1)
        self.layers = nn.ModuleList([
            DilatedResidualBlock(num_channels, dilation=2 ** i, dropout=dropout)
            for i in range(num_layers)
        ])
        self.output_proj = nn.Conv1d(num_channels, num_classes, kernel_size=1)

    def forward(self, step_logits: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # step_logits: [B, L, C] -> [B, C, L]
        x = step_logits.transpose(1, 2)
        if padding_mask is not None:
            mask = (~padding_mask).unsqueeze(1).float()  # [B, 1, L]
            x = x * mask
        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x)
        x = self.output_proj(x)
        if padding_mask is not None:
            x = x * mask
        return x.transpose(1, 2)  # [B, L, C]


class InstrumentHead(nn.Module):
    def __init__(self, d_model: int, num_instrument_classes: int):
        super().__init__()
        self.proj = nn.Linear(d_model, num_instrument_classes)

    def forward(self, encoder_feats: torch.Tensor) -> torch.Tensor:
        return self.proj(encoder_feats)


class BoundaryHead(nn.Module):
    def __init__(self, d_model: int, num_outputs: int = 1):
        super().__init__()
        self.proj = nn.Linear(d_model, int(num_outputs))

    def forward(self, encoder_feats: torch.Tensor) -> torch.Tensor:
        return self.proj(encoder_feats)


class PitVisMultitaskModel(nn.Module):
    def __init__(
        self,
        *,
        backbone_wrapper: Optional[LemonFMBackboneWrapper],
        feature_adapter: ASRMFeatureAdapter,
        encoder: LocalTransformerEncoder,
        step_head: nn.Module,
        instrument_head: InstrumentHead,
        boundary_head: BoundaryHead,
        num_step_classes: int = len(LABELS12),
        step_refine_tcn: Optional[TemporalRefinementTCN] = None,
    ):
        super().__init__()
        self.backbone_wrapper = backbone_wrapper
        self.feature_adapter = feature_adapter
        self.encoder = encoder
        self.step_head = step_head
        self.instrument_head = instrument_head
        self.boundary_head = boundary_head
        self.num_step_classes = int(num_step_classes)
        self.step_refine_tcn = step_refine_tcn
        self.stage_a_norm = nn.LayerNorm(FEATURE_DIM)
        self.stage_a_head = nn.Linear(FEATURE_DIM, self.num_step_classes)

    def extract_backbone_feats(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.dim() == 3:
            return inputs
        if self.backbone_wrapper is None:
            raise ValueError("images mode requires backbone_wrapper.")
        return self.backbone_wrapper(inputs)

    def encode_features(self, backbone_feats: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        # backbone_feats: [B, L, 1536]
        adapted_feats = self.feature_adapter(backbone_feats)  # [B, L, 1536]
        encoder_feats = self.encoder(adapted_feats, padding_mask=padding_mask)  # [B, L, d_model]
        return {
            "backbone_feats": backbone_feats,
            "adapted_feats": adapted_feats,
            "encoder_feats": encoder_feats,
        }

    def forward_stage_a(self, inputs: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        backbone_feats = self.extract_backbone_feats(inputs)
        adapted_feats = self.feature_adapter(backbone_feats)
        center_index = adapted_feats.shape[1] // 2
        center_feats = adapted_feats[:, center_index, :]
        if padding_mask is not None:
            valid_mask = (~padding_mask).float().unsqueeze(-1)
            pooled_feats = (adapted_feats * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp_min(1.0)
        else:
            pooled_feats = adapted_feats.mean(dim=1)
        stage_a_feats = 0.5 * (center_feats + pooled_feats)
        center_step_logits = self.stage_a_head(self.stage_a_norm(stage_a_feats))
        return {
            "center_step_logits": center_step_logits,
            "adapted_feats": adapted_feats,
            "backbone_feats": backbone_feats,
        }

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        padding_mask: Optional[torch.Tensor] = None,
        stage: str = "stage_b",
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        if stage == "stage_a":
            return self.forward_stage_a(inputs, padding_mask=padding_mask)

        backbone_feats = self.extract_backbone_feats(inputs)  # [B, L, 1536]
        encoded = self.encode_features(backbone_feats, padding_mask=padding_mask)
        encoder_feats = encoded["encoder_feats"]  # [B, L, d_model]
        step_logits_init = self.step_head(encoder_feats)  # [B, L, 12]
        if self.step_refine_tcn is not None:
            step_logits = self.step_refine_tcn(step_logits_init, padding_mask=padding_mask)
        else:
            step_logits = step_logits_init
        instrument_logits = self.instrument_head(encoder_feats)  # [B, L, K]
        boundary_logits = self.boundary_head(encoder_feats)  # [B, L]
        result = {
            "step_logits": step_logits,
            "instrument_logits": instrument_logits,
            "boundary_logits": boundary_logits,
            "encoder_feats": encoder_feats,
            "adapted_feats": encoded["adapted_feats"],
            "backbone_feats": backbone_feats,
        }
        if self.step_refine_tcn is not None:
            result["step_logits_init"] = step_logits_init
        return result


# ========== MS-TCN++ Full-Video Architecture ==========


class SingleStageTCN(nn.Module):
    """One TCN stage: 1x1 projection -> dilated residual blocks -> 1x1 output."""

    def __init__(self, in_channels: int, out_channels: int, num_channels: int,
                 num_layers: int, dropout: float = 0.5, causal: bool = False):
        super().__init__()
        self.in_proj = nn.Conv1d(in_channels, num_channels, 1)
        self.layers = nn.ModuleList([
            DilatedResidualBlock(num_channels, dilation=2 ** i, dropout=dropout, causal=causal)
            for i in range(num_layers)
        ])
        self.out_proj = nn.Conv1d(num_channels, out_channels, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, *, return_hidden: bool = False):
        # x: (B, in_channels, T), mask: (B, 1, T) float mask (1=valid, 0=padded)
        x = self.in_proj(x) * mask
        for layer in self.layers:
            x = layer(x) * mask
        logits = self.out_proj(x) * mask
        if return_hidden:
            return logits, x
        return logits


class MultiStageTCN(nn.Module):
    """MS-TCN: one prediction stage followed by N-1 refinement stages.

    Each refinement stage takes the softmax output of the previous stage
    and predicts refined class logits.
    """

    def __init__(self, in_channels: int, num_classes: int, num_channels: int = 64,
                 num_layers: int = 10, num_stages: int = 4, dropout: float = 0.5,
                 causal: bool = False):
        super().__init__()
        self.prediction_stage = SingleStageTCN(
            in_channels, num_classes, num_channels, num_layers, dropout, causal=causal)
        self.refinement_stages = nn.ModuleList([
            SingleStageTCN(num_classes, num_classes, num_channels, num_layers, dropout, causal=causal)
            for _ in range(num_stages - 1)
        ])

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        # x: (B, in_channels, T), mask: (B, 1, T) float
        # Returns: (list of (B, num_classes, T), last_hidden (B, num_channels, T))
        out, hidden = self.prediction_stage(x, mask, return_hidden=True)
        outputs = [out]
        for stage in self.refinement_stages:
            out, hidden = stage(torch.softmax(out, dim=1) * mask, mask, return_hidden=True)
            outputs.append(out)
        return outputs, hidden


class PitVisMSTCNModel(nn.Module):
    """Full-video MS-TCN model for surgical phase recognition.

    Replaces the window-based Transformer + small TCN architecture with a
    full-video multi-stage temporal convolutional network. Each stage sees
    the entire video, providing a receptive field of ~2047 frames with 10 layers.
    """

    def __init__(
        self,
        *,
        backbone_wrapper: Optional[LemonFMBackboneWrapper] = None,
        in_channels: int = FEATURE_DIM,
        num_step_classes: int = len(LABELS12),
        num_instrument_classes: int = 19,
        num_channels: int = 64,
        num_layers: int = 10,
        num_stages: int = 4,
        dropout: float = 0.5,
        input_dropout: float = 0.0,
        instrument_fusion: bool = False,
        num_boundary_outputs: int = 1,
        late_chain_refine: bool = False,
        late_chain_gate_power: float = 1.0,
        coarse_to_fine: bool = False,
        coarse_gate_scale: float = 1.0,
        causal: bool = False,
    ):
        super().__init__()
        self.backbone_wrapper = backbone_wrapper
        self.in_channels = int(in_channels)
        self.num_step_classes = int(num_step_classes)
        self.instrument_fusion = bool(instrument_fusion)
        self.input_dropout = nn.Dropout(float(input_dropout)) if float(input_dropout) > 0 else None
        self.num_boundary_outputs = int(num_boundary_outputs)
        self.late_chain_refine = bool(late_chain_refine)
        self.late_chain_gate_power = float(late_chain_gate_power)
        self.coarse_to_fine = bool(coarse_to_fine)
        self.coarse_gate_scale = float(coarse_gate_scale)
        self.causal = bool(causal)
        self.late_chain_class_indices = tuple(int(idx) for idx in LATE_CHAIN_CLASS_INDICES)
        self.late_chain_context_class_indices = tuple(int(idx) for idx in LATE_CHAIN_CONTEXT_CLASS_INDICES)
        fine_to_coarse_index = torch.tensor(COARSE_STAGE_INDEX_BY_FINE_CLASS, dtype=torch.long)
        self.register_buffer("fine_to_coarse_index", fine_to_coarse_index, persistent=False)
        self.num_coarse_stages = int(fine_to_coarse_index.max().item()) + 1

        if self.instrument_fusion:
            # Instrument encoder: extracts instrument-aware hidden features
            self.instrument_encoder = nn.Sequential(
                nn.Conv1d(in_channels, num_channels, 1),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            # Instrument output head: produces logits for instrument loss
            self.instrument_out = nn.Conv1d(num_channels, num_instrument_classes, 1)
            # MS-TCN sees original features + instrument hidden features
            mstcn_in = in_channels + num_channels
        else:
            self.instrument_proj = nn.Sequential(
                nn.Conv1d(in_channels, num_channels, 1),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv1d(num_channels, num_instrument_classes, 1),
            )
            mstcn_in = in_channels

        self.mstcn = MultiStageTCN(
            mstcn_in, num_step_classes, num_channels, num_layers, num_stages, dropout, causal=self.causal)
        self.boundary_proj = nn.Sequential(
            nn.Conv1d(num_channels, num_channels, 1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(num_channels, self.num_boundary_outputs, 1),
        )
        if self.coarse_to_fine:
            self.coarse_proj = nn.Sequential(
                nn.Conv1d(num_channels, num_channels, 1),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv1d(num_channels, self.num_coarse_stages, 1),
            )
        else:
            self.coarse_proj = None
        if self.late_chain_refine:
            late_chain_in_channels = num_channels + (num_channels if self.instrument_fusion else 0)
            self.late_chain_refiner = nn.Sequential(
                nn.Conv1d(late_chain_in_channels, num_channels, 1),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv1d(num_channels, len(self.late_chain_class_indices), 1),
            )
        else:
            self.late_chain_refiner = None

    def extract_backbone_feats(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.dim() == 3:
            return inputs
        if self.backbone_wrapper is None:
            raise ValueError("images mode requires backbone_wrapper.")
        return self.backbone_wrapper(inputs)

    def _compute_late_chain_gate(self, step_logits: torch.Tensor) -> torch.Tensor:
        late_context_prob = torch.softmax(step_logits.float(), dim=-1)[..., self.late_chain_context_class_indices].sum(dim=-1, keepdim=True)
        gate_power = max(1e-6, float(self.late_chain_gate_power))
        if abs(gate_power - 1.0) > 1e-6:
            late_context_prob = late_context_prob.clamp_min(1e-6) ** gate_power
        return late_context_prob.clamp(0.0, 1.0).to(dtype=step_logits.dtype, device=step_logits.device)

    def _compute_coarse_bias(self, coarse_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        coarse_log_probs = F.log_softmax(coarse_logits.float(), dim=-1)
        coarse_probs = torch.softmax(coarse_logits.float(), dim=-1)
        gather_index = self.fine_to_coarse_index.view(1, 1, -1).expand(
            coarse_log_probs.shape[0],
            coarse_log_probs.shape[1],
            -1,
        )
        coarse_bias = torch.gather(coarse_log_probs, dim=-1, index=gather_index)
        coarse_bias = float(self.coarse_gate_scale) * coarse_bias.to(dtype=coarse_logits.dtype, device=coarse_logits.device)
        return coarse_bias, coarse_probs.to(dtype=coarse_logits.dtype, device=coarse_logits.device)

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        mask: torch.Tensor,
        **kwargs,
    ) -> Dict[str, object]:
        # inputs: (B, T, in_channels) features or images
        # mask: (B, T) bool; True for valid frames
        features = self.extract_backbone_feats(inputs)  # (B, T, C)
        if self.input_dropout is not None:
            features = self.input_dropout(features)
        float_mask = mask.unsqueeze(1).float()  # (B, 1, T)
        x = features.transpose(1, 2)  # (B, C, T)
        inst_hidden = None

        if self.instrument_fusion:
            # Early injection: compute instrument hidden features and enrich input
            inst_hidden = self.instrument_encoder(x * float_mask)   # (B, num_channels, T)
            instrument_logits = self.instrument_out(inst_hidden)    # (B, K, T)
            x_enriched = torch.cat([x, inst_hidden], dim=1)        # (B, C+num_channels, T)
            all_stage_logits, last_hidden = self.mstcn(x_enriched, float_mask)
        else:
            all_stage_logits, last_hidden = self.mstcn(x, float_mask)
            instrument_logits = self.instrument_proj(x * float_mask)  # (B, K, T)

        boundary_logits = self.boundary_proj(last_hidden * float_mask).transpose(1, 2)  # (B, T, Kb)
        all_stage_logits = [logits.transpose(1, 2) for logits in all_stage_logits]
        coarse_logits = None
        coarse_probs = None
        if self.coarse_proj is not None:
            coarse_logits = self.coarse_proj(last_hidden * float_mask).transpose(1, 2)
            coarse_bias, coarse_probs = self._compute_coarse_bias(coarse_logits)
            all_stage_logits = [logits + coarse_bias for logits in all_stage_logits]
        refined_step_logits = all_stage_logits[-1]
        late_chain_logits = None
        late_chain_gate = None
        late_chain_delta = None

        if self.late_chain_refiner is not None:
            late_inputs = [last_hidden]
            if inst_hidden is not None:
                late_inputs.append(inst_hidden)
            late_chain_delta = self.late_chain_refiner(torch.cat(late_inputs, dim=1) * float_mask).transpose(1, 2)
            late_chain_gate = self._compute_late_chain_gate(refined_step_logits)
            refined_step_logits = refined_step_logits.clone()
            refined_step_logits[..., self.late_chain_class_indices] = (
                refined_step_logits[..., self.late_chain_class_indices]
                + late_chain_gate * late_chain_delta
            )
            late_chain_logits = late_chain_delta
            all_stage_logits[-1] = refined_step_logits

        return {
            "all_stage_logits": all_stage_logits,
            "step_logits": refined_step_logits,
            "instrument_logits": instrument_logits.transpose(1, 2),
            "boundary_logits": boundary_logits,
            "coarse_logits": coarse_logits,
            "coarse_probs": coarse_probs,
            "late_chain_logits": late_chain_logits,
            "late_chain_gate": late_chain_gate,
        }


# ========== ASFormer: Transformer for Action Segmentation ==========


def _exponential_decrease(idx: int, p: float = 3.0) -> float:
    return math.exp(-p * idx)


class ASFormerFeedForward(nn.Module):
    """Dilated conv feed-forward used in each ASFormer attention module."""

    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.conv(x))


class ASFormerAttention(nn.Module):
    """Single-head attention for ASFormer with project-friendly local/full modes."""

    def __init__(
        self,
        channels: int,
        r: int = 1,
        window_size: int = 1,
        attention_type: str = "local",
        dropout: float = 0.0,
    ):
        super().__init__()
        reduction = max(1, int(r))
        d = max(1, int(channels) // reduction)
        attention_type = str(attention_type).lower()
        if attention_type not in {"local", "full"}:
            raise ValueError(f"Unsupported ASFormer attention type: {attention_type}")
        self.query_conv = nn.Conv1d(channels, d, 1)
        self.key_conv = nn.Conv1d(channels, d, 1)
        self.value_conv = nn.Conv1d(channels, d, 1)
        self.conv_out = nn.Conv1d(d, channels, 1)
        self.attn_dropout = nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity()
        self.d = d
        self.window_size = max(1, int(window_size))
        self.attention_type = attention_type

    def _forward_full(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        scale = math.sqrt(self.d)
        energy = torch.bmm(q.transpose(1, 2), k) / scale  # (B, L, L)
        # Apply mask: invalid key positions get large negative value.
        # Use 1e4 instead of 1e9; bf16 max is ~65504, so 1e9 overflows to inf and produces NaN in softmax.
        energy = energy.masked_fill(mask < 0.5, -1e4)
        attn = F.softmax(energy, dim=-1)
        attn = self.attn_dropout(attn)
        # Zero out attention rows for padded query positions.
        attn = attn * mask.transpose(1, 2)
        return torch.bmm(v, attn.transpose(1, 2))

    def _forward_local(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if q.shape[-1] != k.shape[-1] or q.shape[-1] != v.shape[-1]:
            raise ValueError("ASFormer local attention requires aligned query/key/value sequence lengths.")
        window_radius = self.window_size
        span = 2 * window_radius + 1
        q = q / math.sqrt(self.d)
        padded_k = F.pad(k, (window_radius, window_radius))
        padded_v = F.pad(v, (window_radius, window_radius))
        padded_mask = F.pad(mask, (window_radius, window_radius))

        k_windows = padded_k.unfold(dimension=-1, size=span, step=1)  # (B, d, L, W)
        v_windows = padded_v.unfold(dimension=-1, size=span, step=1)  # (B, d, L, W)
        key_mask = padded_mask.unfold(dimension=-1, size=span, step=1).squeeze(1)  # (B, L, W)

        energy = (q.unsqueeze(-1) * k_windows).sum(dim=1)  # (B, L, W)
        energy = energy.masked_fill(key_mask < 0.5, -1e4)
        attn = F.softmax(energy, dim=-1)
        attn = self.attn_dropout(attn)
        attn = attn * mask.transpose(1, 2)
        return (v_windows * attn.unsqueeze(1)).sum(dim=-1)

    def forward(
        self,
        x_q: torch.Tensor,
        x_k: torch.Tensor,
        mask: torch.Tensor,
        x_v: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x_q: (B, C, L) query source
        x_k: (B, C, L) key source
        x_v: (B, C, L) value source; defaults to x_k for self-attn
        mask: (B, 1, L) float, 1=valid
        """
        if x_v is None:
            x_v = x_k
        q = self.query_conv(x_q)
        k = self.key_conv(x_k)
        v = self.value_conv(x_v)
        if self.attention_type == "full":
            out = self._forward_full(q, k, v, mask)
        else:
            out = self._forward_local(q, k, v, mask)
        out = self.conv_out(F.relu(out))
        return out * mask


class ASFormerAttModule(nn.Module):
    """One ASFormer layer: dilated-conv feed-forward + attention + 1x1 + residual."""

    def __init__(
        self,
        channels: int,
        dilation: int,
        r: int,
        stage: str,
        alpha: float,
        dropout: float = 0.5,
        attention_type: str = "local",
        attention_dropout: float = 0.0,
        decoder_key_source: str = "decoder",
    ):
        super().__init__()
        self.feed_forward = ASFormerFeedForward(channels, dilation)
        self.instance_norm = nn.InstanceNorm1d(channels, track_running_stats=False)
        self.att = ASFormerAttention(
            channels,
            r=r,
            window_size=dilation,
            attention_type=attention_type,
            dropout=attention_dropout,
        )
        self.conv_1x1 = nn.Conv1d(channels, channels, 1)
        self.dropout = nn.Dropout(dropout)
        self.alpha = alpha
        self.stage = stage
        self.decoder_key_source = str(decoder_key_source).lower()
        if self.decoder_key_source not in {"decoder", "encoder"}:
            raise ValueError(f"Unsupported ASFormer decoder key source: {decoder_key_source}")

    def forward(self, x: torch.Tensor, cross_feats: Optional[torch.Tensor], mask: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, L) input
        cross_feats: (B, C, L) encoder hidden (for decoder cross-attention) or None
        mask: (B, 1, L) float
        """
        out = self.feed_forward(x)
        normed = self.instance_norm(out)
        if self.stage == "decoder" and cross_feats is not None:
            # Original-style decoder uses decoder features for keys and encoder features for values.
            key_source = normed if self.decoder_key_source == "decoder" else cross_feats
            att_out = self.att(normed, key_source, mask, x_v=cross_feats)
        else:
            # Self-attention
            att_out = self.att(normed, normed, mask)
        out = self.alpha * att_out + out
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out) * mask


class ASFormerEncoder(nn.Module):
    """ASFormer encoder: 1x1 projection -> N dilated attention layers -> class output."""

    def __init__(
        self, num_layers: int, channels: int, in_channels: int, num_classes: int,
        r: int = 1, dropout: float = 0.5, channel_masking_rate: float = 0.3,
        attention_type: str = "local", attention_dropout: float = 0.0,
    ):
        super().__init__()
        self.conv_1x1 = nn.Conv1d(in_channels, channels, 1)
        self.layers = nn.ModuleList([
            ASFormerAttModule(
                channels,
                dilation=2 ** i,
                r=r,
                stage="encoder",
                alpha=1.0,
                dropout=dropout,
                attention_type=attention_type,
                attention_dropout=attention_dropout,
            )
            for i in range(num_layers)
        ])
        self.conv_out = nn.Conv1d(channels, num_classes, 1)
        self.channel_masking_rate = float(channel_masking_rate)
        self.channel_dropout = nn.Dropout2d(p=channel_masking_rate) if channel_masking_rate > 0 else None

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        """
        x: (B, in_channels, L)
        mask: (B, 1, L) float
        Returns: (logits (B, num_classes, L), hidden (B, channels, L))
        """
        if self.channel_dropout is not None and self.training:
            x = self.channel_dropout(x.unsqueeze(2)).squeeze(2)

        feature = self.conv_1x1(x)
        for layer in self.layers:
            feature = layer(feature, None, mask)

        out = self.conv_out(feature) * mask
        return out, feature


class ASFormerDecoder(nn.Module):
    """ASFormer decoder: cross-attention to encoder features + self-refinement."""

    def __init__(
        self, num_layers: int, channels: int, num_classes: int,
        r: int = 1, alpha: float = 1.0, dropout: float = 0.5,
        attention_type: str = "local", attention_dropout: float = 0.0,
        decoder_key_source: str = "decoder",
    ):
        super().__init__()
        self.conv_1x1 = nn.Conv1d(num_classes, channels, 1)
        self.layers = nn.ModuleList([
            ASFormerAttModule(
                channels,
                dilation=2 ** i,
                r=r,
                stage="decoder",
                alpha=alpha,
                dropout=dropout,
                attention_type=attention_type,
                attention_dropout=attention_dropout,
                decoder_key_source=decoder_key_source,
            )
            for i in range(num_layers)
        ])
        self.conv_out = nn.Conv1d(channels, num_classes, 1)

    def forward(self, x: torch.Tensor, encoder_feats: torch.Tensor, mask: torch.Tensor):
        """
        x: (B, num_classes, L); softmax output from previous stage
        encoder_feats: (B, channels, L)
        mask: (B, 1, L) float
        Returns: (logits (B, num_classes, L), hidden (B, channels, L))
        """
        feature = self.conv_1x1(x)
        for layer in self.layers:
            feature = layer(feature, encoder_feats, mask)
        out = self.conv_out(feature) * mask
        return out, feature


class PitVisASFormerModel(nn.Module):
    """ASFormer model for surgical phase recognition.

    Drop-in replacement for PitVisMSTCNModel with the same interface:
    - 1 encoder stage + N decoder stages (cross-attention back to encoder)
    - Same forward() signature and return dict
    """

    def __init__(
        self,
        *,
        backbone_wrapper: Optional[LemonFMBackboneWrapper] = None,
        in_channels: int = FEATURE_DIM,
        num_step_classes: int = len(LABELS12),
        num_instrument_classes: int = 19,
        num_channels: int = 64,
        num_layers: int = 10,
        num_decoders: int = 3,
        r: int = 1,
        dropout: float = 0.5,
        attention_dropout: float = 0.1,
        attention_type: str = "local",
        decoder_key_source: str = "decoder",
        input_dropout: float = 0.0,
        feature_layernorm: bool = True,
        channel_masking_rate: float = 0.3,
        instrument_fusion: bool = False,
        num_boundary_outputs: int = 1,
        coarse_to_fine: bool = False,
        coarse_gate_scale: float = 1.0,
    ):
        super().__init__()
        self.backbone_wrapper = backbone_wrapper
        self.in_channels = int(in_channels)
        self.num_step_classes = int(num_step_classes)
        self.instrument_fusion = bool(instrument_fusion)
        self.feature_norm = nn.LayerNorm(in_channels) if bool(feature_layernorm) else None
        self.input_dropout = nn.Dropout(float(input_dropout)) if float(input_dropout) > 0 else None
        self.num_boundary_outputs = int(num_boundary_outputs)
        self.coarse_to_fine = bool(coarse_to_fine)
        self.coarse_gate_scale = float(coarse_gate_scale)
        fine_to_coarse_index = torch.tensor(COARSE_STAGE_INDEX_BY_FINE_CLASS, dtype=torch.long)
        self.register_buffer("fine_to_coarse_index", fine_to_coarse_index, persistent=False)
        self.num_coarse_stages = int(fine_to_coarse_index.max().item()) + 1

        if self.instrument_fusion:
            self.instrument_encoder = nn.Sequential(
                nn.Conv1d(in_channels, num_channels, 1),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.instrument_out = nn.Conv1d(num_channels, num_instrument_classes, 1)
            encoder_in = in_channels + num_channels
        else:
            self.instrument_proj = nn.Sequential(
                nn.Conv1d(in_channels, num_channels, 1),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv1d(num_channels, num_instrument_classes, 1),
            )
            encoder_in = in_channels

        self.encoder = ASFormerEncoder(
            num_layers=num_layers, channels=num_channels, in_channels=encoder_in,
            num_classes=num_step_classes, r=r, dropout=dropout,
            channel_masking_rate=channel_masking_rate,
            attention_type=attention_type,
            attention_dropout=attention_dropout,
        )
        self.decoders = nn.ModuleList([
            ASFormerDecoder(
                num_layers=num_layers, channels=num_channels,
                num_classes=num_step_classes, r=r,
                alpha=_exponential_decrease(s), dropout=dropout,
                attention_type=attention_type,
                attention_dropout=attention_dropout,
                decoder_key_source=decoder_key_source,
            )
            for s in range(num_decoders)
        ])

        self.boundary_proj = nn.Sequential(
            nn.Conv1d(num_channels, num_channels, 1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(num_channels, self.num_boundary_outputs, 1),
        )
        if self.coarse_to_fine:
            self.coarse_proj = nn.Sequential(
                nn.Conv1d(num_channels, num_channels, 1),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv1d(num_channels, self.num_coarse_stages, 1),
            )
        else:
            self.coarse_proj = None

    def extract_backbone_feats(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.dim() == 3:
            return inputs
        if self.backbone_wrapper is None:
            raise ValueError("images mode requires backbone_wrapper.")
        return self.backbone_wrapper(inputs)

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        mask: torch.Tensor,
        **kwargs,
    ) -> Dict[str, object]:
        features = self.extract_backbone_feats(inputs)  # (B, T, C)
        if self.feature_norm is not None:
            features = self.feature_norm(features)
        if self.input_dropout is not None:
            features = self.input_dropout(features)
        float_mask = mask.unsqueeze(1).float()  # (B, 1, T)
        x = features.transpose(1, 2)  # (B, C, T)

        if self.instrument_fusion:
            inst_hidden = self.instrument_encoder(x * float_mask)
            instrument_logits = self.instrument_out(inst_hidden)
            x_enriched = torch.cat([x, inst_hidden], dim=1)
            enc_out, enc_hidden = self.encoder(x_enriched, float_mask)
        else:
            enc_out, enc_hidden = self.encoder(x, float_mask)
            instrument_logits = self.instrument_proj(x * float_mask)

        all_stage_logits_ct = [enc_out]  # channel-first: (B, C, T)
        out = enc_out
        for decoder in self.decoders:
            out, dec_hidden = decoder(
                F.softmax(out, dim=1) * float_mask,
                enc_hidden * float_mask,
                float_mask,
            )
            all_stage_logits_ct.append(out)

        # Use last decoder hidden for boundary prediction
        last_hidden = dec_hidden if self.decoders else enc_hidden
        boundary_logits = self.boundary_proj(last_hidden * float_mask).transpose(1, 2)  # (B, T, Kb)
        coarse_logits = None
        coarse_probs = None

        # Convert to (B, T, C) convention
        all_stage_logits = [logits.transpose(1, 2) for logits in all_stage_logits_ct]
        if self.coarse_proj is not None:
            coarse_logits = self.coarse_proj(last_hidden * float_mask).transpose(1, 2)
            coarse_log_probs = F.log_softmax(coarse_logits.float(), dim=-1)
            coarse_probs = torch.softmax(coarse_logits.float(), dim=-1).to(dtype=coarse_logits.dtype, device=coarse_logits.device)
            gather_index = self.fine_to_coarse_index.view(1, 1, -1).expand(
                coarse_log_probs.shape[0],
                coarse_log_probs.shape[1],
                -1,
            )
            coarse_bias = torch.gather(coarse_log_probs, dim=-1, index=gather_index)
            coarse_bias = float(self.coarse_gate_scale) * coarse_bias.to(dtype=coarse_logits.dtype, device=coarse_logits.device)
            all_stage_logits = [logits + coarse_bias for logits in all_stage_logits]

        return {
            "all_stage_logits": all_stage_logits,
            "step_logits": all_stage_logits[-1],
            "instrument_logits": instrument_logits.transpose(1, 2),
            "boundary_logits": boundary_logits,
            "coarse_logits": coarse_logits,
            "coarse_probs": coarse_probs,
        }


__all__ = [
    "ARStepDecoder",
    "ASFormerAttention",
    "ASFormerAttModule",
    "ASFormerDecoder",
    "ASFormerEncoder",
    "ASFormerFeedForward",
    "ASRMFeatureAdapter",
    "BoundaryHead",
    "IdentityPositionalEncoding",
    "InstrumentHead",
    "LATE_CHAIN_CLASS_INDICES",
    "LATE_CHAIN_RAW_LABELS",
    "LearnedPositionalEncoding",
    "LemonFMBackboneWrapper",
    "LocalTransformerEncoder",
    "MultiStageTCN",
    "ParallelStepHead",
    "PitVisASFormerModel",
    "PitVisMSTCNModel",
    "PitVisMultitaskModel",
    "SingleStageTCN",
    "SinusoidalPositionalEncoding",
    "TemporalRefinementTCN",
]
