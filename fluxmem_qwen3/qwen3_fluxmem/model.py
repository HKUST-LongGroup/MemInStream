"""FluxMem selection before native Qwen3-VL decoder prefill (Transformers 4.57.1).

The memory algorithm is copied unchanged. Its spatial/temporal bookkeeping uses
frame-local coordinates, while the decoder retains native Qwen3 mRoPE positions.
DeepStack layers are indexed by the same retained visual token indices.
"""
import torch
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLForConditionalGeneration, Qwen3VLCausalLMOutputWithPast,
)
from .fluxmem import FluxMem
from .utils import scan_visual_indices


def memory_coordinates(input_ids, grid, config):
    """Qwen3 offsets h/w by timestamp text; FluxMem needs local grid coordinates."""
    if input_ids.shape[0] != 1 or grid.shape != (1, 3):
        raise ValueError('FluxMem Qwen3 evaluation supports batch size 1 and one video per sample')
    t, h, w = map(int, grid[0].tolist())
    merge = config.vision_config.spatial_merge_size
    if h % merge or w % merge:
        raise ValueError('Video grid is not divisible by spatial_merge_size')
    h, w = h // merge, w // merge
    visual = (input_ids[0] == config.video_token_id).nonzero(as_tuple=True)[0]
    scanned = scan_visual_indices(input_ids[0], config.vision_start_token_id, config.vision_end_token_id)
    if not torch.equal(scanned, visual) or visual.numel() != t * h * w:
        raise ValueError('Expected only video patches inside vision delimiters, matching video_grid_thw')
    coords = torch.zeros((3, 1, input_ids.shape[1]), device=input_ids.device, dtype=torch.long)
    coords[0, 0, visual] = torch.arange(t, device=input_ids.device).repeat_interleave(h*w)
    coords[1, 0, visual] = torch.arange(h, device=input_ids.device).repeat_interleave(w).repeat(t)
    coords[2, 0, visual] = torch.arange(w, device=input_ids.device).repeat(t*h)
    return coords, visual


def encode_video(model, pixels, grid, chunk_t=None):
    """Chunk by temporal grids; Qwen3 vision attention is independent per grid."""
    if grid.shape != (1, 3):
        raise ValueError('One video per sample is required')
    t, h, w = map(int, grid[0].tolist())
    chunk_t = t if chunk_t is None else int(chunk_t)
    if chunk_t <= 0:
        raise ValueError('video_chunk_t must be positive')
    embeddings, deep_layers = [], None
    for start in range(0, t, chunk_t):
        count = min(chunk_t, t-start)
        part_grid = grid.clone(); part_grid[0, 0] = count
        features, deep = model.model.get_video_features(pixels[start*h*w:(start+count)*h*w], part_grid)
        embeddings.append(torch.cat(features, dim=0))
        if deep_layers is None:
            deep_layers = [[] for _ in deep]
        if len(deep) != len(deep_layers):
            raise ValueError('Inconsistent DeepStack output count')
        for acc, feature in zip(deep_layers, deep):
            acc.append(feature)
    return torch.cat(embeddings, dim=0), [torch.cat(parts, dim=0) for parts in deep_layers]


@torch.inference_mode()
def prepare_compressed_inputs(model, inputs, short_frames=8, medium_frames=64, video_chunk_t=None):
    """Compress final vision features and carry original positional/DeepStack data."""
    ids = inputs['input_ids']
    if short_frames < 2 or short_frames % 2 or medium_frames < 0 or medium_frames % 2:
        raise ValueError('short_frames >=2 and medium_frames >=0 must be even')
    if ids.shape[0] != 1 or inputs.get('pixel_values') is not None:
        raise ValueError('Only one unpadded video-only sample is supported')
    mask = inputs.get('attention_mask', torch.ones_like(ids))
    if not bool(mask.bool().all()):
        raise ValueError('Padding is not supported in single-sample prefill')
    grid = inputs['video_grid_thw']
    coords, visual = memory_coordinates(ids, grid, model.config)
    original_positions, _ = model.model.get_rope_index(ids, video_grid_thw=grid, attention_mask=mask)
    embeds = model.get_input_embeddings()(ids)
    features, deepstack = encode_video(model, inputs['pixel_values_videos'], grid, video_chunk_t)
    features = features.to(embeds.device, embeds.dtype)
    if features.shape[0] != visual.numel():
        raise ValueError('Vision features do not match video placeholder count')
    embeds[0, visual] = features
    memory = FluxMem(model.config.vision_start_token_id, model.config.vision_end_token_id,
                     short_term_frames=short_frames, mid_term_frames=medium_frames)
    memory.temporal_patch_size = model.config.vision_config.temporal_patch_size
    # RoPE values are not used by the selection algorithm. Native Qwen3 computes
    # actual rotary embeddings later, from the original retained position_ids.
    bookkeeping = embeds.new_zeros((3, 1, ids.shape[1], 1))
    packed, _, _, packed_mask, kept_list = memory.process_memory_streaming(
        embeds, coords, (bookkeeping, bookkeeping), ids, video_grid_thw=grid)
    kept = kept_list[0]
    visual_mask = ids.eq(model.config.video_token_id)
    kept_visual_mask = visual_mask[:, kept]
    visual_ordinals = visual_mask[0].long().cumsum(0)-1
    deep_indices = visual_ordinals[kept[kept_visual_mask[0]]]
    packed_deep = [layer.index_select(0, deep_indices.to(layer.device)) for layer in deepstack]
    positions = original_positions[:, :, kept]
    # The continuation starts at native max-position+1, regardless of pruning.
    rope_delta = original_positions.amax(dim=(0, 2)).reshape(-1, 1) + 1 - kept.numel()
    kwargs = dict(input_ids=ids[:, kept], inputs_embeds=packed, attention_mask=packed_mask,
                  flux_position_ids=positions, flux_rope_delta=rope_delta,
                  flux_visual_mask=kept_visual_mask, flux_deepstack=packed_deep)
    stats = dict(input_tokens=int(ids.shape[1]), retained_tokens=int(kept.numel()),
                 visual_tokens=int(visual.numel()), retained_visual_tokens=int(deep_indices.numel()),
                 temporal_grids=int(grid[0, 0]), deepstack_layers=len(packed_deep),
                 short_frames=short_frames, medium_frames=medium_frames, video_chunk_t=video_chunk_t)
    return kwargs, stats


class Qwen3VLFluxMemForConditionalGeneration(Qwen3VLForConditionalGeneration):
    """Use native Qwen3 text layers and native generation with a compressed prefill.

    No global monkey patches or per-request mutable hooks. Native forward is
    preserved for callers without flux_position_ids. Beam search is unsupported.
    """
    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, cache_position=None,
                flux_position_ids=None, flux_rope_delta=None, flux_visual_mask=None,
                flux_deepstack=None, logits_to_keep=0, **kwargs):
        if flux_position_ids is None:
            return super().forward(input_ids=input_ids, attention_mask=attention_mask,
                position_ids=position_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds,
                cache_position=cache_position, logits_to_keep=logits_to_keep, **kwargs)
        if kwargs.get('labels') is not None:
            raise ValueError('FluxMem adapter is inference-only')
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
        if inputs_embeds.shape[0] != 1:
            raise ValueError('FluxMem adapter supports batch size 1, num_beams=1, num_return_sequences=1')
        past = past_key_values.get_seq_length() if past_key_values is not None else 0
        prefill = past == 0
        if cache_position is None:
            cache_position = torch.arange(past, past+inputs_embeds.shape[1], device=inputs_embeds.device)
        if prefill:
            if inputs_embeds.shape[1] != flux_position_ids.shape[-1]:
                raise ValueError('Compressed prefill length does not match positions')
            position_ids = flux_position_ids
        else:
            pos = cache_position.reshape(1, -1) + flux_rope_delta.to(cache_position.device)
            position_ids = pos.unsqueeze(0).expand(3, -1, -1)
        # Pixel arguments carried by native generation preparation must not be
        # forwarded into text attention. Vision has already been encoded once.
        for key in ('pixel_values', 'pixel_values_videos', 'image_grid_thw', 'video_grid_thw', 'labels'):
            kwargs.pop(key, None)
        outputs = self.model.language_model(input_ids=None, inputs_embeds=inputs_embeds,
            attention_mask=attention_mask, position_ids=position_ids, past_key_values=past_key_values,
            cache_position=cache_position, visual_pos_masks=flux_visual_mask if prefill else None,
            deepstack_visual_embeds=flux_deepstack if prefill else None, **kwargs)
        indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(outputs.last_hidden_state[:, indices, :])
        return Qwen3VLCausalLMOutputWithPast(logits=logits, past_key_values=outputs.past_key_values,
                                            rope_deltas=flux_rope_delta)
