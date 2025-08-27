# SPDX-License-Identifier: Apache-2.0
from typing import TYPE_CHECKING, Optional

import torch
from lmcache.integration.vllm.vllm_v1_adapter import LMCacheConnectorV1Impl

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole)
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request

logger = init_logger(__name__)


class LMCacheConnectorV1(KVConnectorBase_V1):

    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole):
        super().__init__(vllm_config=vllm_config, role=role)
        self._lmcache_engine = LMCacheConnectorV1Impl(vllm_config, role, self)

    # ==============================
    # Worker-side methods
    # ==============================
    def start_load_kv(self, forward_context: "ForwardContext",
                      **kwargs) -> None:
        """
        Start loading the KV cache from the connector to vLLM's paged
        KV buffer. This is called from the forward context before the
        forward pass to enable async loading during model execution.

        Args:
            forward_context (ForwardContext): the forward context.
            **kwargs: additional arguments for the load operation

        Note:
            The number of elements in kv_caches and layer_names should be 
            the same.
            
        """
        self._lmcache_engine.start_load_kv(forward_context, **kwargs)

    def wait_for_layer_load(self, layer_name: str) -> None:
        """
        Block until the KV for a specific layer is loaded into vLLM's
        paged buffer. This is called from within attention layer to ensure
        async copying from start_load_kv is complete.
        
        This interface will be useful for layer-by-layer pipelining.

        Args:
            layer_name: the name of that layer
        """
        self._lmcache_engine.wait_for_layer_load(layer_name)

    def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor,
                      attn_metadata: "AttentionMetadata", **kwargs) -> None:
        """
        Start saving the a layer of KV cache from vLLM's paged buffer 
        to the connector. This is called from within attention layer to
        enable async copying during execution.

        Args:
            layer_name (str): the name of the layer.
            kv_layer (torch.Tensor): the paged KV buffer of the current 
                layer in vLLM.
            attn_metadata (AttentionMetadata): the attention metadata.
            **kwargs: additional arguments for the save operation.
        """
        self._lmcache_engine.save_kv_layer(layer_name, kv_layer, attn_metadata,
                                           **kwargs)

    def wait_for_save(self):
        """
        Block until all the save operations is done. This is called
        as the forward context exits to ensure that the async saving
        from save_kv_layer is complete before finishing the forward.

        This prevents overwrites of paged KV buffer before saving done.
        """
        self._lmcache_engine.wait_for_save()

    def publish_first_decode_logits(self, req_id: str, logits: torch.Tensor) -> None:
        """
        Publish the last prompt token logits for a request to enable
        zero-compute first decode on the decoder worker.
        """
        self._lmcache_engine.publish_first_decode_logits(req_id, logits)

    def try_consume_first_decode_logits(self, req_id: str) -> Optional[torch.Tensor]:
        """
        Try to retrieve the first decode logits for a request.
        """
        return self._lmcache_engine.try_consume_first_decode_logits(req_id)

    # ==============================
    # Scheduler-side methods
    # ==============================
    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
        **kwargs,
    ) -> tuple[int, bool]:
        """
        Get number of new tokens that can be loaded from the
        external KV cache beyond the num_computed_tokens.
        
        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request

        Returns:
            the number of tokens that can be loaded from the 
            external KV cache beyond what is already computed.
        """
        return self._lmcache_engine.get_num_new_matched_tokens(
            request, num_computed_tokens, **kwargs)

    def update_state_after_alloc(self, request: "Request",
                                 blocks: "KVCacheBlocks",
                                 num_external_tokens: int):
        """
        Update KVConnector state after block allocation.
        """
        self._lmcache_engine.update_state_after_alloc(request,
                                                      num_external_tokens)

    def build_connector_meta(
            self, scheduler_output: SchedulerOutput) -> KVConnectorMetadata:
        """
        Build the connector metadata for this step.

        This function should NOT modify fields in the scheduler_output.
        Also, calling this function will reset the state of the connector.

        Args:
            scheduler_output (SchedulerOutput): the scheduler output object.
        """
        return self._lmcache_engine.build_connector_meta(scheduler_output)


class LMCacheConnectorV2(KVConnectorBase_V1):
    """
    Enhanced LMCache connector with layer-wise KV cache transfer support.
    
    This connector leverages LayerAwareLMCacheEngine to enable progressive
    layer-wise transfers where:
    1. Each layer contains ALL tokens for that layer
    2. Layers are transferred progressively: Layer-0 → Layer-1 → Layer-2 → etc.
    3. Decoder can start processing as soon as layer-0 is available
    
    """

    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole):
        super().__init__(vllm_config=vllm_config, role=role)
        self._lmcache_engine = LMCacheConnectorV1Impl(vllm_config, role, self)
        
        # Track layer-wise loading state
        self._layeraware_enabled = self._check_layeraware_support()
        
    def _check_layeraware_support(self) -> bool:
        """Check if LayerAwareLMCacheEngine is available."""
        try:
            from lmcache.experimental.cache_engine import LayerAwareLMCacheEngine
            return True
        except ImportError:
            return False

    # ==============================
    # Worker-side methods
    # ==============================
    def start_load_kv(self, forward_context: "ForwardContext",
                      **kwargs) -> None:
        """
        Start loading the KV cache from the connector to vLLM's paged
        KV buffer. Enhanced for layer-wise transfers.

        Args:
            forward_context (ForwardContext): the forward context.
            **kwargs: additional arguments for the load operation
        """
        self._lmcache_engine.start_load_kv(forward_context, **kwargs)

    def wait_for_layer_load(self, layer_name: str) -> None:
        """
        Block until the KV for a specific layer is loaded into vLLM's
        paged buffer. Enhanced for progressive layer-wise loading.
        
        Key enhancement: This method now supports waiting for individual
        layers to become available, enabling early processing while later
        layers are still being transferred from the remote prefiller.

        Args:
            layer_name: the name of that layer (e.g. "layers.0", "layers.1")
        """
        self._lmcache_engine.wait_for_layer_load(layer_name)

    def _parse_layer_id(self, layer_name: str) -> int:
        """Parse layer ID from layer name (e.g. 'layers.0' -> 0).
        
        expected layer name format: "model.layers.15.self_attn.attn"
        """

        try:
            # NOTE: Layer name format: "model.layers.number.self_attn.attn"
            import re
            numbers = re.findall(r'\d+', layer_name)
            return int(numbers[-1]) if numbers else 0
        except (ValueError, IndexError):
            logger.warning(f"Could not parse layer ID from {layer_name}, defaulting to 0")
            return 0

    def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor,
                      attn_metadata: "AttentionMetadata", **kwargs) -> None:
        """
        Start saving a layer of KV cache from vLLM's paged buffer 
        to the connector. Enhanced for progressive layer transmission.

        Args:
            layer_name (str): the name of the layer.
            kv_layer (torch.Tensor): the paged KV buffer of the current 
                layer in vLLM.
            attn_metadata (AttentionMetadata): the attention metadata.
            **kwargs: additional arguments for the save operation.
        """
        self._lmcache_engine.save_kv_layer(layer_name, kv_layer, attn_metadata, **kwargs)

    def wait_for_save(self):
        """
        Block until all the save operations are done. Enhanced for
        layer-wise save coordination.
        """
        self._lmcache_engine.wait_for_save()

    def publish_first_decode_logits(self, req_id: str, logits: torch.Tensor) -> None:
        """
        Publish the last prompt token logits for a request to enable
        zero-compute first decode on the decoder worker.
        """
        self._lmcache_engine.publish_first_decode_logits(req_id, logits)

    def try_consume_first_decode_logits(self, req_id: str) -> Optional[torch.Tensor]:
        """
        Try to retrieve the first decode logits for a request.
        """
        return self._lmcache_engine.try_consume_first_decode_logits(req_id)

    # ==============================
    # Scheduler-side methods
    # ==============================
    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
        **kwargs,
    ) -> tuple[int, bool]:
        """
        Get number of new tokens that can be loaded from the
        external KV cache beyond the num_computed_tokens.
        
        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally computed tokens

        Returns:
            Tuple of (num_tokens_available, should_load_async)
            - num_tokens_available: number of tokens that can be loaded
            - should_load_async: True if layer-wise loading should begin
        """
        return self._lmcache_engine.get_num_new_matched_tokens(
            request, num_computed_tokens, **kwargs)

    def has_first_decode_logits(self, request: "Request") -> bool:
        """
        Check if first decode logits are available for a request.
        """
        return self._lmcache_engine.has_first_decode_logits(request)
        
    def update_state_after_alloc(self, request: "Request",
                                 blocks: "KVCacheBlocks",
                                 num_external_tokens: int):
        """
        Update KVConnector state after block allocation.
        Enhanced for layer-wise transfer coordination.
        """
        if self._layeraware_enabled and num_external_tokens > 0:
            logger.debug(
                f"📋 Allocated {num_external_tokens} tokens for layer-wise loading "
                f"of request {request.request_id}"
            )
        
        self._lmcache_engine.update_state_after_alloc(request, num_external_tokens)

    def build_connector_meta(
            self, scheduler_output: SchedulerOutput) -> KVConnectorMetadata:
        """
        Build the connector metadata for this step.
        Enhanced to include layer-wise transfer information.

        Args:
            scheduler_output (SchedulerOutput): the scheduler output object.
        """
        meta = self._lmcache_engine.build_connector_meta(scheduler_output)
        
        # Enhanced metadata for layer-wise transfers
        if self._layeraware_enabled:
            # Add layer-wise specific metadata if needed
            # This could include layer readiness information, transfer priorities, etc.
            if hasattr(meta, '__dict__'):
                meta.layerwise_enabled = True
                meta.num_scheduled_layerwise_reqs = len([
                    req for req in scheduler_output.scheduled_new_reqs
                    if req.req_id in [r.req_id for r in getattr(meta, 'requests', [])]
                ])
        
        return meta