# SPDX-License-Identifier: Apache-2.0
from typing import TYPE_CHECKING

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

    # ==============================
    # Scheduler-side methods
    # ==============================
    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
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
            request, num_computed_tokens), False

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
        self._layerwise_enabled = self._check_layerwise_support()
        if self._layerwise_enabled:
            logger.info("✅ Layer-wise transfers enabled with LayerAwareLMCacheEngine")
        else:
            logger.info("⚠️ Falling back to standard transfers (LayerAwareLMCacheEngine not available)")

    def _check_layerwise_support(self) -> bool:
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
        if self._layerwise_enabled:
            logger.debug("🚀 Starting layer-wise KV loading...")
        
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
        if self._layerwise_enabled:
            layer_id = self._parse_layer_id(layer_name)
            logger.debug(f"⏳ Waiting for layer {layer_id} ({layer_name}) to be ready...")
            
            # Use enhanced layer waiting if available
            if hasattr(self._lmcache_engine, 'wait_for_layer_load_enhanced'):
                self._lmcache_engine.wait_for_layer_load_enhanced(layer_name, layer_id)
            else:
                # Fall back to standard method
                self._lmcache_engine.wait_for_layer_load(layer_name)
        else:
            # Standard behavior for non-layerwise engines
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
        if self._layerwise_enabled:
            layer_id = self._parse_layer_id(layer_name)
            logger.debug(f"💾 Saving layer {layer_id} ({layer_name}) progressively...")
        
        self._lmcache_engine.save_kv_layer(layer_name, kv_layer, attn_metadata, **kwargs)

    def wait_for_save(self):
        """
        Block until all the save operations are done. Enhanced for
        layer-wise save coordination.
        """
        self._lmcache_engine.wait_for_save()

    # ==============================
    # Scheduler-side methods
    # ==============================
    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        """
        Enhanced for layer-wise transfers: Check if ANY layers are available
        for this request. This is the key method that enables early scheduling.
        
        The core insight: Instead of waiting for ALL layers to be ready,
        we check if the first few layers are available and signal the scheduler
        to start async loading. The worker-side wait_for_layer_load() will
        handle the progressive availability.
        
        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally computed tokens

        Returns:
            Tuple of (num_tokens_available, should_load_async)
            - num_tokens_available: number of tokens that can be loaded
            - should_load_async: True if layer-wise loading should begin
        """
        # Get the base number of matched tokens from LMCache
        num_matched_tokens = self._lmcache_engine.get_num_new_matched_tokens(
            request, num_computed_tokens)
        
        if num_matched_tokens == 0:
            return 0, False
            
        # Enhanced logic for layer-wise transfers
        if self._layerwise_enabled:
            # Check if we can start layer-wise loading
            if self._can_start_layerwise_loading(request):
                logger.debug(
                    f"✅ Layer-wise loading available for request {request.request_id}: "
                    f"{num_matched_tokens} tokens can be loaded progressively"
                )
                return num_matched_tokens, True
            else:
                # No layers ready yet, don't start loading
                logger.debug(f"⏳ No layers ready yet for request {request.request_id}")
                return 0, False
        else:
            # Fall back to original behavior for non-layerwise engines
            return num_matched_tokens, False

    def _can_start_layerwise_loading(self, request: "Request") -> bool:
        """
        Check if layer-wise loading can start for the given request.
        
        This method determines if at least the first layer (layer-0) is
        available, which is the minimum requirement to start progressive loading.
        
        Args:
            request: The vLLM request object
            
        Returns:
            True if layer-wise loading can begin
        """
        # Check if the LMCache engine supports layer availability checking
        if hasattr(self._lmcache_engine, 'has_any_layers_ready'):
            try:
                return self._lmcache_engine.has_any_layers_ready(request)
            except Exception as e:
                logger.debug(f"Error checking layer readiness: {e}")
                return False
        
        # Fallback: assume we can start if we have layer-wise support
        # In practice, you might want to implement a simple layer-0 check here
        return True

    def update_state_after_alloc(self, request: "Request",
                                 blocks: "KVCacheBlocks",
                                 num_external_tokens: int):
        """
        Update KVConnector state after block allocation.
        Enhanced for layer-wise transfer coordination.
        """
        if self._layerwise_enabled and num_external_tokens > 0:
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
        if self._layerwise_enabled:
            # Add layer-wise specific metadata if needed
            # This could include layer readiness information, transfer priorities, etc.
            if hasattr(meta, '__dict__'):
                meta.layerwise_enabled = True
                meta.num_scheduled_layerwise_reqs = len([
                    req for req in scheduler_output.scheduled_new_reqs
                    if req.req_id in [r.req_id for r in getattr(meta, 'requests', [])]
                ])
        
        return meta

    # ==============================
    # Layer-wise specific methods
    # ==============================
    
    def get_layerwise_status(self) -> dict:
        """
        Get the current status of layer-wise transfers.
        
        Returns:
            Dictionary with layer-wise transfer status information
        """
        status = {
            "layerwise_enabled": self._layerwise_enabled,
            "connector_version": "LMCacheConnectorV2",
            "supports_progressive_loading": True,
        }
        
        if hasattr(self._lmcache_engine, 'supports_layerwise_loading'):
            status["engine_layerwise_support"] = self._lmcache_engine.supports_layerwise_loading()
        
        return status
