"""Inference clients for local and remote models."""

from my_routing_agent.clients.local_client import InferenceResponse, LocalClient
from my_routing_agent.clients.remote_client import RemoteClient

__all__ = ["InferenceResponse", "LocalClient", "RemoteClient"]
