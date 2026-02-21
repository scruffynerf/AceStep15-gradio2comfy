"""
ACE-Step Custom Nodes for ComfyUI
Complete self-contained node package for ACE-Step 1.5 music generation
"""
import os
import importlib
import logging

logger = logging.getLogger(__name__)

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

def load_nodes():
    nodes_dir = os.path.join(os.path.dirname(__file__), "nodes")
    if not os.path.exists(nodes_dir):
        return

    for file in os.listdir(nodes_dir):
        if file.endswith("_node.py"):
            node_name = file[:-3]
            try:
                module = importlib.import_module(f".nodes.{node_name}", package=__name__)
                if hasattr(module, "NODE_CLASS_MAPPINGS"):
                    NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)
                if hasattr(module, "NODE_DISPLAY_NAME_MAPPINGS"):
                    NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)
            except Exception as e:
                logger.error(f"Failed to load node module {node_name}: {e}")

load_nodes()

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

print(f"🎵 ACE-Step Nodes loaded: {len(NODE_CLASS_MAPPINGS)} nodes registered")
