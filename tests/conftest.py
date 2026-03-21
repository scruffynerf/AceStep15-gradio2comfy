import sys
import types
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock ComfyUI's folder_paths so that node files can be imported
mock_folder_paths = types.ModuleType("folder_paths")
mock_folder_paths.models_dir = "/tmp/models"
mock_folder_paths.get_filename_list = lambda x: []
mock_folder_paths.get_full_path = lambda x, y: "/tmp/models/" + y
sys.modules["folder_paths"] = mock_folder_paths
