from . import core, high_level, logic, side_effects, remote

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

for module in [core, high_level, logic, side_effects, remote]:
    NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)
    
# Mangle ids to avoid conflicts
PREFIX = "duanyll::"
NODE_CLASS_MAPPINGS = {
    (k if k.startswith("__") else PREFIX + k): v
    for k, v in NODE_CLASS_MAPPINGS.items()
}
NODE_DISPLAY_NAME_MAPPINGS = {
    (k if k.startswith("__") else PREFIX + k): v
    for k, v in NODE_DISPLAY_NAME_MAPPINGS.items()
}