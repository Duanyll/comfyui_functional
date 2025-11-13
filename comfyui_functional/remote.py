from .utils import (
    serialize,
    deserialize,
    AnyType,
    ContainsDynamicDict,
    Closure,
    create_remote_workflow_from_closure,
)
from .comfy_client import ComfyClient

last_serialized_data = None
"""
Buffer to hold the most recently serialized data, preventing it from being released.
Ensures that shared memory tensors remain accessible during remote execution, until
next call to Serialize node.
"""

class Serialize:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "data": (AnyType("*"),),
                "use_shared_memory": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "run"
    CATEGORY = "duanyll/functional/internal"
    OUTPUT_NODE = True
    
    @classmethod
    def IS_CHANGED(cls, data):
        return float("NaN")

    def run(self, data, use_shared_memory):
        global last_serialized_data
        try:
            text, last_serialized_data = serialize(data, use_shared_memory=use_shared_memory)
            # print(f"Serialized {type(data)} to string of length {len(text)}")
            return {
                "ui": {"text": [text]},
                "result": (text,),
            }
        except Exception as e:
            raise ValueError(
                f"Cannot serialize function output. It may contain unserializable types. Error: {e}"
            )


class Deserialize:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "data": ("STRING", {"multiline": True}),
            }
        }

    RETURN_TYPES = (AnyType("*"),)
    FUNCTION = "run"
    CATEGORY = "duanyll/functional/internal"

    def run(self, data):
        return (deserialize(data),)


class CallClosureRemote:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "closure": ("CLOSURE",),
                "base_url": ("STRING",),
                "timeout": ("FLOAT", {"default": 600.0, "min": 1.0}),
                "use_shared_memory": ("BOOLEAN", {"default": False}),
            },
            "optional": ContainsDynamicDict(
                {
                    "param_0": (AnyType("*"), {"_dynamic": "number"}),
                }
            )
        }
        
    RETURN_TYPES = (AnyType("*"),)
    FUNCTION = "run"
    CATEGORY = "duanyll/functional"
    
    async def run(self, closure, base_url, timeout, use_shared_memory, **kwargs):
        params = []
        for i in range(len(kwargs)):
            params.append(kwargs[f"param_{i}"])
            
        client = ComfyClient(server_base=base_url, timeout=timeout)
        workflow = create_remote_workflow_from_closure(closure, params, use_shared_memory=use_shared_memory)
        client_response = await client.run_workflow(workflow)
        output = client_response.get("__output__", None)
        if output is None:
            raise ValueError("No output received from remote workflow execution.")
        return (deserialize(output[0]),)
    
    
NODE_CLASS_MAPPINGS = {
    "__Serialize__": Serialize,
    "__Deserialize__": Deserialize,
    "CallClosureRemote": CallClosureRemote
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "__Serialize__": "[Internal] Serialize",
    "__Deserialize__": "[Internal] Deserialize",
    "CallClosureRemote": "Call Remote Function",
}