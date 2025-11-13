from typing import TypedDict
import copy
import json
import io
import base64
import numpy as np
import torch
from PIL import Image

class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False


class ContainsDynamicDict(dict):
    """
    A custom dictionary that dynamically returns values for keys based on a pattern.
    - If a key in the passed dictionary has a value with `{"_dynamic": "number"}` in the tuple's second position,
      then any other key starting with the same string and ending with a number will return that value.
    - For other keys, normal dictionary lookup behavior applies.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Store prefixes associated with `_dynamic` values for efficient lookup
        self._dynamic_prefixes = {
            key.rstrip("0123456789"): value
            for key, value in self.items()
            if isinstance(value, tuple)
            and len(value) > 1
            and value[1].get("_dynamic") == "number"
        }

    def __contains__(self, key):
        # Check if key matches a dynamically handled prefix or exists normally
        return any(
            key.startswith(prefix) and key[len(prefix) :].isdigit()
            for prefix in self._dynamic_prefixes
        ) or super().__contains__(key)

    def __getitem__(self, key):
        # Dynamically return the value for keys matching a `prefix<number>` pattern
        for prefix, value in self._dynamic_prefixes.items():
            if key.startswith(prefix) and key[len(prefix) :].isdigit():
                return value
        # Fallback to normal dictionary behavior for other keys
        return super().__getitem__(key)


class Closure(TypedDict):
    body: dict
    captures: list
    output: list

def create_graph_from_closure(closure: Closure, params, caller_unique_id=None):
    body = copy.deepcopy(closure["body"])
    graph = {}
    
    def transform_input(spec):
        if not isinstance(spec, list):
            return spec
        # Return normal spec if not a capture or param
        if spec[0] != "__capture" and spec[0] != "__param":
            return spec
        if spec[0] == "__param":
            if spec[1] >= len(params):
                raise IndexError(
                    f"Parameter index {spec[1]} out of range for function call. "
                    f"Only {len(params)} parameters were provided."
                )
            value = params[spec[1]]
        else:
            value = closure["captures"][spec[1]]
        if isinstance(value, list):
            prefix = f"{caller_unique_id}_" if caller_unique_id is not None else ""
            recover_id = f"{prefix}{spec[0]}_{spec[1]}"
            if recover_id not in graph:
                graph[recover_id] = {
                    "inputs": {
                        # Wrap the list in a tuple so it's treated as a literal, not a link
                        "values": (value,),
                    },
                    "class_type": "__RecoverList__",
                }
            return [recover_id, 0]
        else:
            return value
    
    for node_id, node_data in body.items():
        inputs = node_data["inputs"]
        for key in inputs.keys():
            inputs[key] = transform_input(inputs[key])
        node_data["override_display_id"] = node_id
        graph[f"{caller_unique_id}_{node_id}"] = node_data
    
    output = transform_input(closure["output"])
    return graph, output
    
    
def create_remote_workflow_from_closure(closure, params):
    body = copy.deepcopy(closure["body"])
    graph = {}

    def transform_input(spec):
        if not isinstance(spec, list):
            return spec
        # Return normal spec if not a capture or param
        if spec[0] != "__capture" and spec[0] != "__param":
            return spec
        
        # Check if we have already created a deserialize node for this spec
        deserialize_node_id = f"deserialize_{spec[0]}_{spec[1]}"
        if deserialize_node_id in graph:
            return [deserialize_node_id, 0]
        
        # Try to create a deserialize node
        if spec[0] == "__param":
            if spec[1] >= len(params):
                raise IndexError(
                    f"Parameter index {spec[1]} out of range for function call. "
                    f"Only {len(params)} parameters were provided."
                )
            value = params[spec[1]]
        else:
            value = closure["captures"][spec[1]]
        
        try:
            serialized_value = serialize(value)
            graph[deserialize_node_id] = {
                "inputs": {
                    "data": serialized_value,
                },
                "class_type": "__Deserialize__",
            }
            return [deserialize_node_id, 0]
        except TypeError as e:
            if spec[0] == "__capture":
                raise ValueError(
                    f"Cannot serialize captured value at index {spec[1]}. "
                    "Captured values must be serializable to be used in remote execution. "
                    "Consider disabling capture for this function to initialize the value remotely. "
                    f"Error: {e}"
                )
            else:
                raise ValueError(
                    f"Cannot serialize parameter at index {spec[1]}. "
                    "Parameters must be serializable to be used in remote execution. "
                    f"Error: {e}"
                )
            
    for node_id, node_data in body.items():
        inputs = node_data["inputs"]
        for key in inputs.keys():
            inputs[key] = transform_input(inputs[key])
        graph[node_id] = node_data
        
    output = transform_input(closure["output"])
    graph["__output__"] = {
        "inputs": {
            "data": output,
        },
        "class_type": "__Serialize__",
        "_meta": {
            "title": "%__output__%"
        }
    }
    
    return graph

class WhitelistEncoder(json.JSONEncoder):
    def default(self, obj):
        # 区分 set 和 tuple
        if isinstance(obj, set):
            return {'__type__': 'set', 'data': list(obj)}
        if isinstance(obj, tuple):
            return {'__type__': 'tuple', 'data': list(obj)}
        
        if isinstance(obj, bytes):
            return {'__type__': 'bytes', 'data': base64.b64encode(obj).decode('utf-8')}

        if isinstance(obj, np.ndarray):
            buffer = io.BytesIO()
            np.save(buffer, obj, allow_pickle=False) 
            return {
                '__type__': 'numpy.ndarray',
                'data': base64.b64encode(buffer.getvalue()).decode('utf-8')
            }

        if isinstance(obj, torch.Tensor):
            buffer = io.BytesIO()
            torch.save(obj.cpu(), buffer)
            return {
                '__type__': 'torch.Tensor',
                'data': base64.b64encode(buffer.getvalue()).decode('utf-8')
            }

        if isinstance(obj, Image.Image):
            buffer = io.BytesIO()
            obj.save(buffer, format='PNG')
            return {
                '__type__': 'PIL.Image',
                'data': base64.b64encode(buffer.getvalue()).decode('utf-8')
            }

        raise TypeError(
            f"Object of type '{type(obj).__name__}' is not serializable. "
            "Only basic types, set, tuple, bytes, numpy.ndarray, "
            "torch.Tensor, and PIL.Image are allowed."
        )

# 2. 自定义解码器 (Deserialization)
def whitelist_decoder(dct):
    """
    用于 json.load/loads 的 object_hook，用于恢复自定义类型。
    """
    if '__type__' in dct:
        obj_type = dct['__type__']
        data = dct['data']

        if obj_type == 'set':
            return set(data)
        if obj_type == 'tuple':
            return tuple(data)
        if obj_type == 'bytes':
            return base64.b64decode(data)

        if obj_type == 'numpy.ndarray':
            buffer = io.BytesIO(base64.b64decode(data))
            # allow_pickle=True 是为了恢复可能包含的复杂dtype
            return np.load(buffer, allow_pickle=True) 

        if obj_type == 'torch.Tensor':
            buffer = io.BytesIO(base64.b64decode(data))
            return torch.load(buffer)

        if obj_type == 'PIL.Image':
            buffer = io.BytesIO(base64.b64decode(data))
            img = Image.open(buffer)
            # .copy() 是个好习惯，确保数据已从缓冲区完全加载
            return img.copy() 
            
    return dct


def serialize(obj):
    """
    将对象序列化为 JSON 字符串。
    仅支持基本类型、set、tuple、bytes、numpy.ndarray、torch.Tensor 和 PIL.Image。
    """
    return json.dumps(obj, cls=WhitelistEncoder)


def deserialize(json_str):
    """
    从 JSON 字符串反序列化对象。
    """
    return json.loads(json_str, object_hook=whitelist_decoder)