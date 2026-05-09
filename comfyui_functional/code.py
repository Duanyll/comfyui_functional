import os

from .utils import AnyType, ContainsDynamicDict


class PythonExec:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "code": ("STRING", {"multiline": True}),
            },
            "optional": ContainsDynamicDict(
                {
                    "param_0": (AnyType("*"), {"_dynamic": "number"}),
                }
            ),
        }

    RETURN_TYPES = (AnyType("*"),)
    FUNCTION = "run"
    CATEGORY = "duanyll/functional"

    def run(self, code, **kwargs):
        if os.getenv("COMFYUI_FUNCTIONAL_DANGER_MODE") != "1":
            raise RuntimeError(
                "Execution of arbitrary code is disabled by default for security reasons. "
                "To enable it, carefully review the code you are executing and set the "
                "environment variable COMFYUI_FUNCTIONAL_DANGER_MODE to 1."
            )

        # kwargs: param_0, param_1, ...
        params = []
        for i in range(len(kwargs)):
            params.append(kwargs[f"param_{i}"])
        local_vars = {"params": params}
        exec(code, {}, local_vars)  # noqa
        return (local_vars.get("result", None),)


NODE_CLASS_MAPPINGS = {
    "PythonExec": PythonExec,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PythonExec": "Python Exec",
}
