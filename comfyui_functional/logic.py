from .utils import AnyType

class LogicalAnd:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input1": ("BOOLEAN", ),
                "input2": ("BOOLEAN", {"lazy": True}),
            }
        }

    RETURN_TYPES = ("BOOLEAN",)
    FUNCTION = "logical_and"
    CATEGORY = "duanyll/functional/logic"
    DESCRIPTION = "Logical AND operation with lazy evaluation. If input1 is False, input2 is not evaluated."
    
    def check_lazy_status(self, input1, input2):
        if input1 is True:
            return ["input2"]
        else:
            return []

    def logical_and(self, input1, input2):
        if input1 is True:
            return (input2 is True, )
        else:
            return (False, )
    
    
class LogicalOr:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input1": ("BOOLEAN",),
                "input2": ("BOOLEAN", {"lazy": True}),
            }
        }
        
    RETURN_TYPES = ("BOOLEAN",)
    FUNCTION = "logical_or"
    CATEGORY = "duanyll/functional/logic"
    DESCRIPTION = "Logical OR operation with lazy evaluation. If input1 is True, input2 is not evaluated."

    def check_lazy_status(self, input1, input2):
        if input1 is False:
            return ["input2"]
        else:
            return []

    def logical_or(self, input1, input2):
        if input1 is False:
            return (input2 is True,)
        else:
            return (True,)
        

class IfCondition:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "condition": ("BOOLEAN",),
                "true_value": (AnyType("*"), {"lazy": True}),
                "false_value": (AnyType("*"), {"lazy": True}),
            }
        }

    RETURN_TYPES = (AnyType("*"),)
    FUNCTION = "if_condition"
    CATEGORY = "duanyll/functional/logic"
    DESCRIPTION = "Returns true_value if condition is True, else returns false_value. Supports lazy evaluation."

    def check_lazy_status(self, condition, true_value, false_value):
        if condition is True:
            return ["true_value"]
        else:
            return ["false_value"]
        
    def if_condition(self, condition, true_value, false_value):
        if condition is True:
            return (true_value, )
        else:
            return (false_value, )
        
        
NODE_CLASS_MAPPINGS = {
    "LogicalAnd": LogicalAnd,
    "LogicalOr": LogicalOr,
    "IfCondition": IfCondition,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LogicalAnd": "Logical AND",
    "LogicalOr": "Logical OR",
    "IfCondition": "If Condition",
}