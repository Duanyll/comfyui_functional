# ComfyUI Functional 实现原理

本文档面向开发者，详细介绍 ComfyUI Functional 节点包的内部实现机制。如需了解如何使用这些节点，请参阅 [usage.md](./usage.md) 和 [node-reference.md](./node-reference.md)。

## 目录

1. [整体架构](#整体架构)
2. [执行钩子与工作流预处理](#执行钩子与工作流预处理)
3. [函数定义与闭包转换](#函数定义与闭包转换)
4. [函数调用与图展开](#函数调用与图展开)
5. [协程机制与高阶函数](#协程机制与高阶函数)
6. [副作用处理](#副作用处理)
7. [远程函数调用](#远程函数调用)
8. [类型系统与工具函数](#类型系统与工具函数)
9. [技术细节与限制](#技术细节与限制)

---

## 整体架构

ComfyUI Functional 通过以下几个核心机制实现函数式编程：

```
┌─────────────────────────────────────────────────────────────────┐
│                     ComfyUI 执行流程                              │
├─────────────────────────────────────────────────────────────────┤
│  1. 用户提交工作流 (Prompt)                                       │
│           ↓                                                      │
│  2. hook.py 拦截执行请求                                          │
│           ↓                                                      │
│  3. transform.py 预处理工作流                                     │
│      - 转换 FunctionParam/FunctionEnd → CreateClosure            │
│      - 预处理 Inspect 节点                                        │
│           ↓                                                      │
│  4. ComfyUI 执行转换后的工作流                                     │
│      - CallClosure 通过 expand 机制动态展开函数体                   │
│      - CoroutineNodeBase 通过 IntermidiateCoroutine 实现迭代      │
│           ↓                                                      │
│  5. 返回执行结果                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 核心文件说明

| 文件 | 职责 |
|------|------|
| `hook.py` | 钩住 ComfyUI 的 `PromptExecutor.execute` 方法，在执行前预处理工作流 |
| `transform.py` | 工作流图的静态变换算法，将函数定义转换为闭包节点 |
| `core.py` | 核心节点实现：`FunctionParam`、`FunctionEnd`、`CreateClosure`、`CallClosure`、`CoroutineNodeBase` |
| `utils.py` | 工具函数：`AnyType` 通配类型、`Closure` 数据结构、图创建与序列化 |
| `high_level.py` | 高阶函数节点：`Map`、`Fold`、`Nest` 等 |
| `side_effects.py` | 副作用节点：`Sow`、`Reap`、`Inspect`、`Sleep` |
| `logic.py` | 逻辑控制节点：`IfCondition`、`LogicalAnd`、`LogicalOr` |
| `remote.py` | 远程函数调用节点：`CallClosureRemote`、`Serialize`、`Deserialize` |
| `comfy_client.py` | ComfyUI HTTP/WebSocket 客户端，用于提交远程工作流 |

---

## 执行钩子与工作流预处理

### 钩子注入 (`hook.py`)

ComfyUI Functional 在模块加载时（`__init__.py`）自动钩住 ComfyUI 的执行器：

```python
# hook.py
def hook_comfyui_execution():
    from execution import PromptExecutor

    original_execute = PromptExecutor.execute

    def hooked_execute(self, prompt, prompt_id, extra_data={}, execute_outputs=[]):
        reset_reap_storage()  # 重置 Sow/Reap 的全局存储
        prompt, warnings = transform_workflow(prompt, execute_outputs)
        return original_execute(self, prompt, prompt_id, extra_data, execute_outputs)

    PromptExecutor.execute = hooked_execute
```

这个钩子确保每次执行工作流前：
1. 清空 `Sow`/`Reap` 的全局存储
2. 对工作流进行静态变换，将函数定义转换为闭包

---

## 函数定义与闭包转换

### 用户视角的函数定义

用户通过以下节点组合定义函数：
- **`FunctionParam`**：声明函数参数，通过 `index` 指定参数顺序
- **`FunctionEnd`**：标记函数返回值，输出一个 `CLOSURE` 类型

### 静态变换算法 (`transform.py`)

`transform_workflow` 函数将用户定义的函数子图转换为 `CreateClosure` 节点。算法流程如下：

#### 步骤 1：构建图结构

```python
class WorkflowGraph:
    def __init__(self, workflow: Dict):
        self.adj: Dict[str, List[str]]      # 邻接表：节点 → 后继节点列表
        self.rev_adj: Dict[str, List[str]]  # 反向邻接表：节点 → 前驱节点列表
        self.in_degree: Dict[str, int]      # 入度
        self.out_degree: Dict[str, int]     # 出度
```

#### 步骤 2：预处理 Inspect 节点

`_preprocess_inspect_nodes` 将 `__Inspect__` 节点后直接连接的输出节点"吸收"到 Inspect 的 `body` 中，转换为 `__InspectImpl__` 节点。这样可以避免函数体内的输出节点被过早执行。

#### 步骤 3：拓扑排序并处理函数定义

按拓扑顺序遍历所有 `__FunctionEnd__` 节点，对每个函数执行 `_transform_single_function`：

```python
def _transform_single_function(end_node_id: str, workflow: Dict, graph: WorkflowGraph):
    # 1. 从 FunctionEnd 反向遍历，找到完整的函数子图
    subgraph_nodes = _find_function_subgraph(end_node_id, graph)

    # 2. 从 FunctionParam 正向遍历，找到依赖参数的节点（函数体）
    param_nodes = _find_param_nodes(subgraph_nodes, workflow, graph)

    # 3. 计算捕获节点 = 子图 - 参数依赖节点
    capture_nodes = subgraph_nodes - param_nodes
```

#### 函数子图的划分

```
┌─────────────────────────────────────────────────────────────────┐
│                        函数子图                                  │
├──────────────────────┬──────────────────────────────────────────┤
│    捕获节点           │           参数依赖节点（函数体）            │
│   (capture_nodes)    │           (param_nodes)                  │
├──────────────────────┼──────────────────────────────────────────┤
│ - 不依赖 FunctionParam│ - 直接或间接依赖 FunctionParam            │
│ - 在函数定义时求值     │ - 每次调用时重新求值                       │
│ - 作为闭包的捕获变量   │ - 序列化到闭包的 body 中                   │
└──────────────────────┴──────────────────────────────────────────┘
```

#### 步骤 4：创建 CreateClosure 节点

```python
closure_node = {
    "class_type": "__CreateClosure__",
    "inputs": {
        "body": json.dumps(param_subgraph_copy),  # 函数体的节点定义
        "output": json.dumps(output_edge),         # 输出边的引用
        "side_effects": float("NaN") if has_side_effects else 0.0,
        "capture_0": [source_id, source_idx],     # 捕获的边
        "capture_1": [...],
        ...
    }
}
```

函数体中的边会被重写：
- 来自 `FunctionParam` 的边 → `["__param", param_index]`
- 来自捕获节点的边 → `["__capture", capture_index]`

#### 示例：函数转换过程

原始工作流：
```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ LoadCheckpt │───→│  KSampler   │───→│ FunctionEnd │
│     (1)     │    │     (4)     │    │     (5)     │
└─────────────┘    └─────────────┘    └─────────────┘
                         ↑
                   ┌─────────────┐
                   │ FunctionParam│
                   │     (3)     │
                   └─────────────┘
```

转换后：
```
┌─────────────┐    ┌─────────────────────────────────────────────┐
│ LoadCheckpt │───→│            CreateClosure (5)                 │
│     (1)     │    │  body: {KSampler 的序列化定义}                │
│  (capture)  │    │  capture_0: [1, 0]                          │
└─────────────┘    │  output: ["__param", 0] 或 节点引用           │
                   └─────────────────────────────────────────────┘
```

---

## 函数调用与图展开

### Closure 数据结构 (`utils.py`)

```python
class Closure(TypedDict):
    body: dict      # 函数体节点的序列化定义
    captures: list  # 捕获变量的值列表
    output: list    # 输出边的规格 [node_id, slot] 或 ["__param", idx]
```

### CallClosure 节点 (`core.py`)

`CallClosure` 是函数调用的入口点。它使用 ComfyUI 的 `expand` 机制动态展开函数体：

```python
class CallClosure:
    def run(self, closure, unique_id, **kwargs):
        # 1. 检查递归深度（unique_id 长度表示嵌套层数）
        if len(unique_id) > RECURSION_LIMIT:
            raise RecursionError("...")

        # 2. 收集参数
        params = [kwargs[f"param_{i}"] for i in range(len(kwargs))]

        # 3. 从闭包创建执行图
        graph, output = create_graph_from_closure(closure, params, caller_unique_id=unique_id)

        # 4. 如果图为空（纯参数/捕获引用），直接返回值
        if len(graph) == 0:
            return (output,)

        # 5. 返回展开的图
        return {"result": (output,), "expand": graph}
```

### 图创建算法 (`utils.py: create_graph_from_closure`)

```python
def create_graph_from_closure(closure: Closure, params, caller_unique_id=None):
    body = copy.deepcopy(closure["body"])
    graph = {}

    def transform_input(spec):
        if not isinstance(spec, list):
            return spec

        prefix = f"{caller_unique_id}_" if caller_unique_id else ""

        # 普通节点引用：添加前缀
        if spec[0] != "__capture" and spec[0] != "__param":
            return [f"{prefix}{spec[0]}", spec[1]]

        # 参数引用：直接替换为参数值
        if spec[0] == "__param":
            value = params[spec[1]]
        # 捕获引用：直接替换为捕获值
        else:
            value = closure["captures"][spec[1]]

        # 如果值是列表，需要包装成 RecoverList 节点
        if isinstance(value, list):
            recover_id = f"{prefix}{spec[0]}_{spec[1]}"
            graph[recover_id] = {
                "inputs": {"values": (value,)},  # 元组包装防止被解析为边
                "class_type": "__RecoverList__",
            }
            return [recover_id, 0]
        else:
            return value

    # 转换所有节点的输入
    for node_id, node_data in body.items():
        for key in node_data["inputs"].keys():
            node_data["inputs"][key] = transform_input(node_data["inputs"][key])
        node_data["override_display_id"] = node_id  # 保持原始显示 ID
        graph[f"{caller_unique_id}_{node_id}"] = node_data

    output = transform_input(closure["output"])
    return graph, output
```

### unique_id 的作用

ComfyUI 的每个节点有唯一的 `unique_id`。在函数调用时：
- 展开的节点 ID 格式：`{caller_unique_id}_{original_node_id}`
- 嵌套调用时：`{outer_id}_{inner_id}_{node_id}`
- 通过 `unique_id` 长度检测递归深度

---

## 协程机制与高阶函数

### 设计动机

高阶函数（如 `Map`、`Fold`）需要多次调用用户提供的函数。由于 ComfyUI 的执行模型是基于图的单次遍历，无法直接实现循环。

解决方案：**Python 协程 + 图展开**

### CoroutineNodeBase (`core.py`)

所有高阶函数继承自 `CoroutineNodeBase`：

```python
class CoroutineNodeBase:
    def run_coroutine(self, **kwargs):
        """子类实现：生成器函数，通过 yield 调用闭包"""
        raise NotImplementedError()
        yield  # 使其成为生成器

    def run(self, unique_id, **kwargs):
        coroutine = self.run_coroutine(**kwargs)
        return {
            "result": ([f"{unique_id}_0", 0],),
            "expand": {
                f"{unique_id}_0": {
                    "inputs": {
                        "return_value": None,
                        "coroutine": coroutine,
                    },
                    "class_type": "__IntermidiateCoroutine__",
                }
            },
        }
```

### IntermidiateCoroutine (`core.py`)

协程的中间状态管理器，负责驱动生成器执行：

```python
class IntermidiateCoroutine:
    def run(self, return_value, coroutine, unique_id):
        (base_id, index) = unique_id.rsplit("_", 1)
        graph = {}

        while len(graph) == 0:
            try:
                # 将上一次调用的结果发送给协程
                (closure, params) = coroutine.send(return_value)
            except StopIteration as e:
                # 协程结束，返回最终结果
                return (e.value,)

            index = int(index) + 1
            next_node_id = f"{base_id}_{index}"

            if index > COROUTINE_LIMIT:
                raise RecursionError("Coroutine execution limit exceeded")

            # 展开闭包调用
            graph, output = create_graph_from_closure(closure, params, caller_unique_id=unique_id)

            if len(graph) == 0:
                # 如果不需要展开图，直接使用结果继续循环
                return_value = output

        # 返回展开的图和下一个中间协程节点
        return {
            "result": ([next_node_id, 0],),
            "expand": {
                **graph,
                next_node_id: {
                    "inputs": {
                        "return_value": output,
                        "coroutine": coroutine,
                    },
                    "class_type": "__IntermidiateCoroutine__",
                },
            },
        }
```

### 执行流程示例：Map

```python
class Map(CoroutineNodeBase):
    def run_coroutine(self, function, items):
        results = []
        for item in items:
            val = yield (function, [item])  # 暂停，请求调用 function(item)
            results.append(val)
        return results
```

执行 `Map(f, [a, b, c])` 的流程：

```
1. Map.run() 创建协程，展开 IntermidiateCoroutine_0

2. IC_0.run(None, coroutine):
   - coroutine.send(None) → yield (f, [a])
   - 展开 f(a) 的图 + IC_1

3. IC_1.run(result_a, coroutine):
   - coroutine.send(result_a) → yield (f, [b])
   - 展开 f(b) 的图 + IC_2

4. IC_2.run(result_b, coroutine):
   - coroutine.send(result_b) → yield (f, [c])
   - 展开 f(c) 的图 + IC_3

5. IC_3.run(result_c, coroutine):
   - coroutine.send(result_c) → StopIteration([result_a, result_b, result_c])
   - 返回最终结果
```

---

## 副作用处理

### Sow/Reap 机制 (`side_effects.py`)

全局存储用于跨节点收集值：

```python
reap_storage = {}  # 全局存储：{tag: [values]}

class Sow:
    def run(self, signal, value, tag):
        if tag not in reap_storage:
            reap_storage[tag] = []
        reap_storage[tag].append(value)
        return (signal,)  # 传递信号以保证执行顺序

class Reap:
    def run(self, signal, tag):
        values = reap_storage.get(tag, [])
        reap_storage[tag] = []  # 清空
        return (values,)
```

### 禁用缓存

副作用节点通过 `IS_CHANGED` 返回 `float("NaN")` 来禁用 ComfyUI 的缓存：

```python
@classmethod
def IS_CHANGED(cls, signal, value, tag):
    return float("NaN")  # NaN != NaN，总是被视为已更改
```

### Inspect 节点

`Inspect` 节点用于在函数体内查看中间值，同时不破坏惰性求值：

```python
class Inspect:
    def run(self, signal, value):
        return (
            signal,  # 第一个输出：传递信号
            ExecutionBlocker("..."),  # 第二个输出：阻止直接使用
        )
```

`transform.py` 中的 `_preprocess_inspect_nodes` 会将连接到 `Inspect` 第二个输出的叶子节点吸收到 `InspectImpl` 的 `body` 中，在运行时动态展开。

---

## 远程函数调用

### 设计目标

远程函数调用允许在其他 ComfyUI 实例上执行函数体，实现：
- 流水线并行：将工作流的不同阶段分布到不同 GPU/机器
- 避免模型反复装卸：在专用 worker 上保持大模型常驻

### CallClosureRemote (`remote.py`)

```python
class CallClosureRemote:
    async def run(self, closure, base_url, timeout, use_shared_memory, **kwargs):
        params = [kwargs[f"param_{i}"] for i in range(len(kwargs))]

        client = ComfyClient(server_base=base_url, timeout=timeout)
        workflow = create_remote_workflow_from_closure(closure, params, use_shared_memory)
        client_response = await client.run_workflow(workflow)

        output = client_response.get("__output__", None)
        return (deserialize(output[0]),)
```

与本地 `CallClosure` 的关键区别：
1. 不使用 `expand` 机制，而是通过 HTTP/WebSocket 提交完整工作流
2. 参数和返回值需要序列化/反序列化
3. 使用 `async` 异步执行

### 远程工作流生成 (`utils.py: create_remote_workflow_from_closure`)

```python
def create_remote_workflow_from_closure(closure, params, use_shared_memory=False):
    body = copy.deepcopy(closure["body"])
    graph = {}

    def transform_input(spec):
        if spec[0] == "__param" or spec[0] == "__capture":
            # 创建 Deserialize 节点来恢复序列化的值
            deserialize_node_id = f"deserialize_{spec[0]}_{spec[1]}"
            value = params[spec[1]] if spec[0] == "__param" else closure["captures"][spec[1]]
            serialized_value, _ = serialize(value, use_shared_memory)
            graph[deserialize_node_id] = {
                "inputs": {"data": serialized_value},
                "class_type": "__Deserialize__",
            }
            return [deserialize_node_id, 0]
        return spec

    # 转换所有输入引用
    for node_id, node_data in body.items():
        for key in node_data["inputs"]:
            node_data["inputs"][key] = transform_input(node_data["inputs"][key])
        graph[node_id] = node_data

    # 添加输出序列化节点
    output = transform_input(closure["output"])
    graph["__output__"] = {
        "inputs": {"data": output, "use_shared_memory": use_shared_memory},
        "class_type": "__Serialize__",
        "_meta": {"title": "%__output__%"},
    }

    return graph
```

### 捕获模式与远程调用

| 方面 | 本地调用 (`capture=True`) | 远程调用 (`capture=False`) |
|------|--------------------------|---------------------------|
| 模型加载 | 在函数外加载，被捕获复用 | 必须在函数体内加载 |
| 节点 ID | 每次调用重新分配前缀 | 保持原始 ID |
| 缓存 | 无效（ID 变化） | 有效（ID 不变） |
| 序列化 | 不需要 | 参数/返回值必须可序列化 |

禁用捕获时，函数体的所有节点（包括模型加载器）都会被序列化到 `body` 中，在远程执行时直接运行。由于节点 ID 不变，远程 ComfyUI 可以正常缓存已加载的模型。

### 序列化与共享内存

参见 [类型系统与工具函数](#类型系统与工具函数) 中的序列化支持部分。

当 `use_shared_memory=True` 且两个 ComfyUI 实例在同一机器上时，PyTorch 张量通过共享内存传递：

```python
if self.use_shared_memory:
    obj.share_memory_()
    storage = obj.untyped_storage()
    manager_handle, handle, size = storage._share_filename_cpu_()
    return {
        "__type__": "torch.Tensor_shared",
        "manager_handle": manager_handle,
        "handle": handle,
        # ... 其他元数据
    }
```

接收端通过 `torch.UntypedStorage._new_shared_filename_cpu()` 重建张量，避免数据复制。

---

## 类型系统与工具函数

### AnyType (`utils.py`)

ComfyUI 的类型系统要求输入输出类型匹配。`AnyType` 通过重写 `__ne__` 实现通配：

```python
class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False  # 与任何类型比较都返回"相等"
```

### ContainsDynamicDict (`utils.py`)

支持动态数量的输入参数（如 `param_0`, `param_1`, ...）：

```python
class ContainsDynamicDict(dict):
    """
    如果键以数字结尾且前缀匹配 _dynamic 模式，
    则动态返回对应的类型定义。
    """
    def __contains__(self, key):
        return any(
            key.startswith(prefix) and key[len(prefix):].isdigit()
            for prefix in self._dynamic_prefixes
        ) or super().__contains__(key)
```

### 序列化支持 (`utils.py`)

用于远程执行的序列化/反序列化：

```python
class WhitelistEncoder(json.JSONEncoder):
    """支持序列化：set, tuple, bytes, numpy.ndarray, torch.Tensor, PIL.Image"""

    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            if self.use_shared_memory:
                # 共享内存优化：只传递元数据
                obj.share_memory_()
                storage = obj.untyped_storage()
                manager_handle, handle, size = storage._share_filename_cpu_()
                return {"__type__": "torch.Tensor_shared", ...}
            else:
                # 普通序列化
                buffer = io.BytesIO()
                torch.save(obj, buffer)
                return {"__type__": "torch.Tensor", "data": base64.b64encode(...)}
```

---

## 技术细节与限制

### 递归与循环限制

```python
RECURSION_LIMIT = 50    # CallClosure 的最大嵌套深度
COROUTINE_LIMIT = 100   # 单个协程的最大迭代次数
```

限制通过检查 `unique_id` 的长度或迭代计数实现。

### ComfyUI expand 机制

ComfyUI 支持节点返回 `{"result": ..., "expand": graph}` 格式来动态添加节点。这是实现函数调用和协程迭代的基础。

### 节点 ID 命名约定

- 内部节点使用双下划线前缀：`__FunctionParam__`、`__CreateClosure__` 等
- 用户可见节点无前缀：`CallClosure`、`HighLevelMap` 等
- 展开的节点 ID：`{caller_id}_{original_id}`

### 已知限制

1. **依赖 ComfyUI 内部 API**：`PromptExecutor.execute` 的钩子、`expand` 返回格式等可能在 ComfyUI 更新后失效

2. **列表类型**：必须使用 Basic Data Handling 的 `LIST` 类型，不能使用 ComfyUI 原生的 Data List

3. **缓存失效**：副作用节点会禁用缓存，影响执行性能

4. **调试困难**：展开的图在 ComfyUI UI 中不可见，需要使用 `Inspect` 节点调试

### 扩展开发

创建自定义高阶函数，继承 `CoroutineNodeBase` 并实现 `run_coroutine`：

```python
class MyHighOrderNode(CoroutineNodeBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "function": ("CLOSURE",),
                # 其他输入...
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    CATEGORY = "my_category"
    RETURN_TYPES = (AnyType("*"),)

    def run_coroutine(self, function, **kwargs):
        # 使用 yield (closure, params) 调用函数
        result = yield (function, [arg1, arg2])
        # 处理结果...
        return final_result
```

---

## 参考资料

- [ComfyUI 源码](https://github.com/comfyanonymous/ComfyUI)
- [Basic Data Handling](https://github.com/StableLlama/ComfyUI-basic_data_handling)
- [用户使用指南](./usage.md)
- [节点参考](./node-reference.md)
