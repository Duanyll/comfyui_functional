import asyncio
import json
import uuid
import aiohttp
from PIL import Image
import io
import hashlib
from typing import Dict, Any, List, Union

class WorkflowExecutionError(Exception):
    """自定义异常，用于表示 ComfyUI 工作流执行期间的错误。"""
    def __init__(self, message, node_id, node_type, exception_message):
        super().__init__(message)
        self.node_id = node_id
        self.node_type = node_type
        self.exception_message = exception_message

    def __str__(self):
        return f"{super().__str__()} - Node ID: {self.node_id}, Type: {self.node_type}, Details: {self.exception_message}"


class ComfyClient:
    """
    一个用于与 ComfyUI 服务器交互的客户端。
    """
    def __init__(self, server_base: str, timeout: int = 600):
        """
        初始化 ComfyClient。

        :param server_base: ComfyUI 服务器的基础地址 (例如 "http://127.0.0.1:8188")。
        :param timeout: WebSocket 接收消息和执行工作流的超时时间（秒）。
        """
        self.server_base = server_base
        self.timeout = timeout
        self.base_url = server_base.rstrip('/')
        ws_base = self.base_url.replace("http://", "ws://").replace("https://", "wss://")
        self.ws_url = f"{ws_base}/ws"

    async def _upload_image(self, image: Image.Image, session: aiohttp.ClientSession) -> str:
        """
        将 PIL.Image 对象上传到 ComfyUI 服务器。

        :param image: 要上传的 PIL.Image 对象。
        :param session: aiohttp.ClientSession 对象。
        :return: 服务器上保存的文件名。
        """
        image_bytes = io.BytesIO()
        image.save(image_bytes, format='PNG')
        image_bytes.seek(0)

        hash = hashlib.sha256(image_bytes.getvalue()).hexdigest()
        filename = f"{hash}.png"
        image_bytes.seek(0)
        
        form_data = aiohttp.FormData()
        form_data.add_field('image', image_bytes, filename=filename, content_type='image/png')
        form_data.add_field('type', 'input')
        form_data.add_field('overwrite', 'true')

        async with session.post(f"{self.base_url}/upload/image", data=form_data) as response:
            response.raise_for_status()
            result = await response.json()
            return result['name']

    async def _get_image(self, filename: str, subfolder: str, folder_type: str, session: aiohttp.ClientSession) -> Image.Image:
        """
        从 ComfyUI 服务器下载图片。

        :param filename: 文件名。
        :param subfolder: 子文件夹。
        :param folder_type: 文件夹类型 ('input', 'output', 'temp')。
        :param session: aiohttp.ClientSession 对象。
        :return: 下载的 PIL.Image 对象。
        """
        params = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        async with session.get(f"{self.base_url}/view", params=params) as response:
            response.raise_for_status()
            image_data = await response.read()
            return Image.open(io.BytesIO(image_data))

    def _find_nodes(self, workflow: dict) -> tuple[dict, dict]:
        """
        在工作流中查找用 % 标记的输入和输出节点。

        :param workflow: 工作流字典。
        :return: (输入节点映射, 输出节点映射) 元组。
        """
        input_map = {}  # key: placeholder_name, value: node_id
        output_map = {} # key: node_id, value: placeholder_name

        for node_id, node_data in workflow.items():
            meta_title = node_data.get("_meta", {}).get("title")
            if meta_title and meta_title.startswith('%') and meta_title.endswith('%'):
                name = meta_title.strip('%')
                # 假设所有带 % 标记的节点都可能是输出
                output_map[node_id] = name
                # 如果节点类型是基本类型或 LoadImage，则认为是输入
                if node_data['class_type'] in [
                    "PrimitiveBoolean", "PrimitiveFloat", "PrimitiveInt", 
                    "PrimitiveString", "PrimitiveStringMultiline", "LoadImage"
                ]:
                    input_map[name] = node_id
        
        return input_map, output_map
    
    async def _prepare_workflow(
        self, 
        workflow: dict, 
        session: aiohttp.ClientSession, 
        **kwargs
    ) -> tuple[dict, dict]:
        """
        准备工作流：验证并注入参数，处理图片上传。

        :param workflow: 原始工作流模板。
        :param session: aiohttp.ClientSession 对象。
        :param kwargs: 用户提供的参数。
        :return: (准备好的工作流, 输出节点映射) 元组。
        """
        prepared_workflow = json.loads(json.dumps(workflow))
        input_map, output_map = self._find_nodes(prepared_workflow)

        for name, value in kwargs.items():
            if name not in input_map:
                raise ValueError(f"输入参数 '{name}' 在工作流中没有找到对应的 %{name}% 节点。")

            node_id = input_map[name]
            node = prepared_workflow[node_id]

            if node['class_type'] == 'LoadImage':
                if isinstance(value, str):  # 路径
                    with open(value, 'rb') as f:
                        pil_image = Image.open(io.BytesIO(f.read()))
                    filename = await self._upload_image(pil_image, session)
                elif isinstance(value, Image.Image):
                    filename = await self._upload_image(value, session)
                else:
                    raise TypeError(f"LoadImage 输入 '{name}' 的类型不受支持: {type(value).__name__}。需要 str (路径) 或 PIL.Image 对象。")
                node['inputs']['image'] = filename
            else:
                if 'value' not in node['inputs']:
                    raise ValueError(f"节点 '{node_id}' ('{name}') 被标记为输入，但在其 inputs 中没有 'value' 字段。")
                
                default_value = node['inputs']['value']
                if not isinstance(value, type(default_value)) and default_value is not None:
                     raise TypeError(f"输入 '{name}' 的类型不匹配。期望类型: {type(default_value).__name__}, 得到类型: {type(value).__name__}。")

                node['inputs']['value'] = value
        
        return prepared_workflow, output_map

    async def run_workflow(self, workflow: dict, **kwargs) -> Dict[str, List[Any]]:
        """
        异步执行一个 ComfyUI 工作流。

        :param workflow: 工作流模板 (Python 字典)。
        :param kwargs: 要注入工作流的参数。键对应模板中被 % 包围的节点标题。
        :return: 一个字典，键是输出节点的标题，值是输出内容的列表。
        """
        client_id = str(uuid.uuid4())
        
        async with aiohttp.ClientSession() as session:
            prepared_workflow, output_map = await self._prepare_workflow(workflow, session, **kwargs)
            
            async with session.ws_connect(f"{self.ws_url}?clientId={client_id}", timeout=self.timeout, max_msg_size=128*1024*1024) as ws:
                prompt_payload = {"prompt": prepared_workflow, "client_id": client_id}
                async with session.post(f"{self.base_url}/prompt", json=prompt_payload) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise ConnectionError(f"提交工作流失败: {response.status} {error_text}")
                    queue_data = await response.json()
                    prompt_id = queue_data['prompt_id']

                outputs: Dict[str, list] = {name: [] for name in output_map.values()}
                image_download_tasks = []
                current_executing_node = None
                is_done = False

                while not is_done:
                    try:
                        msg_data = await ws.receive(timeout=self.timeout)
                    except asyncio.TimeoutError:
                        raise TimeoutError(f"工作流执行超时（{self.timeout}秒）。")

                    if msg_data.type == aiohttp.WSMsgType.TEXT:
                        message = msg_data.json()
                        msg_type = message.get('type')
                        data = message.get('data', {})
                        
                        if data.get('prompt_id') != prompt_id:
                            continue
                        
                        print(f"[*] 收到消息: {msg_type} - {data.get('display_node', '')}")

                        if msg_type == 'execution_error':
                            raise WorkflowExecutionError(
                                "工作流执行出错",
                                node_id=data.get('node_id'),
                                node_type=data.get('node_type'),
                                exception_message=data.get('exception_message')
                            )
                        elif msg_type == 'execution_success':
                            is_done = True
                        elif msg_type == 'executing':
                            current_executing_node = data.get('display_node') or data.get('node')
                        elif msg_type == 'executed':
                            node_id = data.get('display_node') or data.get('node')
                            if node_id in output_map:
                                output_name = output_map[node_id]
                                output_data = data.get('output', {})
                                if 'text' in output_data:
                                    outputs[output_name].extend(output_data['text'])
                                if 'images' in output_data:
                                    for img_info in output_data['images']:
                                        task = self._get_image(
                                            img_info['filename'],
                                            img_info.get('subfolder', ''),
                                            img_info['type'],
                                            session
                                        )
                                        image_download_tasks.append((output_name, task))

                    elif msg_data.type == aiohttp.WSMsgType.BINARY:
                        if current_executing_node and current_executing_node in output_map:
                            output_name = output_map[current_executing_node]
                            image_data = msg_data.data[8:]  # 去掉8字节头部
                            image = Image.open(io.BytesIO(image_data))
                            outputs[output_name].append(image)
                    
                    elif msg_data.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                        is_done = True

            if image_download_tasks:
                results = await asyncio.gather(*(task for _, task in image_download_tasks))
                for i, (output_name, _) in enumerate(image_download_tasks):
                    outputs[output_name].append(results[i])
            
            return {k: v for k, v in outputs.items() if v}

    def run_workflow_sync(self, workflow: dict, **kwargs) -> Dict[str, List[Any]]:
        """
        同步执行一个 ComfyUI 工作流。
        这是 run_workflow 的同步版本。

        :param workflow: 工作流模板 (Python 字典)。
        :param kwargs: 要注入工作流的参数。
        :return: 一个字典，键是输出节点的标题，值是输出内容的列表。
        """
        return asyncio.run(self.run_workflow(workflow, **kwargs))


def _str_to_primitive(s: str) -> Union[bool, int, float, str]:
    """尝试将字符串转换为布尔值、整数或浮点数。"""
    s_lower = s.lower()
    if s_lower == 'true':
        return True
    if s_lower == 'false':
        return False
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return s

def main_cli():
    """命令行接口的主函数。"""

    import argparse
    import sys
    from pathlib import Path
    
    parser = argparse.ArgumentParser(
        description="ComfyUI Python 客户端命令行接口。",
        epilog="工作流的输入参数请使用 '--key=value' 的格式，例如: --prompt=\"a cat\" --steps=25"
    )
    parser.add_argument("workflow_file", help="要执行的工作流 JSON 文件路径。")
    parser.add_argument(
        "--output_dir",
        default="workflow_outputs",
        help="用于保存输出图片的目录。(默认: workflow_outputs)"
    )
    parser.add_argument(
        "--server",
        default="http://127.0.0.1:8188",
        help="ComfyUI 服务器地址。(默认: http://127.0.0.1:8188)"
    )

    args, unknown_args = parser.parse_known_args()

    # --- 解析动态工作流参数 ---
    workflow_kwargs = {}
    for arg in unknown_args:
        if not arg.startswith('--'):
            continue
        try:
            key, value = arg.lstrip('-').split("=", 1)
            workflow_kwargs[key] = _str_to_primitive(value)
        except ValueError:
            parser.error(f"无法解析参数 '{arg}'。请使用 '--key=value' 格式。")

    # --- 日志函数，输出到 stderr ---
    log = lambda *a, **kw: print(*a, **kw, file=sys.stderr)

    try:
        # --- 准备和执行工作流 ---
        log(f"[*] 正在从 {args.workflow_file} 加载工作流...")
        with open(args.workflow_file, 'r', encoding='utf-8') as f:
            workflow = json.load(f)

        client = ComfyClient(args.server)

        log(f"[*] 正在连接到 ComfyUI 服务器: {args.server}")
        log(f"[*] 使用以下输入参数执行工作流: {workflow_kwargs}")
        
        results = client.run_workflow_sync(workflow, **workflow_kwargs)

        log("[*] 工作流执行完毕。")

        # --- 处理输出 ---
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        log(f"[*] 正在处理输出结果...")

        for output_name, output_items in results.items():
            for i, item in enumerate(output_items, 1):
                if isinstance(item, Image.Image):
                    filename = f"{output_name}_{i}.png"
                    filepath = output_dir / filename
                    item.save(filepath)
                    log(f"  - 图片已保存到: {filepath}")
                else:
                    # 将文本和其他原始类型输出到 stdout
                    print(item)

    except FileNotFoundError:
        log(f"[错误] 工作流文件未找到: '{args.workflow_file}'")
        sys.exit(1)
    except Exception as e:
        log(f"[错误] 执行过程中发生错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main_cli()