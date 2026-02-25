"""
OCR 服务
使用 GLM-4.5v 和 Qwen3-VL-Plus 双模型进行 OCR 识别
"""

import re
import json
import base64
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Tuple, Any, Union

import httpx

from app.config import settings
from app.services.grading_service import SYSTEM_OCR_FAILURE_TOKEN, parse_mixed_output


class PDFSplitterAgent:
    """混合试卷拆分智能体"""
    
    # 日志目录
    LOG_DIR = Path("_logs/splitting_logs")
    
    SPLIT_DECISION_PROMPT = r"""# Role
你是一个专业的试卷分拣员。你的任务是视觉识别当前的图片是否是**另一位新同学**的答卷首页。

# Inputs
1. 当前图片：试卷的一页。
2. 上下文信息：
   - 上一位同学的学号：{prev_student_id}
   - 第一题的描述：{q1_desc}

# Decision Criteria (判定依据 - 优先级递减)
请综合以下判据进行推理：
1. **学号变化 (Critical)**: 
   - 寻找页面顶部或密封线内是否有学号/姓名填写区。
   - 识别出的学号是否与“上一位同学学号”({prev_student_id})明显不同？
   - 如果出现了新的学号，这绝对是新同学的首页。
2. **第一题特征 (High)**:
   - 页面是否包含第一题的题号（如 "1.", "一、", "Question 1"）？
   - 页面内容是否匹配第一题的描述 `{q1_desc}`？
   - 通常每个学生的答卷都从第一题开始。
3. **版面特征 (Medium)**:
   - 这页是否看起来像是一个新的开始（有明显的标题头、得分栏、试卷抬头）？
4. **排除续页 (Constraint)**:
   - 如果页面以 "接上页"、"2."、"3." 或中间计算过程开头，这很可能是续页，返回 false。

# Output Format
请先在 Markdown 代码块 ````markdown ... ```` 中进行思考分析，列出你看到的视觉证据（学号、题号、内容），然后输出 JSON 结果。

Example:
```markdown
Analysis:
1. I detected a student ID "2024002" at the top, which is different from previous "2024001".
2. The page starts with "1. Calculate..." which matches Question 1.
Conclusion: This is a new student page.
```
```json
{{
    "is_new_student": true,
    "detected_id": "2024002",
    "reason": "New student ID detected and Q1 start."
}}
```

现在请分析提供的图片：
"""

    def __init__(self, ocr_service: 'OCRService'):
        self.ocr_service = ocr_service
        # 加载配置
        self.config = settings.AGENT_MODEL_CONFIG.get("splitter", {
            "model": "glm-4.6v",
            "temperature": 0.0,
            "max_tokens": 4000
        })

    async def detect_new_student_page(
        self, 
        image_path: Path, 
        prev_student_id: str, 
        q1_desc: str
    ) -> Dict[str, Any]:
        """判断该页是否为新学生的首页"""
        
        prompt = self.SPLIT_DECISION_PROMPT.format(
            prev_student_id=prev_student_id if prev_student_id else "None (Start of Batch)",
            q1_desc=q1_desc[:200]  # 截取前200字符避免过长
        )

        last_error = None
        # 3 attempts = 1 initial + 2 retries
        for attempt in range(3):
            try:
                # 复用 OCRService 的底层 API 调用逻辑，但使用 splitter 的配置
                response_text = await self.ocr_service._call_api(
                    image_input=image_path,
                    prompt=prompt,
                    model=self.config.get("model", "glm-4.6v"),
                    provider="glm", # 强制使用 GLM (支持 Thinking/Vision 较好)
                    max_tokens=self.config.get("max_tokens", 4000),
                    temperature=self.config.get("temperature", 0.0)
                )

                if response_text:
                    # 使用 grading_service 的工具解析混合输出
                    result = parse_mixed_output(response_text)
                    
                    return {
                        "is_new_student": result.get("is_new_student", False),
                        "reason": result.get("reason", "Parsed from JSON"),
                        "detected_id": result.get("detected_id"),
                        "raw_analysis": result.get("markdown_content")
                    }
                
                print(f"WARN: Splitter attempt {attempt+1} failed (empty response) for {image_path.name}")
                
            except Exception as e:
                print(f"ERROR: Splitter attempt {attempt+1} failed for {image_path.name}: {e}")
                last_error = e

        return {"is_new_student": False, "reason": f"Failed after 3 attempts. Last error: {str(last_error)}", "detected_id": None}

    async def split_mixed_pdf_images(
        self, 
        image_files: List[Path], 
        questions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        核心逻辑：扫描所有图片，切分为不同学生的组
        
        Args:
            image_files: 已按页码排序的图片路径列表
            questions: 题目列表（用于获取第一题信息）
            
        Returns:
            List[Dict]: [{"student_id": "...", "pages": [Path, Path...]}]
        """
        if not image_files:
            return []

        # 确保日志目录存在
        self.LOG_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 初始化日志数据
        log_data = {
            "timestamp": timestamp,
            "total_pages": len(image_files),
            "decisions": []
        }

        # 获取第一题描述作为判定锚点
        q1_desc = "Question 1"
        if questions and len(questions) > 0:
            q = questions[0]
            q1_desc = f"{q.get('question', '')}"
        
        groups = []
        current_pages = []
        current_student_id = None
        
        # 强制第一页为第一个学生
        # 我们先不提取ID，等确定了分组后再统一提取或者是利用 split 过程中的 detected_id
        
        for i, img_path in enumerate(image_files):
            is_new = False
            detected_id_candidate = None
            detection_reason = ""
            
            if i == 0:
                is_new = True # 规则：第一页必须是新学生
                detection_reason = "First page of file"
            else:
                # 调用 AI 判断
                decision = await self.detect_new_student_page(
                    img_path, 
                    prev_student_id=current_student_id, 
                    q1_desc=q1_desc
                )
                is_new = decision.get("is_new_student", False)
                detected_id_candidate = decision.get("detected_id")
                detection_reason = decision.get("reason", "")
                
                # 记录详细决策日志
                log_data["decisions"].append({
                    "page_index": i,
                    "image": img_path.name,
                    "is_new": is_new,
                    "detected_id": detected_id_candidate,
                    "reason": detection_reason,
                    "raw_analysis": decision.get("raw_analysis")
                })
            
            if is_new:
                # 1. 保存之前的组 (如果有)
                if current_pages:
                    # 如果之前的组还没确定 ID (比如第一组)，尝试从之前组的第一页提取
                    # 但如果在 loop 中维护 current_student_id，这里应该已有值
                    saved_id = current_student_id or f"UNKNOWN_{timestamp}_{len(groups)+1}"
                    groups.append({
                        "student_id": saved_id,
                        "pages": current_pages
                    })
                
                # 2. 开始新组
                current_pages = [img_path]
                
                # 3. 确定新组的 student_id
                # 优先使用 Splitter 顺手识别到的 ID
                if detected_id_candidate and len(str(detected_id_candidate)) >= 6:
                    current_student_id = str(detected_id_candidate)
                else:
                    # 如果 Splitter 没读出来（可能只关注了是个新页面），则专门调用 OCR 提取一次
                    # 这里调用 extract_student_id (利用双模型)
                    print(f"📄 [Splitter] No ID from splitter, trying dedicated OCR for {img_path.name}...")
                    extracted_id, _, _ = await self.ocr_service.extract_student_id(img_path)
                    
                    if extracted_id:
                         current_student_id = extracted_id
                    else:
                        # Fallback ID generation
                         current_student_id = f"UNKNOWN_{timestamp}_{len(groups)+1}"
                
                print(f"📄 [Splitter] Found new student at {img_path.name}. ID: {current_student_id}. Reason: {detection_reason}")
                
            else:
                # 续页
                current_pages.append(img_path)
        
        # 保存最后一组
        if current_pages:
            final_id = current_student_id or f"UNKNOWN_{timestamp}_{len(groups)+1}"
            groups.append({
                "student_id": final_id,
                "pages": current_pages
            })
            
        # 写入日志文件
        log_file = self.LOG_DIR / f"split_{timestamp}.json"
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
            
        return groups


class OCRService:
    """OCR 服务 - 双模型容错识别"""
    
    # 学生信息提取 Prompt
    STUDENT_INFO_PROMPT = r"""请逐行、完整、精准、如实输出这张图片中的所有文字、数字和符号。

重要提示：
1. 这张图片包含学生信息，其中应有一个12位的学号（由12个连续的数字组成）；
2. 必须精准识别并如实记录每一个符号、数字、文字，不能遗漏、不能修改、不能猜测；
3. 特别注意：学号是12位数字，请确保完整识别所有12位数字，不要遗漏任何一位；
4. 当出现重复数字（例如"11""0000"等）时，必须完整保留所有重复数字，绝对不能合并、压缩或丢弃任何一位数字。

要求：
1. 按从上到下、从左到右的顺序输出；
2. 不要添加任何说明、解释或总结；
3. 不要试图重新排版或重新组织内容；
4. 不要额外生成"学号：""姓名："等说明性文字，只输出你在图片中看到的文字、数字和符号本身；
5. 对于数字，必须逐位准确识别，特别是学号的12位数字，要确保每一位都准确无误，包括所有重复数字。"""

    # 通用 OCR 格式化规则
    OCR_FORMATTING_RULES = r"""
1. 精确识别每个符号的细节，包括：
   - 矢量箭头（如 →, ←, ↑, ↓, ⇀, ↼ 等）
   - 字母后的撇号（如 a', b', x' 等）
   - 上标和下标（如 x², H₂O, aₙ 等）
   - 数学符号（如 ∑, ∫, ∂, ∇, ±, ×, ÷ 等）
   - 希腊字母（如 α, β, γ, θ, π, λ 等）

2. 完整还原每一行的公式和文字，保持原有的格式和结构

3. 识别被划去、涂改的内容：
   - 仔细观察图片中是否有被横线划掉、涂改、删除的内容
   - 如果内容被明显划去（有横线、涂改痕迹等），必须如实识别并记录
   - 被划去的内容使用Markdown删除线格式标记，即用两个波浪号包围：~~被划去的内容~~
   - 例如：如果看到 "错误答案" 被划掉，应记录为 ~~错误答案~~
   - 如果原内容被涂改后写了新内容，应同时记录原内容（用删除线）和新内容

4. 使用Markdown格式输出，包括：
   - 使用适当的标题层级
   - 数学公式使用LaTeX格式（如 $E=mc^2$）
   - 保持文本的换行和段落结构
   - 被划去的内容使用删除线格式（~~text~~）

5. LaTeX 格式特别说明：
   - 不要对 LaTeX 命令的反斜杠进行转义（请使用 \frac 而不是 \\frac）
   - 严禁对数学符号进行 Markdown 转义（例如不要写 \*，直接写 * 或 \times）
   - 务必检查花括号 {} 的配对，不要出现多余的右括号或遗漏左括号

6. 在输出内容时，请避免使用以下分隔符：
   - 不要使用 `---`（三个或更多连字符）作为分隔符
   - 不要使用 `***`（三个或更多星号）作为分隔符
   - 这些分隔符会被用于后续处理，请确保输出内容中不包含这些符号
"""

    # 结构化输出示例
    STRUCTURED_OUTPUT_EXAMPLE = r"""```json
{
  "responses": [
      {
        "question_id": 1,
        "student_handwriting": "解：由牛顿第二定律 $F=ma$ 可得...",
        "confidence": "high"
      },
      {
        "question_id": 2,
        "student_handwriting": "NO_ANSWER",
        "confidence": "high"
      }
  ]
}
```"""

    # 答题内容提取 Prompt
    # Note: 使用 format 或 replace 注入规则，避免 f-string 中 {{}} 转义的混淆
    _ANSWER_EXTRACTION_TEMPLATE = """请将这张手写答题纸的所有内容提取为Markdown格式。

要求：
{rules}

7. 注意小题序号可能不连续，请如实记录

请直接输出提取的内容，不要添加额外的说明。"""

    ANSWER_EXTRACTION_PROMPT = _ANSWER_EXTRACTION_TEMPLATE.format(rules=OCR_FORMATTING_RULES)

    def __init__(self):
        self.api_base = settings.DMX_API_BASE
        if not self.api_base.endswith("/v1"):
             # Simple fix to ensure we have a valid base if user provided something else, 
             # though strictly DMX usually is .../v1. 
             # But here we append /chat/completions to base usually in the caller or here.
             # ChatOpenAI uses /v1, so let's check if we need to construct the full URL for httpx.
             pass
             
        self.api_url = f"{settings.DMX_API_BASE}/chat/completions"
        self.api_key = settings.DMX_API_KEY
        
        self.thinking_enabled = settings.GLM_THINKING_ENABLED
        self.thinking_budget = settings.GLM_THINKING_BUDGET

        # 初始化 Splitter Agent
        self.splitter_agent = PDFSplitterAgent(self)
    
    def _encode_image(self, image_path: Path) -> str:
        """将图片编码为 Base64"""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')
    
    async def _call_api(
        self,
        image_input: Union[Path, List[Path]],
        prompt: str,
        model: str,
        provider: str = "glm",
        max_tokens: int = 4000,
        temperature: float = 0.0,
        timeout: float = 180.0,
        max_retries: int = 3
    ) -> Optional[str]:
        """
        通用 API 调用（支持 GLM 和 Qwen，支持单图或多图）
        """
        if not self.api_key:
            return None
        
        for attempt in range(max_retries):
            try:
                # 构建消息内容
                messages_content = []
                
                # 处理图片（支持单张或列表）
                images = image_input if isinstance(image_input, list) else [image_input]
                
                for img_path in images:
                    base64_image = self._encode_image(img_path)
                    messages_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    })
                
                # 添加文本 Prompt
                messages_content.append({
                    "type": "text",
                    "text": prompt
                })
                
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                payload = {
                    "model": model,
                    "messages": [
                        {
                            "role": "user",
                            "content": messages_content
                        }
                    ],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                }
                
                # GLM 思考模式
                if provider == "glm" and self.thinking_enabled:
                    payload["thinking"] = {
                        "type": "enabled",
                        "budget": self.thinking_budget,
                    }
                
                async with httpx.AsyncClient(timeout=timeout) as client:  # 使用配置的超时时间
                    response = await client.post(self.api_url, headers=headers, json=payload)
                    response.raise_for_status()
                    result = response.json()
                
                if "choices" in result and len(result["choices"]) > 0:
                    content = result["choices"][0]["message"]["content"]
                    return content.strip()
                
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    
            except Exception as e:
                print(f"API Call Error (Attempt {attempt+1}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                continue
        
        return None
    
    async def call_glm(self, image_input: Union[Path, List[Path]], prompt: Optional[str] = None) -> Optional[str]:
        """调用 GLM-4.5v API"""
        config = settings.OCR_MODEL_CONFIG.get("glm", {})
        return await self._call_api(
            image_input,
            prompt or self.ANSWER_EXTRACTION_PROMPT,
            config.get("model", "glm-4.6v"),
            "glm",
            max_tokens=config.get("max_tokens", 4000),
            temperature=config.get("temperature", 0.0),
            timeout=config.get("timeout", 180.0)
        )
    
    async def call_qwen(self, image_input: Union[Path, List[Path]], prompt: Optional[str] = None) -> Optional[str]:
        """调用 Qwen3-VL-Plus API"""
        config = settings.OCR_MODEL_CONFIG.get("qwen", {})
        return await self._call_api(
            image_input,
            prompt or self.ANSWER_EXTRACTION_PROMPT,
            config.get("model", "qwen-vl-ocr-latest"),
            "qwen",
            max_tokens=config.get("max_tokens", 4000),
            temperature=config.get("temperature", 0.0),
            timeout=config.get("timeout", 180.0)
        )
    
    def _get_continuous_digits_info(self, text: str) -> Tuple[int, int, str]:
        """
        分析文本中的连续数字
        返回: (是否有12位数字, 最长序列长度, 最长序列)
        """
        if not text:
            return (0, 0, "")
        
        digits_sequences = re.findall(r"\d+", text)
        if not digits_sequences:
            return (0, 0, "")
        
        max_seq = max(digits_sequences, key=len)
        max_len = len(max_seq)
        
        # 优先选择12位数字序列
        has_12 = 1 if any(len(seq) == 12 for seq in digits_sequences) else 0
        
        if has_12:
            for seq in digits_sequences:
                if len(seq) == 12:
                    return (1, 12, seq)
        
        return (0, max_len, max_seq)
    
    async def extract_student_id(self, header_image: Path) -> Tuple[Optional[str], str, str]:
        """
        从头部图片提取学号
        使用双模型并选择最佳结果
        
        Returns:
            (学号, GLM识别文本, Qwen识别文本)
        """
        # 并行调用两个模型
        glm_task = self.call_glm(header_image, self.STUDENT_INFO_PROMPT)
        qwen_task = self.call_qwen(header_image, self.STUDENT_INFO_PROMPT)
        
        glm_text, qwen_text = await asyncio.gather(glm_task, qwen_task)
        
        glm_text = glm_text or ""
        qwen_text = qwen_text or ""
        
        if not glm_text and not qwen_text:
            return None, "", ""
        
        # 分析数字
        glm_info = self._get_continuous_digits_info(glm_text)
        qwen_info = self._get_continuous_digits_info(qwen_text)
        
        glm_has_12, glm_max_len, glm_seq = glm_info
        qwen_has_12, qwen_max_len, qwen_seq = qwen_info
        
        # 选择逻辑：优先12位数字，其次最长序列
        selected_id = None
        
        if glm_has_12 and not qwen_has_12:
            selected_id = glm_seq
        elif qwen_has_12 and not glm_has_12:
            selected_id = qwen_seq
        elif glm_has_12 and qwen_has_12:
            selected_id = qwen_seq  # 两者都有时优先 Qwen
        else:
            if glm_max_len > qwen_max_len:
                selected_id = glm_seq
            else:
                selected_id = qwen_seq
        
        return selected_id, glm_text, qwen_text
    
    def _fix_latex_braces(self, text: str) -> str:
        """
        修复 LaTeX 中常见的不匹配花括号问题 (特别是多余的右括号)
        """
        if not text:
            return ""
            
        result = []
        balance = 0
        i = 0
        length = len(text)
        
        while i < length:
            char = text[i]
            
            # 处理转义字符 (skip next)
            if char == '\\':
                result.append(char)
                if i + 1 < length:
                    result.append(text[i+1])
                    i += 2
                else:
                    i += 1
                continue
            
            if char == '{':
                balance += 1
                result.append(char)
            elif char == '}':
                if balance > 0:
                    balance -= 1
                    result.append(char)
                else:
                    # 发现多余的右括号，直接丢弃
                    pass 
            else:
                result.append(char)
            
            i += 1
            
        # 补齐缺失的右括号
        if balance > 0:
            result.append('}' * balance)
            
        return "".join(result)

    def _clean_content(self, content: str) -> str:
        """
        清理 OCR 内容：移除 Markdown 代码块标记和特殊 Token
        并智能修复被 LLM 过度转义的 LaTeX 命令
        """
        if not content:
            return ""

        # 1. 移除 Markdown 代码块标记
        content = re.sub(r"^```[a-zA-Z]*\s*", "", content.strip())
        content = re.sub(r"\s*```$", "", content)
        
        # 2. 移除 GLM/Model 特殊 Token
        content = content.replace("<|begin_of_box|>", "").replace("<|end_of_box|>", "")
        content = re.sub(r"<\|.*?\|>", "", content)
        
        # 3. 智能修复双重转义的 LaTeX 命令 (Double Escaping Fix)
        # 现象：LLM 有时会在 JSON 中过度转义 LaTeX，输出 "\\frac" (String value: \\frac) 而非 "\frac"
        # 逻辑：查找 "\\Command" 格式，将其替换为 "\Command"
        # 策略：
        #   - 匹配：两个反斜杠 (\\\\ 在正则中表示) + 至少一个字母
        #   - 替换：一个反斜杠 + 捕获的字母
        #   - 排除：末尾的 `\\` (换行符) 通常后面跟空格或换行，不会被此正则捕获 ([a-zA-Z]条件)
        content = re.sub(r'\\\\([a-zA-Z]+)', r'\\\1', content)

        # 4. 修复 LaTeX 花括号匹配 (保持不变)
        content = self._fix_latex_braces(content.strip())
        
        return content

    async def extract_answer_page(self, image_path: Path) -> Tuple[Optional[str], Optional[str]]:
        """
        提取答题页内容
        
        Returns:
            (GLM识别内容, Qwen识别内容)
        """
        glm_task = self.call_glm(image_path, self.ANSWER_EXTRACTION_PROMPT)
        qwen_task = self.call_qwen(image_path, self.ANSWER_EXTRACTION_PROMPT)
        
        results = await asyncio.gather(glm_task, qwen_task)
        return (self._clean_content(results[0]), self._clean_content(results[1]))
    
    def _group_images(self, image_files: List[Path]) -> Dict[str, List[Path]]:
        """按前缀分组图片"""
        groups = {}
        pattern = re.compile(r'^(.+)-(\d+)\.png$')
        
        for img_file in sorted(image_files):
            match = pattern.match(img_file.name)
            if match:
                prefix = match.group(1)
                page_num = int(match.group(2))
                if prefix not in groups:
                    groups[prefix] = []
                groups[prefix].append((page_num, img_file))
        
        for prefix in groups:
            groups[prefix].sort(key=lambda x: x[0])
            groups[prefix] = [img for _, img in groups[prefix]]
        
        return groups
    
    async def extract_all(
        self, 
        directory: Path, 
        image_files: List[Path],
        known_student_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        处理所有图片，提取学号和答题内容
        
        Args:
            directory: 图片所在目录
            image_files: 图片文件列表
            known_student_id: 已知学号（如果提供，则跳过OCR提取学号）
            
        Returns:
            {
                "student_id": "学号",
                "responses": [{"question_id": 1, "answer": "..."}]
            }
        """
        # 分离头部图片和答题图片
        header_image = None
        answer_images = []
        
        for img in sorted(image_files, key=lambda x: x.name):
            if img.name.endswith("-0.png"):
                header_image = img
            elif img.suffix == '.png':
                answer_images.append(img)
        
        # Determine student_id
        student_id = known_student_id
        
        if not student_id:
            if not header_image:
                return None
            
            # 提取学号
            student_id, glm_header, qwen_header = await self.extract_student_id(header_image)
            
            if not student_id:
                return None
        
        # 提取答题内容
        responses = []
        
        for i, answer_image in enumerate(answer_images, start=1):
            glm_content, qwen_content = await self.extract_answer_page(answer_image)
            
            # 检查双模型是否均失败
            if not glm_content and not qwen_content:
                responses.append({
                    "question_id": i,
                    "answer": SYSTEM_OCR_FAILURE_TOKEN
                })
                continue

            # 合并双模型结果
            combined_parts = []
            if glm_content:
                combined_parts.append(f"### OCR Result Source 1\n{glm_content}")
            if qwen_content:
                combined_parts.append(f"### OCR Result Source 2\n{qwen_content}")
            
            if combined_parts:
                full_answer = "\n\n".join(combined_parts)
                responses.append({
                    "question_id": i,
                    "answer": full_answer
                })
        
        return {
            "student_id": student_id,
            "responses": responses
        }

    # 结构化提取 - 提示词模板
    _STRUCTURED_EXTRACTION_TEMPLATE = """# Role
你是一个专业的试卷手写内容提取专家。你的任务是将图片中的手写答题内容，精准地对应到给定的题目列表中。

# Context
这些图片是学生的答题纸（按顺序排列）。
试卷包含以下题目（按顺序排列）：
{question_map_str}

# Task
请分析所有图片，提取每一道题的学生作答内容。
注意：一道题的答案可能跨越图片（例如从第1页延伸到第2页），请务必将属于同一题的所有内容合并提取。

# Extraction Rules (Critical - Priority Order)
1. **第一优先级：显式题号 ID 匹配 (Explicit Question ID Matching)**
   - **核心规则**：`question_id` 必须是**纯数字**（例如 `1`, `2`）。
   - **严禁**将题目原文（如 "一、简述..."）填入 `question_id`。
   - 如果图片中出现了 "1.", "Q1", "一、" 等标记，请提取其对应的数字 ID。
   
2. **第二优先级：综合推断（当显式标记缺失时）**
   - 当没有明确题号时，请**同时结合**【内容语义】和【空间顺序】来推断它属于哪个 ID。
   - **跨页处理**：如果某页开头的孤立内容看起来是上一题的延续（如未完成的公式、接续的文字），请将其归类到上一题。

3. **格式化要求 (Formatting Compliance)**
   **这非常重要！** 提取的 `student_handwriting` 字段必须严格遵循以下 Markdown/LaTeX 规范：
{formatting_rules}
   - **Anti-Escaping Rule**: 不要对 LaTeX 命令的反斜杠进行双重转义。例如必须输出 `\\frac{{a}}{{b}}` 而不是 `\\\\frac{{a}}{{b}}`，也不要输出 `\\\\` (除非是换行)。

4. **未作答处理**：如果某道题在图片中完全找不到对应的作答痕迹，请在该题内容中标记为 "NO_ANSWER"。

# Output Format
请直接输出一个纯 JSON **对象** (Object)，包含一个 `responses` 数组。
为了协助解析，请务必将 JSON 包裹在 ```json 代码块中。

格式示例：
{output_example}
"""

    def _parse_structured_json(self, content: str) -> List[Dict[str, Any]]:
        """辅助函数：解析结构化 JSON 响应"""
        if not content:
            return []
            
        try:
            # 预处理：在解析 JSON 之前，先尝试修复字符串里常见的双反斜杠问题
            # 这是一个激进的清理，因为 parse_mixed_output 内部是用 json.loads
            # 如果 json 字符串里本身写了 "\\\\frac"，load 出来是 "\\frac"，我们需要的是 "\frac"
            pass    
        
            parsed_data = parse_mixed_output(content)
            extracted_list = []
            
            if isinstance(parsed_data, list):
                extracted_list = parsed_data
            elif isinstance(parsed_data, dict):
                if "responses" in parsed_data and isinstance(parsed_data["responses"], list):
                    extracted_list = parsed_data["responses"]
                elif "question_id" in parsed_data:
                    # 单个对象的情况
                    extracted_list = [parsed_data]
            
            # 后处理：清理提取出的文本字段
            cleaned_list = []
            for item in extracted_list:
                # 修复 ID (如果模型还是返回了文本，这里还可以兜底一次)
                # 但主要依赖 Prompt 的强化
                
                # 修复 handwriting 内容
                if "student_handwriting" in item:
                    item["student_handwriting"] = self._clean_content(item["student_handwriting"])
                cleaned_list.append(item)
                
            return cleaned_list
        except Exception as e:
            print(f"Error parsing structured JSON: {e}")
            return []

    async def extract_structured_responses(
        self,
        image_files: List[Path],
        questions: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        结构化提取答题内容（自动分题模式）
        
        Returns:
            (responses_list, debug_info_dict)
        """
        # 1. 构建题目地图
        question_map_str = "\n".join([
            f"ID:{q['id']} - 题目内容:{q['question'][:50]}..." 
            for q in questions
        ])
        
        # 使用 format 进行注入，避免 f-string 潜在的转义问题
        prompt = self._STRUCTURED_EXTRACTION_TEMPLATE.format(
            question_map_str=question_map_str,
            formatting_rules=self.OCR_FORMATTING_RULES,
            output_example=self.STRUCTURED_OUTPUT_EXAMPLE
        )

        all_responses = []
        
        # 过滤掉 -0.png (header)
        answer_images = sorted([img for img in image_files if not img.name.endswith("-0.png")], key=lambda x: x.name)
        
        # 并行调用两个模型 (GLM & Qwen)
        print("DEBUG: Calling both GLM and Qwen for structured extraction with retries...")
        
        # 定义带重试的调用包装器
        async def call_model_with_retry(model_func, imgs, pmpt, model_name):
            for i in range(3):
                try:
                    res = await model_func(imgs, pmpt)
                    # 简单的有效性检查: 只要有内容就视为成功
                    if res and len(res.strip()) > 0:
                        return res
                    print(f"WARN: {model_name} structured extraction returned empty (Attempt {i+1}/3). Retrying...")
                except Exception as e:
                    print(f"ERROR: {model_name} structured extraction failed (Attempt {i+1}/3): {e}")
            return None

        glm_task = call_model_with_retry(self.call_glm, answer_images, prompt, "GLM")
        qwen_task = call_model_with_retry(self.call_qwen, answer_images, prompt, "Qwen")
        
        results = await asyncio.gather(glm_task, qwen_task)
        glm_content, qwen_content = results
        
        # 收集 debug 信息
        debug_info = {
            "glm_raw": glm_content,
            "qwen_raw": qwen_content
        }

        # 分别解析
        glm_items = self._parse_structured_json(glm_content)
        qwen_items = self._parse_structured_json(qwen_content)
        
        print(f"DEBUG: GLM items found: {len(glm_items)}")
        print(f"DEBUG: Qwen items found: {len(qwen_items)}")

        # 聚合逻辑：按 question_id 合并
        # 优先使用 Qwen，如果 GLM 也有内容则追加名为 "### OCR Result Source 2 (GLM)"
        # 为了与普通模式保持一致 (Source 1 = GLM, Source 2 = Qwen)
        
        merged_responses = {q['id']: [] for q in questions}

        def process_items(items, source_label):
            for item in items:
                q_id = item.get("question_id")
                text = item.get("student_handwriting")
                
                # 寻找匹配的 ID (更加鲁棒的匹配逻辑)
                found_id = None
                
                # 尝试1: 直接字符串匹配
                for target_id in merged_responses.keys():
                    if str(target_id) == str(q_id):
                        found_id = target_id
                        break
                
                # 尝试2: 去除 ID: 前缀后匹配
                if found_id is None and q_id:
                    clean_qid = str(q_id).replace("ID:", "").strip()
                    for target_id in merged_responses.keys():
                        if str(target_id) == clean_qid:
                            found_id = target_id
                            break
                            
                # 尝试3: 提取纯数字匹配 (应对 "Question 1", "1." 等情况)
                if found_id is None and q_id:
                     import re
                     digits = re.findall(r"\d+", str(q_id))
                     if digits:
                         # 假设第一个数字序列就是 ID
                         candidate_id = digits[0]
                         for target_id in merged_responses.keys():
                             if str(target_id) == candidate_id:
                                 found_id = target_id
                                 break
                
                if found_id is not None and text and text != "NO_ANSWER":
                    merged_responses[found_id].append(f"### OCR Result {source_label}\n{text}")

        # 普通模式保持一致 (Source 1 = GLM, Source 2 = Qwen)
        process_items(glm_items, "Source 1")
        process_items(qwen_items, "Source 2")
        
        final_output = []
        for q_id, parts in merged_responses.items():
            if not parts:
                final_content = "NO_ANSWER"
            else:
                final_content = "\n\n".join(parts)
            
            final_output.append({
                "question_id": q_id,
                "answer": final_content
            })
            
        return final_output, debug_info
