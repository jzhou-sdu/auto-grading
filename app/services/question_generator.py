"""
题目生成服务
根据文本生成标准答案和评分标准
"""

import re
import json
import requests
from typing import List, Dict, Any, Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.runnables import RunnableLambda, Runnable
from pydantic import BaseModel, Field

from app.config import settings
from app.services.grading_service import fix_json_latex, parse_mixed_output


# ============================================================================
# 数据结构
# ============================================================================

class ScoringCriterion(BaseModel):
    description: str = Field(description="评分标准的描述")
    points: float = Field(description="该标准的分值")

class ScoringDetails(BaseModel):
    max: float = Field(description="问题总分")
    criteria: List[ScoringCriterion] = Field(description="评分标准列表")

class QuestionAnswer(BaseModel):
    markdown_content: Optional[str] = Field(None, description="详细的参考答案内容（Markdown格式）")
    id: int = Field(description="问题编号")
    question: str = Field(description="问题内容")
    answer: str = Field(description="参考答案（简略版或指向Markdown）")
    score: ScoringDetails = Field(description="评分详情")
    feedback: Optional[str] = Field(None, description="审核反馈")

class QuestionMeta(BaseModel):
    id: int = Field(description="题目编号")
    content: str = Field(description="题目内容")
    domain: str = Field(description="物理学分支领域")
    keywords: List[str] = Field(description="核心关键词列表")

class QuestionList(BaseModel):
    questions: List[QuestionMeta] = Field(description="题目列表")

class ReviewResult(BaseModel):
    passed: bool = Field(description="是否通过审核")
    feedback: str = Field(description="审核意见简述")
    # Added markdown_content to support mixed output parsing
    markdown_content: Optional[str] = Field(None, description="详细的审核反馈（Markdown格式）")


class DesignerOutput(BaseModel):
    score: ScoringDetails = Field(description="评分详情")


# ============================================================================
# Generating Service
# ============================================================================

# ============================================================================
# Few-Shot Examples
# ============================================================================

FEW_SHOT_ANALYST = r"""
输入："1. 计算氢原子基态能量... 2. 在一个法布里-珀罗谐振腔中..."
输出：
### 试卷结构分析
这份试卷包含两道大题。
第一题涉及量子力学中的氢原子模型，需要计算基态能量。
第二题涉及光学中的法布里-珀罗谐振腔，考察干涉和精细度概念。

```json
{
    "questions": [
        {"id": 1, "content": "计算氢原子基态能量...", "domain": "量子力学", "keywords": ["氢原子势", "薛定谔方程"]},
        {"id": 2, "content": "在一个法布里-珀罗谐振腔中...", "domain": "光学", "keywords": ["谐振腔", "干涉", "精细度"]}
    ]
}
```

输入："三、考虑玻色-爱因斯坦凝聚体的激发谱... (a) 写出哈密顿量 (b) 求解激发谱"
输出：
### 试卷结构分析
这是一道关于统计力学的综合题，主要考察玻色-爱因斯坦凝聚（BEC）的性质。
题目分为两个小问，分别要求写出哈密顿量和求解激发谱（博戈留波夫变换）。

```json
{
    "questions": [
        {"id": 3, "content": "考虑玻色-爱因斯坦凝聚体的激发谱... (a) 写出哈密顿量 (b) 求解激发谱", "domain": "统计力学", "keywords": ["玻色爱因斯坦凝聚", "博戈留波夫变换"]}
    ]
}
```
"""

FEW_SHOT_TA = r"""
问题：
计算一个静止质量为 $m$ 的粒子，以速度 $v=0.6c$ 运动时的动能。

答案：
根据相对论动能公式：
$$ E_k = (\gamma - 1)mc^2 $$
其中洛伦兹因子 $\gamma$ 定义为：
$$ \gamma = \frac{1}{\sqrt{1 - v^2/c^2}} $$
代入 $v=0.6c$：
$$ \gamma = \frac{1}{\sqrt{1 - 0.36}} = \frac{1}{0.8} = 1.25 $$
因此，粒子的动能为：
$$ E_k = (1.25 - 1)mc^2 = 0.25mc^2 $$
"""

FEW_SHOT_PROFESSOR = r"""
**示例 1 (审核不通过)：**
问题：简述热力学第二定律的开尔文表述。
答案：不可能从单一热源吸热使之完全变为有用功。
输出：
### 审核反馈
助教的回答遗漏了关键的限定条件。开尔文表述的核心在于“不产生其他影响”。
正确的表述应为：不可能从单一热源吸热使之完全变为有用功，**而不产生其他影响**。
缺少这个条件，该过程在物理上是可行的（例如理想气体等温膨胀），但不构成循环。

```json
{
    "passed": false,
    "feedback": "See Markdown details"
}
```

**示例 2 (审核通过)：**
问题：计算 $f(x) = x^2$ 的导数。
答案：根据幂函数求导法则 $(x^n)' = nx^{n-1}$。这里 $n=2$，所以 $f'(x) = 2x^{2-1} = 2x$。
输出：
### 审核反馈
解答正确，步骤清晰。
公式推导符合规范。

```json
{
    "passed": true,
    "feedback": "See Markdown details"
}
```
"""

FEW_SHOT_DESIGNER = r"""
题目：证明2个类空事件，总是能找到一个参照系，在这个参照系中，二个事件同一时间发生。
输出：
### 评分标准设计思路
本题主要考察狭义相对论中的洛伦兹变换和类空间隔的概念。
评分应侧重于物理概念的理解（60%）和数学推导的正确性（40%）。
需要检查学生是否正确写出了洛伦兹变换公式，并利用类空条件证明时间差为零的可能性。

```json
{
  "score": {
    "max": 10,
    "criteria": [
      {"description": "物理概念的正确理解与应用（例如设置了合理的事件坐标，应用了洛伦兹变换等）", "points": 6},
      {"description": "正确给出相对速度的公式。若结果错误，中间推导过程部分正确，给予相应的步骤分。", "points": 3},
      {"description": "速度合法性检查 (|v| < c)", "points": 1}
    ]
  }
}
```

题目：康普顿散射：推导散射角 $\theta$ 下射出光子的能量 E 的表达式。
输出：
### 评分标准设计思路
本题考察相对论动力学中的能量守恒和动量守恒。
关键点在于列出正确的守恒方程，并进行代数运算消去电子动量。
评分标准应包含方程建立（6分）和结果推导（4分）。

```json
{
  "score": {
    "max": 10,
    "criteria": [
      {"description": "物理概念正确理解与应用（例如，能量守恒，动量守恒，粒子的在壳条件等，正确列出解题所需的方程）", "points": 6},
      {"description": "在物理概念基本正确的基础上，正确给出最终结果。若最终结果错误，中间推导过程部分正确，给予相应的步骤分。", "points": 4}
    ]
  }
}
```

题目：简答题：
1) 估算广岛原子弹爆炸有多少公斤物质转化为能量？
2) 一个不带电的小磁针在静电场中高速运动，是否会受到力的作用，并解释原因？
3) 一个超过车库长度的木杆，在接近光速移动时发生洛伦兹收缩，比车库短，可完全容纳进车库；但在杆自身静止系中，车库高速移动而收缩，似乎无法容纳木杆。请解释此悖论？
4) 电场和磁场是否是洛伦兹协变四矢量的空间三分量？
输出：
### 评分标准设计思路
这是一道综合简答题，包含四个独立的小问，每问2.5分。
1) 考察质能方程估算。
2) 考察电磁场的相对论变换（运动磁矩产生电偶极矩）。
3) 考察同时性的相对性（梯子悖论）。
4) 考察电磁场张量的性质。

```json
{
  "score": {
    "max": 10,
    "criteria": [
      {"description": "1) 质能估算，差距在一个数量级以内得5分，差距在两个数量级以内得3分，差距在三个数量级以内得2分，差距在四个数量级以内得1分，否则得0分", "points": 2.5},
      {"description": "2) 电场中运动磁矩的受力分析", "points": 2.5},
      {"description": "3) 梯子（木杆）悖论解释", "points": 2.5},
      {"description": "4) 电磁场参考系变换", "points": 2.5}
    ]
  }
}
```
"""

# ============================================================================
# 生成器类
# ============================================================================

class QuestionGenerator:
    """问题-答案-评分生成器 (多智能体版)"""
    
    def __init__(self):
        self.config = settings.AGENT_MODEL_CONFIG
        self.logs = []

    def _log(self, message: str):
        print(message)
        self.logs.append(message)

    def _get_llm(self, role: str) -> Runnable:
        """根据角色获取配置的模型"""
        # 获取角色配置，如果不存在则使用默认配置
        role_config = self.config.get(role, {
            "model": "gemini-3-flash-preview",
            "temperature": 0.0,
            "timeout": 180,
            "max_tokens": 8000
        })
        
        model_name = role_config.get("model", "gemini-3-flash-preview")
        temperature = role_config.get("temperature", 0.0)
        timeout = role_config.get("timeout", 180)
        max_tokens = role_config.get("max_tokens", 8000)
        model_kwargs = role_config.get("model_kwargs", {}).copy()
        extra_body = model_kwargs.pop("extra_body", None)
        
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=settings.DMX_API_KEY,
            base_url=settings.DMX_API_BASE,
            timeout=timeout,
            max_tokens=max_tokens,
            model_kwargs=model_kwargs,
            extra_body=extra_body
        )

    # ------------------------------------------------------------------------
    # 1. 题目分析员 (Analyst)
    # ------------------------------------------------------------------------
    def _analyze_questions(self, text: str) -> List[QuestionMeta]:
        llm = self._get_llm("analyst")
        parser = JsonOutputParser(pydantic_object=QuestionList)
        
        system_prompt = r"""你是一位资深的物理系教务专员。你的任务是将输入的试卷文本切分为独立的大题，并分析每道题所属的物理学分支领域和核心关键词。

优先将题目归类为以下四大领域之一（请严格输出括号内的英文或中文标准名称）：
- 理论力学 (Classical Mechanics)
- 电动力学 (Electrodynamics)
- 量子力学 (Quantum Mechanics)
- 热力学与统计物理 (Thermodynamics & Statistical Physics)

如果题目明显属于其他分支，则如实分析。

请先在 Markdown 部分进行详细分析，最后严格按照 JSON 格式输出，包含一个 "questions" 字段。
**注意：** domain 字段请尽量匹配上述四大领域的标准名称，不要自行组合（例如不要输出“理论力学(Classical Mechanics)”，而是只输出“理论力学”或“Classical Mechanics”）。

**特别重要：**
请务必将 JSON 输出包裹在 markdown 代码块中，如下所示：
```json
{{
    "questions": [ ... ]
}}
```

**输出格式说明：**
{format_instructions}

**Few-shot 示例：**

{few_shot}
"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "请分析以下试卷内容：\n\n{text}")
        ])
        
        chain = prompt.partial(
            format_instructions=parser.get_format_instructions(),
            few_shot=FEW_SHOT_ANALYST
        ) | llm | RunnableLambda(parse_mixed_output)
        
        try:
            # 尝试直接解析
            result = chain.invoke({"text": text})
            if isinstance(result, dict) and "markdown_content" in result:
                 self._log(f"Analyst Analysis: {result['markdown_content']}")
            self._log(f"Analyst Output: {json.dumps(result, ensure_ascii=False, indent=2)}")
            
            # 兼容处理：如果返回的是列表（旧模型习惯），尝试直接转换
            if isinstance(result, list):
                return [QuestionMeta(**item) for item in result]
                
            questions_data = result.get("questions", [])
            if not questions_data:
                self._log("Analyst extracted 0 questions from JSON. Attempting fallback parse.")
                return self._fallback_split(text)
                
            return [QuestionMeta(**item) for item in questions_data]
            
        except Exception as e:
            self._log(f"Analyst failed: {e}")
            # 降级策略：如果 LLM 解析失败，使用简单的正则切分，并给默认标签
            return self._fallback_split(text)

    def _fallback_split(self, text: str) -> List[QuestionMeta]:
        """简单的正则切分作为后备"""
        pattern = r'(?:^|\n)(?:[一二三四五六七八九十]+[、：:]|\d+[.、：:])'
        splits = re.split(pattern, text)
        if splits and not splits[0].strip():
            splits = splits[1:]
        
        results = []
        for i, content in enumerate(splits, 1):
            if content.strip():
                results.append(QuestionMeta(
                    id=i, 
                    content=content.strip(), 
                    domain="物理", 
                    keywords=["综合"]
                ))
        return results

    # ------------------------------------------------------------------------
    # 2. 解题助教 (Teaching Assistant)
    # ------------------------------------------------------------------------
    def _solve_question(self, question: str, domain: str, keywords: List[str], feedback: str = "") -> str:
        llm = self._get_llm("ta")
        # 动态教科书引用（使用 settings.DOMAIN_MAPPING）
        textbook_ref = ""
        domain_mapping = settings.DOMAIN_MAPPING
        for key, info in domain_mapping.items():
            if key in domain:
                textbook_ref = f"请参考 {info['textbook']} 的定义和符号体系，并严格遵循其学术规范。"
                break

        # Gemini 3 Pro (Thinking) 优化 Prompt
        # 移除过多的格式限制，相信 Thinking 模型的原生推理能力
        # 但保留核心的学术要求和 Tex 格式要求
        system_prompt = r"""你是 {domain} 领域的顶尖专家，精通 {keywords_str} 等相关理论。{textbook_ref}

请为给定的物理问题撰写一份完美的标准参考答案。

要求：
1. **深度思考**：充分利用你的推理能力，仔细审视每一个物理假设和数学步骤。
2. **格式规范**：只能使用 Standard LaTeX 格式（`$` 和 `$$`）。严禁使用 `\(` 或 `\)`。
3. **详尽过程**：必须展示所有的中间计算步骤，不得省略。
4. **错题修正**：如果收到审核教授的反馈，请结合反馈深度反思并修正。

**Few-shot 示例：**

{few_shot}
"""
        user_prompt = "问题：\n{question}"
        invoke_args = {
            "domain": domain, 
            "keywords_str": ", ".join(keywords),
            "question": question,
            "textbook_ref": textbook_ref,
            "few_shot": FEW_SHOT_TA
        }

        if feedback:
            # 使用变量占位符，而不是直接拼接字符串，防止 feedback 中的 {} 被识别为变量
            user_prompt += "\n\n来自审核教授的修改意见：\n{feedback}\n\n请根据意见重新生成答案。"
            invoke_args["feedback"] = feedback
            
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", user_prompt)
        ])
        
        chain = prompt | llm | StrOutputParser()
        try:
            result = chain.invoke(invoke_args)
            if not result:
                self._log("  [警告] 助教生成的内容为空！可能是模型超时或被截断。")
                return ""
            self._log(f"  [助教生成答案] (Length: {len(result)})\n{result}")
            return result
        except Exception as e:
            self._log(f"  [错误] 助教生成失败: {e}")
            raise e

    # ------------------------------------------------------------------------
    # 3. 审核教授 (Professor)
    # ------------------------------------------------------------------------
    def _review_answer(self, question: str, answer: str, domain: str, keywords: List[str]) -> ReviewResult:
        llm = self._get_llm("professor")
        parser = JsonOutputParser(pydantic_object=ReviewResult)
        
        # 动态教科书引用
        textbook_ref = ""
        for key, info in settings.DOMAIN_MAPPING.items():
            if key in domain:
                textbook_ref = f"你的审核标准应以 {info['textbook']} 为准。"
                break

        system_prompt = r"""你是 {domain} 领域的权威教授，精通 {keywords_str}。{textbook_ref}
你以治学严谨、眼光挑剔著称。
你的任务是审核助教撰写的参考答案。

审核标准：
1. **内容优先**：首先检查物理概念是否准确、数学推导是否正确。这是**最重要**的审核标准。
2. **完整性**：解题步骤是否完整？是否遗漏了关键的中间步骤、假设说明或边界条件讨论？
3. **符号规范**：符号使用是否符合物理学通用规范？
4. **格式宽容度**：对于 LaTeX 格式问题（如 `\(` vs `$`），只要不影响阅读和歧义，**不要**因此拒绝通过。
5. **错误报告**：如果发现错误，请列出**所有**发现的错误，而不仅仅是第一个。

如果发现任何**内容错误**或**严重格式错误**，请明确指出错误位置和原因，并拒绝通过（passed=false）。
只有当答案在物理和数学上正确，且步骤详尽时，才予以通过（passed=true）。

请先在 Markdown 部分提供详细的审核反馈（包含公式），最后严格按照 JSON 格式输出。
**注意：** 在 JSON 中引用 LaTeX 公式时，直接使用单反斜杠（如 `\alpha`），**不需要**双写反斜杠，系统会自动处理。
**Few-shot 示例：**

{few_shot}
"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt + "\n\n{format_instructions}"),
            ("human", "问题：\n{question}\n\n助教提供的答案：\n{answer}")
        ])
        
        chain = prompt.partial(
            format_instructions=parser.get_format_instructions(),
            few_shot=FEW_SHOT_PROFESSOR
        ) | llm | RunnableLambda(parse_mixed_output)
        
        try:
            result_dict = chain.invoke({
                "domain": domain,
                "keywords_str": ", ".join(keywords),
                "textbook_ref": textbook_ref,
                "question": question,
                "answer": answer
            })
            
            # 确保返回的是 ReviewResult 对象
            if isinstance(result_dict, dict):
                review_result = ReviewResult(**result_dict)
            else:
                # 理论上不应该走到这里，除非 parser 行为改变
                review_result = ReviewResult(passed=True, feedback="解析结果格式异常，默认通过")
                
            if review_result.markdown_content:
                self._log(f"  [教授详细反馈] {review_result.markdown_content}")
                # 移除重复赋值，避免 feedback 字段冗余
                # review_result.feedback = review_result.markdown_content
                
            self._log(f"  [教授审核结果] Passed: {review_result.passed}\n  Feedback: {review_result.feedback}")
            return review_result
        except Exception as e:
            self._log(f"Review failed: {e}, assuming passed to avoid deadlock.")
            return ReviewResult(passed=True, feedback="自动审核出错，默认通过。")

    # ------------------------------------------------------------------------
    # 4. 评分设计师 (Designer)
    # ------------------------------------------------------------------------
    def _design_grading(self, question: str, final_answer: str, question_id: int, feedback: str = "") -> Dict[str, Any]:
        llm = self._get_llm("designer")
        parser = JsonOutputParser(pydantic_object=DesignerOutput)
        
        system_prompt = r"""你是一位专业的物理考试评分标准制定者。
你的任务是根据【题目】设计评分标准。
【最终参考答案】仅供参考，用于验证题目可解以及确定关键得分点。

**设计原则：**
1. **以题目为核心**：评分标准应基于题目考察的物理概念和逻辑步骤，而非死板地匹配参考答案的特定解法。
2. **兼容多种解法**：物理题通常有多种解法（如使用不同的参考系、使用四维矢量 vs 分量计算）。评分标准应描述为“正确应用xx原理”、“列出正确的xx方程”，而不是“写出公式 x=y”。
3. **注重物理图景**：给予“物理概念理解”较大的权重。
4. **步骤分**：明确中间步骤的得分，允许结果错误但过程正确的情况得分。
5. **总分固定**：每个大题满分 **10分**。
6. **小题分项**：如果题目包含明确的小题（如 1), 2) 或 (a), (b)），请直接将每个小题作为一个独立的评分标准项，并根据难度合理分配分数。

请先在 Markdown 部分简要说明评分标准的设计思路，最后严格按照 JSON 格式输出评分标准（score 字段）。
**注意：** 不需要输出参考答案，只需要输出评分标准。

**Few-shot 示例：**

{few_shot}

**输出要求：**
- 严格遵循 JSON 格式。
- 包含 `score` (包含 `max` 和 `criteria`)。
- `score.max` 必须为 10。
"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt + "\n\n{format_instructions}"),
            ("human", "题号：{question_id}\n\n问题：\n{question}\n\n最终参考答案：\n{final_answer}")
        ])
        
        chain = prompt.partial(
            format_instructions=parser.get_format_instructions(),
            few_shot=FEW_SHOT_DESIGNER
        ) | llm | RunnableLambda(parse_mixed_output)
        
        result = chain.invoke({
            "question_id": question_id,
            "question": question,
            "final_answer": final_answer
        })
        
        # 手动构建最终结果，直接使用传入的 final_answer
        final_result = {
            "id": question_id,
            "question": question,
            "answer": final_answer,
            "score": result.get("score", {"max": 10, "criteria": []}),
            "markdown_content": result.get("markdown_content")
        }
        
        self._log(f"Designer Output: {json.dumps(final_result, ensure_ascii=False, indent=2)}")
        
        # 注入反馈信息
        if feedback:
            final_result['feedback'] = feedback
            
        return final_result

    # ------------------------------------------------------------------------
    # 主流程
    # ------------------------------------------------------------------------
    def generate_from_text(self, text: str) -> Dict[str, Any]:
        """
        从原始文本生成问题列表（多智能体协作流程）
        """
        self.logs = []
        self._log("--- 开始题目分析 (Analyst Agent) ---")
        questions_meta = self._analyze_questions(text)
        self._log(f"识别到 {len(questions_meta)} 道题目")
        
        results = []
        
        for q_meta in questions_meta:
            self._log(f"\n正在处理第 {q_meta.id} 题: {q_meta.domain} - {q_meta.keywords}")
            
            current_answer = ""
            feedback = ""
            max_loops = settings.GENERATION_REVIEW_LOOPS
            
            # 解题-审核循环
            for attempt in range(max_loops):
                try:
                    self._log(f"  [第 {attempt+1} 轮] 解题助教正在作答...")
                    current_answer = self._solve_question(q_meta.content, q_meta.domain, q_meta.keywords, feedback)
                    
                    # 如果是最后一次循环，且不需要强制审核通过才结束，则可以跳过审核直接输出（或者审核了也不影响结果）
                    # 但为了获取反馈日志，我们还是执行审核
                    self._log(f"  [第 {attempt+1} 轮] 审核教授正在审核...")
                    review = self._review_answer(q_meta.content, current_answer, q_meta.domain, q_meta.keywords)
                    
                    if review.passed:
                        self._log("  审核通过！")
                        break
                    else:
                        # 优先使用 Markdown 详细反馈，如果为空则使用简略反馈
                        full_feedback = review.markdown_content if review.markdown_content else review.feedback
                        self._log(f"  审核未通过: {full_feedback[:100]}...")
                        feedback = full_feedback
                        
                        if attempt == max_loops - 1:
                            self._log("  已达到最大审核轮次，将使用当前版本的答案继续后续流程。")
                            
                except Exception as e:
                    self._log(f"  [错误] 第 {attempt+1} 轮循环发生异常: {e}")
                    if attempt == max_loops - 1:
                        self._log("  达到最大重试次数或发生不可恢复错误，使用当前结果继续。")
            
            # 评分设计
            self._log("  评分设计师正在制定标准...")
            try:
                if not current_answer:
                    self._log("  警告: 没有生成有效的参考答案，跳过评分设计。")
                    continue
                    
                final_json = self._design_grading(q_meta.content, current_answer, q_meta.id, feedback)
                # 确保 ID 一致
                final_json['id'] = q_meta.id
                # 注入元数据供评分阶段使用
                final_json['domain'] = q_meta.domain
                final_json['keywords'] = q_meta.keywords
                results.append(final_json)
            except Exception as e:
                self._log(f"  评分设计失败: {e}")
        
        return {
            "questions": results,
            "logs": self.logs
        }
