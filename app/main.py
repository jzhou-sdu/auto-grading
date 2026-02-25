"""
AI 自动阅卷系统 - FastAPI 服务
支持 PDF/图片处理、OCR识别、LangChain 批改的完整流程

启动方式：
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

接口说明：
    POST /api/v1/grade
        - 接收学生答卷（PDF URL 或 Base64）
        - 返回评分结果

    POST /api/v1/generate-answers
        - 生成题目的标准答案和评分标准
"""

import os
import json
import re
import base64
import tempfile
import asyncio
import contextvars
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
from contextlib import asynccontextmanager

import httpx
import time
from urllib.parse import quote
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from PIL import Image

# 导入核心模块
from app.services.pdf_processor import PDFProcessor
from app.services.ocr_service import OCRService
from app.services.grading_service import GradingService
from app.services.question_generator import QuestionGenerator
from app.config import settings


# ============================================================================
# 应用初始化
# ============================================================================

# 全局信号量，用于控制并发评分任务数
grading_semaphore: Optional[asyncio.Semaphore] = None

# 任务管理器：追踪正在运行的任务及其取消状态
# 结构: {task_id: {"cancelled": bool, "status": str, "created_at": datetime}}
active_tasks: Dict[str, Dict[str, Any]] = {}
active_tasks_lock = asyncio.Lock()


class TaskCancelledException(Exception):
    """任务被取消时抛出的异常"""
    pass


async def check_task_cancelled(task_id: str) -> None:
    """检查任务是否被取消，如果被取消则抛出异常"""
    if task_id and task_id in active_tasks:
        if active_tasks[task_id].get("cancelled", False):
            print(f"🛑 Task {task_id} was cancelled by user")
            raise TaskCancelledException(f"任务 {task_id} 已被用户取消")


async def register_task(task_id: str) -> None:
    """注册一个新任务"""
    async with active_tasks_lock:
        active_tasks[task_id] = {
            "cancelled": False,
            "status": "running",
            "created_at": datetime.now().isoformat()
        }
    print(f"📋 Task registered: {task_id}")


async def unregister_task(task_id: str) -> None:
    """移除已完成的任务"""
    async with active_tasks_lock:
        if task_id in active_tasks:
            del active_tasks[task_id]
    print(f"🗑️ Task unregistered: {task_id}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时检查 API Keys
    if not settings.DMX_API_KEY:
        print("⚠️  DMX_API_KEY 未设置")
        print("   部分功能可能无法使用")
    else:
        print("✅ DMX_API_KEY 已配置")
    
    # 初始化并发信号量
    global grading_semaphore
    grading_semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_GRADING_TASKS)
    print(f"✅ 并发评分任务限制已设置为: {settings.MAX_CONCURRENT_GRADING_TASKS}")

    # 确保日志目录存在
    log_dir = Path(settings.GRADING_LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"✅ 评分日志目录已准备: {log_dir.absolute()}")
    
    gen_log_dir = Path(settings.GENERATION_LOG_DIR)
    gen_log_dir.mkdir(parents=True, exist_ok=True)
    print(f"✅ 生成答案日志目录已准备: {gen_log_dir.absolute()}")
    
    yield
    # 关闭时清理
    print("服务正在关闭...")


app = FastAPI(
    title="AI 自动阅卷系统",
    description="基于大语言模型的物理试卷自动批改系统，采用对抗式评分机制确保公平性",
    version="1.0.0",
    lifespan=lifespan
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# 日志中间件
# ============================================================================

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    
    # 打印请求信息
    print(f"\n{'='*20} Incoming Request {'='*20}")
    print(f"Method: {request.method}")
    print(f"URL: {request.url}")
    
    # 尝试读取并打印请求体（注意：这会消耗流，需要重新设置）
    try:
        body = await request.body()
        if body:
            # 避免打印过大的 Base64 数据
            try:
                body_str = body.decode('utf-8')
                if len(body_str) > 1000:
                    print(f"Body: {body_str[:500]}...[truncated]...{body_str[-100:]}")
                else:
                    print(f"Body: {body_str}")
            except Exception:
                print("Body: (binary or non-utf8 data)")
        
        # 重置请求体，以便后续路由使用
        # ⚠️ 关键修正：无论 body 是否为空，都必须重置 receive，
        # 否则 GET 请求（body为空）读取后会导致 downstream app (如 StaticFiles) 
        # 此时尝试 receive() 时无限挂起 (hang)。
        async def receive():
            return {"type": "http.request", "body": body}
        request._receive = receive
            
    except Exception as e:
        print(f"Error reading body: {e}")

    response = await call_next(request)
    
    process_time = time.time() - start_time
    print(f"\n{'='*20} Response Sent ({process_time:.2f}s) {'='*20}")
    print(f"Status Code: {response.status_code}")
    
    return response



# ============================================================================
# 请求/响应模型
# ============================================================================

class FileInput(BaseModel):
    """文件输入（支持 URL 或 Base64）"""
    url: Optional[str] = Field(None, description="PDF 文件的 URL 地址")
    base64_data: Optional[str] = Field(None, description="PDF 文件的 Base64 编码数据")
    filename: Optional[str] = Field("student.pdf", description="文件名（用于处理）")


class CriterionInput(BaseModel):
    """评分标准项"""
    description: str = Field(..., description="评分标准描述")
    points: float = Field(..., description="分值")


class ScoreInput(BaseModel):
    """评分详情"""
    max: int = Field(10, description="该题满分")
    criteria: List[CriterionInput] = Field(..., description="评分标准列表")


class QuestionInput(BaseModel):
    """题目输入"""
    id: int = Field(..., description="题目编号")
    question: str = Field(..., description="题目内容")
    answer: str = Field(..., description="参考答案（支持 LaTeX）")
    score: ScoreInput = Field(..., description="评分详情，包含 max 和 criteria")


class GradeRequest(BaseModel):
    """批改请求"""
    student_file: FileInput = Field(..., description="学生答卷文件")
    questions: List[QuestionInput] = Field(..., description="题目列表（含标准答案和评分标准）")
    examples: Optional[List[Dict[str, Any]]] = Field(None, description="Few-shot 示例（可选）")
    use_ocr_student_id: bool = Field(True, description="是否从PDF中识别学号")
    segmentation_mode: str = Field("page_per_question", description="分题模式: page_per_question / auto_segment")
    task_id: Optional[str] = Field(None, description="任务ID，用于支持取消操作")


class GradeResponse(BaseModel):
    """批改响应"""
    success: bool = Field(..., description="是否成功")
    student_id: Optional[str] = Field(None, description="学生学号")
    total_score: Optional[float] = Field(None, description="总分")
    max_score: Optional[float] = Field(None, description="满分")
    questions: Optional[List[Dict[str, Any]]] = Field(None, description="各题详细评分结果")
    error: Optional[str] = Field(None, description="错误信息（如有）")
    warnings: Optional[List[str]] = Field(None, description="警告信息列表")
    task_id: Optional[str] = Field(None, description="任务ID")
    cancelled: bool = Field(False, description="任务是否被取消")


class GenerateAnswersRequest(BaseModel):
    """生成答案请求"""
    questions_text: str = Field(..., description="题目文本（原始文本，按题号分隔）")
    model_type: str = Field("reasoner", description="模型类型: thinking/reasoner/chat/speciale")


class GenerateAnswersResponse(BaseModel):
    """生成答案响应"""
    success: bool
    questions: Optional[List[Dict[str, Any]]] = None
    logs: Optional[List[str]] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    api_keys_configured: Dict[str, bool]


# ============================================================================
# API 路由
# ============================================================================

@app.get("/health", response_model=HealthResponse, tags=["健康检查"])
async def health_check():
    """检查服务状态和 API Keys 配置"""
    return HealthResponse(
        status="healthy",
        api_keys_configured={
            "dmx": bool(settings.DMX_API_KEY)
        }
    )


@app.post("/api/v1/cancel-task/{task_id}", tags=["任务管理"])
async def cancel_task(task_id: str):
    """
    取消正在运行的任务
    
    前端在发起批改请求时传入 task_id，之后可通过此接口取消任务。
    任务将在下一个检查点被中断。
    """
    async with active_tasks_lock:
        if task_id not in active_tasks:
            return {"success": False, "message": f"任务 {task_id} 不存在或已完成"}
        
        active_tasks[task_id]["cancelled"] = True
        active_tasks[task_id]["status"] = "cancelling"
    
    print(f"🛑 Task {task_id} marked for cancellation")
    return {"success": True, "message": f"任务 {task_id} 已标记为取消，将在下一个检查点中断"}


@app.get("/api/v1/active-tasks", tags=["任务管理"])
async def list_active_tasks():
    """
    列出当前所有活跃的任务（调试用）
    """
    return {"tasks": dict(active_tasks)}


@app.post("/api/v1/split-pdf", tags=["工具"])
async def split_mixed_pdf_api(
    file: UploadFile = File(...),
    questions_json: Optional[str] = None
):
    """
    上传混合PDF，智能拆分不同学生，返回 ZIP 包
    """
    start_time = time.time()
    
    # 1. 解析题目信息 (用于辅助拆分)
    questions = []
    if questions_json:
        try:
            questions = json.loads(questions_json)
        except Exception:
            print("Warning: Failed to parse questions_json")
            
    # 2. 创建临时环境
    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        pdf_path = temp_dir / (file.filename or "uploaded.pdf")
        output_dir = temp_dir / "split_output"
        output_dir.mkdir()
        
        # 3. 保存上传文件
        content = await file.read()
        with open(pdf_path, "wb") as f:
            f.write(content)
            
        # 4. 初始化服务
        pdf_processor = PDFProcessor()
        ocr_service = OCRService()
        
        # 5. 执行拆分
        try:
            split_results = await pdf_processor.process_mixed_pdf(
                pdf_path=pdf_path,
                output_dir=output_dir,
                ocr_service=ocr_service,
                questions=questions
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Split failed: {str(e)}")
            
        # 6. 生成 Manifest
        manifest = {
            "meta": {
                "original_filename": file.filename,
                "split_time": datetime.now().isoformat(),
                "duration_seconds": time.time() - start_time
            },
            "results": split_results
        }
        with open(output_dir / "split_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
            
        # 7. 打包 ZIP
        import zipfile
        import io
        
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            # 遍历 output_dir 所有文件
            for file_path in output_dir.iterdir():
                if file_path.is_file():
                    zip_file.write(file_path, arcname=file_path.name)
        
        zip_buffer.seek(0)
        
        filename_prefix = Path(file.filename).stem if file.filename else "split_result"
        # URL encode filename to handle non-ASCII characters
        encoded_filename = quote(f"{filename_prefix}_split.zip")
        
        return Response(
            content=zip_buffer.getvalue(),
            media_type="application/zip",
            headers={
                "Content-Disposition": f"attachment; filename*=UTF-8''{encoded_filename}"
            }
        )


@app.post("/api/v1/grade", response_model=GradeResponse, tags=["批改"])
async def grade_paper(request: GradeRequest):
    """
    批改学生答卷
    
    完整流程：
    1. 下载/解析 PDF 文件
    2. PDF 转图片（裁剪头部学号区域和答题区域）
    3. OCR 识别（双模型：GLM-4.5v + Qwen3-VL-Plus）
    4. 对抗式评分（Sentinel → Teacher → Student → Principal）
    5. 返回结构化评分结果
    
    支持通过 task_id 取消任务
    """
    # 获取或生成 task_id
    task_id = request.task_id or f"task_{int(time.time() * 1000)}_{request.student_file.filename or 'unknown'}"
    
    # 初始化全流程数据记录
    full_log_data = {
        "timestamp": datetime.now().isoformat(),
        "request_id": f"req_{int(time.time())}",
        "task_id": task_id,
        "input_meta": {
             "filename": request.student_file.filename,
             "use_ocr_student_id": request.use_ocr_student_id,
             "segmentation_mode": request.segmentation_mode
        },
        "student_id": "unknown",
        "ocr_raw_output": [],  # 存储每页/每题的原始 OCR 结果
        "ocr_parsed_responses": [], # 存储清理合并后的 OCR 结果
        "grading_details": [],
        "final_result": None,
        "cancelled": False
    }

    if grading_semaphore is None:
        raise HTTPException(status_code=503, detail="系统初始化中，请稍后")

    # 注册任务
    await register_task(task_id)

    # 使用信号量控制并发
    async with grading_semaphore:
        try:
            # ========== 检查点 1: 开始处理前 ==========
            await check_task_cancelled(task_id)
            
            # 1. 获取 PDF 文件
            pdf_bytes = await _fetch_file(request.student_file)
            if not pdf_bytes:
                raise HTTPException(status_code=400, detail="无法获取 PDF 文件")
            
            # ========== 检查点 2: 文件下载后 ==========
            await check_task_cancelled(task_id)
            
            # 2. 创建临时目录处理文件
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                # Use original filename to preserve extension
                original_filename = request.student_file.filename or "student.pdf"
                file_path = temp_path / original_filename
                
                # 保存文件
                with open(file_path, 'wb') as f:
                    f.write(pdf_bytes)
                
                image_files = []
                
                # Check if it is an image
                lower_filename = original_filename.lower()
                is_image = any(lower_filename.endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp'])
                
                if is_image:
                    def _process_single_image():
                        try:
                            with Image.open(file_path) as img:
                                # Convert to RGB
                                rgb_img = img.convert('RGB')
                                
                                # Save two copies: filename-0.png (header) and filename-1.png (answer)
                                stem = Path(original_filename).stem
                                
                                h_path = temp_path / f"{stem}-0.png"
                                rgb_img.save(h_path)
                                
                                a_path = temp_path / f"{stem}-1.png"
                                rgb_img.save(a_path)
                                return [h_path, a_path]
                        except Exception as e:
                            print(f"Image processing error: {e}")
                            raise HTTPException(status_code=400, detail=f"图片处理失败: {str(e)}")

                    image_files = await asyncio.to_thread(_process_single_image)
                else:
                    # 3. PDF 转图片
                    pdf_processor = PDFProcessor()
                    image_files = await asyncio.to_thread(
                        pdf_processor.process_pdf, file_path, temp_path
                    )
                
                if not image_files:
                    raise HTTPException(status_code=400, detail="文件处理失败，未生成图片")
                
                # ========== 检查点 3: 图片处理后 ==========
                await check_task_cancelled(task_id)
                
                # 4. OCR 识别与分题
                ocr_service = OCRService()
                
                # 4.1 提取学号
                known_student_id = None
                if not request.use_ocr_student_id:
                    # Use filename as student_id (remove extension)
                    filename = request.student_file.filename or "unknown"
                    known_student_id = Path(filename).stem
                
                # 无论哪种模式，先尝试提取学号（如果需要）
                # extract_all 内部会处理学号提取
                # 但为了支持不同模式，我们这里手动分步处理
                
                student_id = known_student_id
                if not student_id:
                    # 查找 -0.png
                    header_img = next((img for img in image_files if img.name.endswith("-0.png")), None)
                    if header_img:
                        student_id, _, _ = await ocr_service.extract_student_id(header_img)
                
                # 如果学号未识别，则回退到文件名
                if not student_id:
                    print("Warning: Student ID not recognized from OCR. Falling back to filename.")
                    filename = request.student_file.filename or "unknown_student.pdf"
                    student_id = Path(filename).stem
                
                # 记录学号
                full_log_data["student_id"] = student_id

                # ========== 检查点 4: OCR 学号提取后 ==========
                await check_task_cancelled(task_id)

                # 4.2 提取答题内容
                responses = []
                questions_data = [q.model_dump() for q in request.questions]
                
                if request.segmentation_mode == "auto_segment":
                    # 分支 B: AI 自动分题
                    print("Using Auto Segmentation Mode...")
                    # 4.2.1 调用双模型并行提取 (返回 response list 和 debug info)
                    responses, debug_info = await ocr_service.extract_structured_responses(image_files, questions_data)
                    
                    full_log_data["ocr_parsed_responses"] = responses
                    # 记录原始 OCR 数据 (结构化模式下的 raw string)
                    full_log_data["ocr_raw_output"].append({
                        "mode": "auto_segment",
                        "glm_raw": debug_info.get("glm_raw"),
                        "qwen_raw": debug_info.get("qwen_raw")
                    })
                    
                else:
                    # 分支 A: 每页一题 (默认)
                    print("Using Page-per-Question Mode...")
                    # 过滤掉 -0.png
                    answer_images = sorted([img for img in image_files if not img.name.endswith("-0.png")], key=lambda x: x.name)
                    
                    # 逐页 OCR
                    for i, img_path in enumerate(answer_images):
                        # ========== 检查点 5: 每页 OCR 前 ==========
                        await check_task_cancelled(task_id)
                        
                        # 假设第 i 页对应第 i 个题目 (索引从0开始)
                        if i >= len(questions_data):
                            break # 页数多于题目数，忽略多余页
                            
                        q_id = questions_data[i]['id']
                        
                        # 调用 OCR 提取该页全部内容
                        glm_content, qwen_content = await ocr_service.extract_answer_page(img_path)
                        
                        # 记录原始 OCR 数据
                        full_log_data["ocr_raw_output"].append({
                            "question_id": q_id,
                            "page_index": i,
                            "glm_raw": glm_content,
                            "qwen_raw": qwen_content
                        })
                        
                        # 检查双模型是否均失败
                        if not glm_content and not qwen_content:
                            from app.services.grading_service import SYSTEM_OCR_FAILURE_TOKEN
                            full_answer = SYSTEM_OCR_FAILURE_TOKEN
                        else:
                            # 合并结果
                            combined_parts = []
                            if glm_content:
                                combined_parts.append(f"### OCR Result Source 1\n{glm_content}")
                            if qwen_content:
                                combined_parts.append(f"### OCR Result Source 2\n{qwen_content}")
                            
                            full_answer = "\n\n".join(combined_parts) if combined_parts else "NO_CONTENT_DETECTED"
                        
                        resp_item = {
                            "question_id": q_id,
                            "answer": full_answer
                        }
                        responses.append(resp_item)
                        full_log_data["ocr_parsed_responses"].append(resp_item)

                # ========== 检查点 6: OCR 全部完成后，评分前 ==========
                await check_task_cancelled(task_id)

                # 5. 对抗式评分
                grading_service = GradingService()
                
                # 构建题目索引
                q_index = {q['id']: q for q in questions_data}
                
                # 加载示例（如有）
                if request.examples:
                    grading_service.load_examples(request.examples)
                
                # 评分
                results = []
                total_score = 0.0
                max_total = 0.0
                
                # 遍历所有题目进行评分（确保即使没识别到答案的题也有记录）
                for q_data in questions_data:
                    # ========== 检查点 7: 每题评分前 ==========
                    await check_task_cancelled(task_id)
                    
                    q_id = q_data['id']
                    
                    # 查找对应的学生答案
                    student_resp = next((r for r in responses if str(r['question_id']) == str(q_id)), None)
                    student_answer_text = student_resp['answer'] if student_resp else "NO_ANSWER_FOUND"
                    
                    grade_result = await asyncio.to_thread(
                        grading_service.grade_question, q_data, student_answer_text
                    )
                    
                    question_result = {
                        "question_id": q_id,
                        "question": q_data['question'],
                        "student_answer": student_answer_text,
                        "final_score": grade_result['score_awarded'],
                        "max_score": q_data['score']['max'],
                        "details": grade_result
                    }
                    results.append(question_result)
                    full_log_data["grading_details"].append(question_result)
                    
                    total_score += grade_result['score_awarded']
                    max_total += q_data['score']['max']
                
                # 收集警告信息
                warnings = []
                for res in results:
                    if res.get("details", {}).get("warning"):
                        warnings.append(f"题目 {res['question_id']}: {res['details']['warning']}")

                response_obj = GradeResponse(
                    success=True,
                    student_id=student_id,
                    total_score=total_score,
                    max_score=max_total,
                    questions=results,
                    warnings=warnings if warnings else None,
                    task_id=task_id
                )
                
                full_log_data["final_result"] = response_obj.model_dump()
                return response_obj
            
        except TaskCancelledException:
            # 任务被取消
            full_log_data["cancelled"] = True
            full_log_data["error"] = "Task cancelled by user"
            print(f"⚠️ Task {task_id} was cancelled")
            return GradeResponse(
                success=False,
                error="任务已被用户取消",
                task_id=task_id,
                cancelled=True
            )
        except HTTPException:
            full_log_data["error"] = "HTTPException occurred"
            raise
        except Exception as e:
            full_log_data["error"] = str(e)
            import traceback
            trace = traceback.format_exc()
            full_log_data["traceback"] = trace
            print(f"CRITICAL ERROR in grade_paper: {str(e)}")
            return GradeResponse(
                success=False,
                error=f"批改过程中发生错误: {str(e)}",
                task_id=task_id
            )
        finally:
            # 注销任务
            await unregister_task(task_id)
            
            # 最终：保存完整日志到文件
            try:
                # 生成文件名: timestamp_studentID.json
                safe_id = re.sub(r'[\\/*?:"<>|]', '_', str(full_log_data.get("student_id", "unknown")))
                timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                log_filename = f"{timestamp_str}_{safe_id}.json"
                log_path = Path(settings.GRADING_LOG_DIR) / log_filename
                
                with open(log_path, "w", encoding="utf-8") as f:
                    json.dump(full_log_data, f, ensure_ascii=False, indent=2)
                
                print(f"📝 Full grading log saved to: {log_path.absolute()}")
                
            except Exception as log_err:
                print(f"❌ Failed to save grading log: {log_err}")



@app.post("/api/v1/generate-answers", response_model=GenerateAnswersResponse, tags=["题目生成"])
async def generate_answers(request: GenerateAnswersRequest):
    """
    根据题目文本生成标准答案和评分标准
    
    输入：原始题目文本（按题号分隔）
    输出：包含答案和评分标准的结构化 JSON
    """
    # 记录生成请求的元数据
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "request_type": "generate_answers",
        "input_text": request.questions_text,
        "model_type": request.model_type,
        "result": None,
        "error": None
    }
    
    try:
        generator = QuestionGenerator()
        result = await asyncio.to_thread(
            generator.generate_from_text, request.questions_text
        )
        
        log_data["result"] = result
        
        return GenerateAnswersResponse(
            success=True,
            questions=result["questions"],
            logs=result["logs"]
        )
        
    except Exception as e:
        log_data["error"] = str(e)
        import traceback
        log_data["traceback"] = traceback.format_exc()
        
        return GenerateAnswersResponse(
            success=False,
            error=f"生成答案时发生错误: {str(e)}"
        )
    finally:
        # 保存生成日志到文件
        try:
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = f"gen_{timestamp_str}.json"
            log_path = Path(settings.GENERATION_LOG_DIR) / log_filename
            
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(log_data, f, ensure_ascii=False, indent=2)
            print(f"📝 Generation log saved to: {log_path.absolute()}")
        except Exception as log_err:
            print(f"❌ Failed to save generation log: {log_err}")


@app.post("/api/v1/ocr", tags=["OCR"])
async def ocr_images(files: List[FileInput]):
    """
    对图片进行 OCR 识别（独立接口，用于调试）
    """
    try:
        ocr_service = OCRService()
        results = []
        
        for file_input in files:
            file_bytes = await _fetch_file(file_input)
            if file_bytes:
                # 保存到临时文件
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                    tmp.write(file_bytes)
                    tmp_path = Path(tmp.name)
                
                try:
                    glm_text = await ocr_service.call_glm(tmp_path)
                    qwen_text = await ocr_service.call_qwen(tmp_path)
                    results.append({
                        "filename": file_input.filename,
                        "glm_result": glm_text,
                        "qwen_result": qwen_text
                    })
                finally:
                    tmp_path.unlink(missing_ok=True)
        
        return {"success": True, "results": results}
        
    except Exception as e:
        return {"success": False, "error": str(e)}


# ============================================================================
# 辅助函数
# ============================================================================

async def _fetch_file(file_input: FileInput) -> Optional[bytes]:
    """从 URL 或 Base64 获取文件内容"""
    if file_input.base64_data:
        try:
            return base64.b64decode(file_input.base64_data)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Base64 解码失败: {e}")
    
    if file_input.url:
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.get(file_input.url)
                response.raise_for_status()
                return response.content
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"下载文件失败: {e}")
    
    return None


# 挂载静态文件 (必须放在所有 API 路由之后，否则会拦截 API 请求)
static_dir = Path(__file__).parent / "static"
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")


# ============================================================================
# 入口
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
