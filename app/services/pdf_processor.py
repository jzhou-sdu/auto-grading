"""
PDF 处理服务
将 PDF 转换为图片，并进行裁剪处理
"""

from pathlib import Path
from typing import List, Optional, Dict, Any
import fitz  # PyMuPDF
from PIL import Image
import shutil
import asyncio
from app.config import settings


class PDFProcessor:
    """PDF 处理器"""
    
    def __init__(self):
        self.dpi = settings.PDF_DPI
        self.top_region_height_cm = settings.TOP_REGION_HEIGHT_CM
        self.crop_start_y_cm = settings.CROP_START_Y_CM
        self.cm_to_inch = 0.393701
    
    def cm_to_pixels(self, cm: float) -> int:
        """厘米转像素"""
        inches = cm * self.cm_to_inch
        return int(inches * self.dpi)
    
    def process_pdf(self, pdf_path: Path, output_dir: Path) -> List[Path]:
        """
        处理单个 PDF 文件
        
        流程：
        1. 提取首页完整内容作为 -0.png（用于学号识别）
        2. 所有页面完整内容作为 -1.png, -2.png 等（答题区域）
        
        Args:
            pdf_path: PDF 文件路径
            output_dir: 输出目录
            
        Returns:
            生成的图片文件路径列表
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        base_name = pdf_path.stem
        output_files = []
        
        try:
            doc = fitz.open(pdf_path)
            
            if doc.page_count == 0:
                return []
            
            # 转换所有页面为图片
            zoom = self.dpi / 72
            mat = fitz.Matrix(zoom, zoom)
            images = []
            
            for page in doc:
                pix = page.get_pixmap(matrix=mat)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                images.append(img)
            
            doc.close()
            
            # 步骤1: 提取首页完整内容（学号区域）
            first_page = images[0]
            top_output_path = output_dir / f"{base_name}-0.png"
            first_page.save(top_output_path, 'PNG')
            output_files.append(top_output_path)
            
            # 步骤2: 保存所有页面的完整内容
            for i, img in enumerate(images, start=1):
                output_path = output_dir / f"{base_name}-{i}.png"
                img.save(output_path, 'PNG')
                output_files.append(output_path)
            
            return output_files
            
        except Exception as e:
            raise RuntimeError(f"PDF 处理失败: {e}")
    
    def process_directory(self, input_dir: Path, output_dir: Optional[Path] = None) -> List[Path]:
        """
        处理目录中的所有 PDF 文件
        
        Args:
            input_dir: 输入目录
            output_dir: 输出目录（默认与输入目录相同）
            
        Returns:
            所有生成的图片文件路径
        """
        if output_dir is None:
            output_dir = input_dir
        
        all_files = []
        pdf_files = sorted(input_dir.glob("*.pdf"))
        
        for pdf_file in pdf_files:
            files = self.process_pdf(pdf_file, output_dir)
            all_files.extend(files)
        
        return all_files

    async def process_mixed_pdf(
        self, 
        pdf_path: Path, 
        output_dir: Path, 
        ocr_service: Any, # Avoid circular import, pass instance
        questions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        处理混合 PDF：先拆分为临时图片，再由 AI 分组，最后重命名为标准格式
        
        Returns:
            分组信息 [{"student_id": "...", "files": [...]}]
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        temp_dir = output_dir / "temp_split"
        temp_dir.mkdir(exist_ok=True) # 临时存放原始转换图片
        
        try:
            # Step 1: 转换所有页面为临时图片 (Move to thread to avoid blocking event loop)
            def _convert_pages():
                print(f"📄 [PDFProcessor] Converting pages for {pdf_path.name}...", flush=True)
                doc = fitz.open(pdf_path)
                if doc.page_count == 0:
                    doc.close()
                    return []
                
                zoom = self.dpi / 72
                mat = fitz.Matrix(zoom, zoom)
                
                paths = []
                for i, page in enumerate(doc):
                    pix = page.get_pixmap(matrix=mat)
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    
                    # 保存为 temp-001.png 格式以便排序
                    p = temp_dir / f"temp-{i:04d}.png"
                    img.save(p, 'PNG')
                    paths.append(p)
                doc.close()
                print(f"   [PDFProcessor] Converted {len(paths)} pages.", flush=True)
                return paths

            print(f"📄 [PDFProcessor] Starting async conversion...", flush=True)
            temp_images = await asyncio.to_thread(_convert_pages)
            if not temp_images:
                return []
            
            # 2. 调用 AI 拆分
            if not hasattr(ocr_service, 'splitter_agent'):
                raise RuntimeError("OCRService does not have splitter_agent initialized")
            
            print(f"🤖 [PDFProcessor] Asking AI to split {len(temp_images)} pages...", flush=True)
            groups = await ocr_service.splitter_agent.split_mixed_pdf_images(temp_images, questions)
            
            # 3. 根据分组结果生成独立 PDF (Move to thread)
            def _reconstruct_pdfs():
                results = []
                original_doc = fitz.open(pdf_path)
                
                for i, group in enumerate(groups, start=1):
                    student_id = group['student_id']
                    image_paths = group['pages'] 
                    if not image_paths:
                        continue
                    
                    page_indices = []
                    for p_path in image_paths:
                        try:
                            idx = int(Path(p_path).stem.split('-')[-1])
                            page_indices.append(idx)
                        except ValueError:
                            continue
                    
                    if not page_indices:
                        continue
                        
                    page_indices.sort()
                    
                    try:
                        new_doc = fitz.open()
                        for pg_idx in page_indices:
                            if 0 <= pg_idx < len(original_doc):
                                new_doc.insert_pdf(original_doc, from_page=pg_idx, to_page=pg_idx)
                        
                        out_p = output_dir / f"{i}.pdf"
                        new_doc.save(out_p)
                        new_doc.close()
                        
                        results.append({
                            "student_id": student_id,
                            "page_count": len(page_indices),
                            "files": [out_p.name]
                        })
                        print(f"✅ [PDFProcessor] Created {out_p.name} (Student ID: {student_id})", flush=True)
                    except Exception as e:
                        print(f"Error creating PDF for {student_id} (Index {i}): {e}", flush=True)

                original_doc.close()
                return results

            print(f"📦 [PDFProcessor] Reconstructing PDFs...", flush=True)
            final_results = await asyncio.to_thread(_reconstruct_pdfs)

            # 清理临时目录
            shutil.rmtree(temp_dir, ignore_errors=True)
            
            return final_results
            
        except asyncio.CancelledError:
             print("❌ Processing Cancelled (cleanup...)", flush=True)
             if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
             raise
        except Exception as e:
            # 清理临时目录
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
            raise RuntimeError(f"Mixed PDF processing failed: {e}")
