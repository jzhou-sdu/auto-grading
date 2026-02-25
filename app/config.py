"""
配置模块 - 从环境变量加载配置
"""

import os
from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import ClassVar, Dict


class Settings(BaseSettings):
    """应用配置"""
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
        "case_sensitive": True
    }
    
    # DMX API (统一接入)
    DMX_API_KEY: str = ""
    DMX_API_BASE: str = "https://www.dmxapi.cn/v1"

    # Domain Knowledge Mapping (领域知识映射)
    # 包含教材引用和领域特定的等价性规则
    DOMAIN_MAPPING: ClassVar[Dict] = {
        # 电动力学
        "电动力学": {
            "textbook": "J.D. Jackson 的《Classical Electrodynamics》",
            "rules": r"""1. **电动力学 (Electrodynamics)**：
   - **单位制**：自动识别并兼容 SI 单位制与高斯（Gaussian/CGS）单位制。例如：$\nabla \cdot \mathbf{{E}} = \rho/\epsilon_0$ (SI) 与 $\nabla \cdot \mathbf{{E}} = 4\pi\rho$ (CGS) 视为等价。
   - **规范变换**：电势 $\phi$ 和磁矢势 $\mathbf{{A}}$ 允许相差一个规范项。"""
        },
        "Electrodynamics": {
            "textbook": "J.D. Jackson 的《Classical Electrodynamics》",
            "rules": r"""1. **电动力学 (Electrodynamics)**：
   - **单位制**：自动识别并兼容 SI 单位制与高斯（Gaussian/CGS）单位制。例如：$\nabla \cdot \mathbf{{E}} = \rho/\epsilon_0$ (SI) 与 $\nabla \cdot \mathbf{{E}} = 4\pi\rho$ (CGS) 视为等价。
   - **规范变换**：电势 $\phi$ 和磁矢势 $\mathbf{{A}}$ 允许相差一个规范项。"""
        },
        # 量子力学
        "量子力学": {
            "textbook": "J.J. Sakurai 的《Modern Quantum Mechanics》",
            "rules": r"""1. **量子力学 (Quantum Mechanics)**：
   - **整体相位**：波函数 $\psi$ 允许相差一个常数复数相位因子 $e^{{i\theta}}$（或 $-1$）。
   - **归一化**：除非题目要求“归一化波函数”，否则只比较函数形式，忽略常数系数差异。
   - **表象变换**：矩阵力学形式与波动力学形式若描述同一物理态，视为等价。"""
        },
        "Quantum Mechanics": {
            "textbook": "J.J. Sakurai 的《Modern Quantum Mechanics》",
            "rules": r"""1. **量子力学 (Quantum Mechanics)**：
   - **整体相位**：波函数 $\psi$ 允许相差一个常数复数相位因子 $e^{{i\theta}}$（或 $-1$）。
   - **归一化**：除非题目要求“归一化波函数”，否则只比较函数形式，忽略常数系数差异。
   - **表象变换**：矩阵力学形式与波动力学形式若描述同一物理态，视为等价。"""
        },
        # 理论力学
        "理论力学": {
            "textbook": "Herbert Goldstein 的《Classical Mechanics》",
            "rules": r"""1. **理论力学 (Classical Mechanics)**：
   - **拉格朗日量**：$L$ 与 $L'$ 若相差一个关于时间的全微分项 $\frac{{d}}{{dt}}f(q,t)$，视为描述同一系统。
   - **广义坐标**：选择不同的广义坐标导致运动方程形式不同，但物理预测一致，视为正确。"""
        },
        "Classical Mechanics": {
            "textbook": "Herbert Goldstein 的《Classical Mechanics》",
            "rules": r"""1. **理论力学 (Classical Mechanics)**：
   - **拉格朗日量**：$L$ 与 $L'$ 若相差一个关于时间的全微分项 $\frac{{d}}{{dt}}f(q,t)$，视为描述同一系统。
   - **广义坐标**：选择不同的广义坐标导致运动方程形式不同，但物理预测一致，视为正确。"""
        },
        # 热统
        "热力学": {
            "textbook": "R.K. Pathria 的《Statistical Mechanics》",
            "rules": r"""1. **热统 (Thermal & Stat. Phys)**：
   - **符号习惯**：兼容不同的配分函数写法（$Z$ vs $Q$）或能量符号（$U$ vs $E$）。
   - **近似条件**：若涉及高温/低温极限，检查泰勒展开阶数是否符合物理精度要求。"""
        },
        "统计力学": {
            "textbook": "R.K. Pathria 的《Statistical Mechanics》",
            "rules": r"""1. **热统 (Thermal & Stat. Phys)**：
   - **符号习惯**：兼容不同的配分函数写法（$Z$ vs $Q$）或能量符号（$U$ vs $E$）。
   - **近似条件**：若涉及高温/低温极限，检查泰勒展开阶数是否符合物理精度要求。"""
        },
        "Thermodynamics": {
            "textbook": "R.K. Pathria 的《Statistical Mechanics》",
            "rules": r"""1. **热统 (Thermal & Stat. Phys)**：
   - **符号习惯**：兼容不同的配分函数写法（$Z$ vs $Q$）或能量符号（$U$ vs $E$）。
   - **近似条件**：若涉及高温/低温极限，检查泰勒展开阶数是否符合物理精度要求。"""
        }
    }

    # Agent Model Configuration
    # 全部切换为 gemini-3-flash-preview
    # 注意：通过 extra_body 传递 thinking 参数以避免 OpenAI SDK 报错
    AGENT_MODEL_CONFIG: dict = {
        # --- 出题服务 (Question Generator) ---
        "analyst": {
            "model": "gemini-3-flash-preview",
            "temperature": 0.0,
            "timeout": 180,
            "max_tokens": 8000,
            "model_kwargs": {"extra_body": {"thinking": {"type": "disabled"}}}
        },
        "ta": {
            "model": "gemini-3-flash-preview-thinking", # 切换为 Thinking 变体
            "temperature": 0.0,
            "timeout": 600,
            "max_tokens": 32000
        },
        "professor": {
            "model": "gemini-3-flash-preview",
            "temperature": 0.0,
            "timeout": 600,
            "max_tokens": 32000,
            "model_kwargs": {"extra_body": {"thinking": {"type": "disabled"}}}
        },
        "designer": {
            "model": "gemini-3-flash-preview",
            "temperature": 0.0,
            "timeout": 180,
            "max_tokens": 8000,
            "model_kwargs": {"extra_body": {"thinking": {"type": "disabled"}}}
        },
        
        # --- 评分服务 (Grading Service) ---
        "teacher": {
            "model": "gemini-3-flash-preview",
            "temperature": 0.0,
            "timeout": 600,
            "max_tokens": 32000,
            "model_kwargs": {"extra_body": {"thinking": {"type": "disabled"}}}
        },
        "student": {
            "model": "gemini-3-flash-preview",
            "temperature": 0.7,
            "timeout": 120,
            "max_tokens": 4000,
            "model_kwargs": {"extra_body": {"thinking": {"type": "disabled"}}}
        },
        "sentinel": {
            "model": "gemini-3-flash-preview",
            "temperature": 0.0,
            "timeout": 60,
            "max_tokens": 2000,
            "model_kwargs": {"extra_body": {"thinking": {"type": "disabled"}}}
        },
        "principal": {
            "model": "gemini-3-flash-preview",
            "temperature": 0.0,
            "timeout": 600,
            "max_tokens": 32000,
            "model_kwargs": {"extra_body": {"thinking": {"type": "disabled"}}}
        },
        "splitter": {
            "model": "glm-4.6v",
            "temperature": 0.0,
            "timeout": 180,
            "max_tokens": 4000
        }
    }

    # OCR Model Configuration
    OCR_MODEL_CONFIG: dict = {
        "glm": {
            "model": "glm-4.6v",
            "temperature": 0.0,
            "timeout": 180,
            "max_tokens": 4000
        },
        "qwen": {
            "model": "qwen-vl-ocr-latest",
            "temperature": 0.0,
            "timeout": 180,
            "max_tokens": 4000
        }
    }
    
    # PDF 处理参数
    PDF_DPI: int = 300
    TOP_REGION_HEIGHT_CM: float = 12.0
    CROP_START_Y_CM: float = 2.5
    
    # OCR 参数
    GLM_THINKING_ENABLED: bool = False
    GLM_THINKING_BUDGET: int = 2048
    
    # 批改参数
    GRADING_MAX_RETRIES: int = 2
    GRADING_TIMEOUT: int = 120
    GRADING_MAX_TOKENS: int = 8000
    
    # 生成参数
    GENERATION_REVIEW_LOOPS: int = 2

    # 评分并发控制与日志
    MAX_CONCURRENT_GRADING_TASKS: int = 3
    GRADING_LOG_DIR: str = "_logs/grading_logs"
    GENERATION_LOG_DIR: str = "_logs/generation_logs"

@lru_cache()
def get_settings() -> Settings:
    """获取配置单例"""
    return Settings()


# 全局配置实例
settings = get_settings()
