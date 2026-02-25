# AI Grading System - Copilot Instructions

## Project Overview
Python 3.12+ system for automated exam grading using Multi-Agent Systems (DeepSeek, Gemini, etc.) and a FastAPI backend.
- **Root**: `c:/0/_test/ai-grading`
- **Web App**: [app/](app/)
- **CLI Tools**: [_file/code/](_file/code/)

## Critical Patterns & Architecture

### 1. Adversarial Grading Agent Chain
The grading logic in [app/services/grading_service.py](app/services/grading_service.py) follows a strict 4-step debate protocol:
1.  **Sentinel**: Security scan. Prevents prompt injection.
2.  **Teacher**: Initial grading.
3.  **Student**: Generates appeals/rebuttals (simulates a student arguing for points).
4.  **Principal**: Final binding arbitration.

**Rule**: When modifying grading logic, preserve this chain. Do not merge roles.

### 2. LLM Output Handling (Critical)
LLMs (specifically DeepSeek/Reasoning models) output mixed Markdown (reasoning) and JSON.
- **NEVER** use `json.loads` directly on model output.
- **ALWAYS** use `parse_mixed_output(text)` from [app/services/grading_service.py](app/services/grading_service.py) to extract structured data.
- **ALWAYS** use `fix_json_latex(text)` before parsing to fix LaTeX backslash escaping in JSON strings.

### 3. Security & Sanitization
- **Input**: All user/student text must pass through `sanitize_student_input()` in [app/services/grading_service.py](app/services/grading_service.py) to strip injection markers like `<|im_start|>` or internal flags `__SYSTEM_OCR_FAILED_CRITICAL__`.
- **Injection**: The Sentinel agent must run *before* any grading agent sees the text.

### 4. Domain Knowledge Injection
- Science domains (Physics/Math) require specific textbook contexts (e.g., SI vs CGS units).
- **Source**: [app/config.py](app/config.py) -> `DOMAIN_MAPPING`.
- **Math Comp**: Use `check_math_equivalence` (SymPy-based) in [app/services/grading_service.py](app/services/grading_service.py) instead of string matching for formulas.

## Developer Workflows

### Testing & Verification
- **Quick Logic Test**: `python _file/local_test_grading.py` (Mocks API, fast iteration).
- **Full Grading Batch**: `python _file/code/2-grading.py` with arguments.
- **Web Server**: `uvicorn app.main:app --reload`.

### Key Directories
- `app/services/`: Core business logic (Agents, OCR, PDF).
- `_file/code/pdf-ocr/`: OCR pipeline scripts (PDF -> Images -> Markdown).
- `_logs/`: Execution logs (invaluable for debugging multi-agent conversations).

## Coding Standards
- **Models**: Use Pydantic (`BaseModel`) for all agent outputs and internal structures.
- **Parsing**: Handle "Thinking Chain" content carefully—it usually precedes the JSON block.
- **Configuration**: All API Keys/URLs go through `app.config.Settings`.
