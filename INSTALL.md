# AI 阅卷系统搭建指南

本文档介绍如何在本地环境搭建并运行 AI 阅卷系统。

## 1. 环境要求

*   **操作系统**: Windows / macOS / Linux
*   **Python 版本**: Python 3.12 (推荐3.12，与现有库兼容性更好)

## 2. 快速搭建步骤

### 2.1 获取代码
确保你已经拥有项目代码，并进入项目根目录：
```bash
cd auto-grading
```

### 2.2 安装依赖
使用 pip 安装项目所需的 Python 库：
```bash
pip install -r requirements.txt
```

### 2.3 配置 API 密钥
项目使用 `.env` 文件来管理敏感配置。

1. 在项目根目录下创建一个名为 `.env` 的文件。
2. 将以下内容复制到 `.env` 文件中，并填入你的 API Key：


现在统一从 DMX 调用 API （网址 https://www.dmxapi.cn/）
```
DMX_API_KEY=xxx
```

### 2.4 启动服务
在终端中运行以下命令启动后端服务：

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

启动成功后，你会看到类似以下的输出：
```
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 2.5 访问系统
本机访问，打开浏览器：
[http://localhost:8000]

### 2.6 局域网访问（让其他人访问）
如果你希望同一局域网（校园网）下的其他人访问此网站：

1.  **获取本机 IP 地址**：
    *   **Windows**: 打开终端运行 `ipconfig`，找到 IPv4 地址（例如 `192.168.1.x`）。
    *   **Mac/Linux**: 打开终端运行 `ifconfig` 或 `ip addr` 查看 IP。
2.  **访问地址**：
    其他人可以通过 `http://你的IP地址:8000` 访问。
    例如：`http://192.168.1.5:8000`
3.  **防火墙设置**：如果无法访问，请检查电脑的防火墙设置，确保允许 Python 或 8000 端口的入站连接。

---

## 补充说明：使用 Conda 管理环境 (推荐)

如果你希望环境更加干净，建议使用 Conda (如 Anaconda 或 Miniconda) 来创建独立的 Python 环境。

### 1. 创建环境
创建一个名为 `ai-grading` 的环境，指定 Python 版本为 3.12：
```bash
conda create -n ai-grading python=3.12
```

### 2. 激活环境
```bash
conda activate ai-grading
```

### 3. 后续步骤
激活环境后，按照上文 **2.2 安装依赖** 开始继续操作即可：
```bash
pip install -r requirements.txt

```
