# 🚀 Evalscope Toolkit

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![GitHub](https://img.shields.io/badge/GitHub-evalscope--toolkit-blue?logo=github)](https://github.com/maxuan1798/evalscope_toolkit)

一个简化的、模块化的大语言模型评估工具包，基于 vLLM 和 Evalscope 构建。

## ✨ 特性

- 🎯 **简单易用** - 只需配置模型和数据集即可开始评估
- 📦 **模块化设计** - 清晰的代码结构，易于维护和扩展
- 🔄 **自动化流程** - 自动管理数据集、vLLM 服务和评估流程
- 📊 **多数据集支持** - 内置 10+ 种标准评估数据集
- 🎨 **多种使用方式** - Notebook、Python 脚本、命令行均可
- 🛡️ **健壮可靠** - 完善的错误处理和日志记录

## 🎯 支持的数据集

| 数据集 | 类型 | 描述 |
|--------|------|------|
| `gsm8k` | 数学推理 | 小学数学问题 |
| `humaneval` | 代码生成 | Python 代码生成 |
| `mmlu` | 通识知识 | 多任务语言理解 |
| `competition_math` | 数学 | 竞赛级数学问题 |
| `drop` | 阅读理解 | 离散推理 |
| `hellaswag` | 常识推理 | 句子补全 |
| `arc` | 科学 | AI2 推理挑战 |
| `truthfulqa` | 真实性 | 真实性问答 |
| `winogrande` | 常识 | Winograd 模式 |
| `math_500` | 数学 | 数学问题集 |

## 📦 快速开始

### 方式 1: Notebook（推荐）

```bash
git clone https://github.com/maxuan1798/evalscope_toolkit.git
cd evalscope_toolkit
jupyter notebook eval.ipynb
```

在 Notebook 中直接运行评估代码。

### 方式 2: Python 脚本

```python
from evalscope_toolkit import EvalConfig, Evaluator

# 配置评估
config = EvalConfig(
    models=["unsloth/Llama-3.2-3B-Instruct"],
    datasets=["gsm8k", "humaneval"],
    gpus="0",
    gpu_memory_utilization=0.6
)

# 运行评估
evaluator = Evaluator(config)
results = evaluator.run()
```

### 方式 3: 命令行

```bash
python -m evalscope_toolkit.cli \
    --models "model1,model2" \
    --datasets "gsm8k,humaneval" \
    --gpus "0,1"
```

## 📦 项目结构

```
evalscope_toolkit/
├── evalscope_toolkit/              # 核心 Python 包
│   ├── __init__.py                # 包初始化，导出主要类
│   ├── config.py                  # EvalConfig - 配置管理
│   ├── dataset_manager.py         # DatasetManager - 数据集管理
│   ├── vllm_service.py            # VLLMService - vLLM 服务管理
│   ├── evaluator.py               # Evaluator - 评估编排
│   ├── utils.py                   # 工具函数（依赖安装等）
│   └── cli.py                     # 命令行接口
│
├── eval.ipynb                     # 原版完整 Notebook
├── setup.py                       # Python 包安装配置
├── README.md                      # 项目文档
└── .gitignore                     # Git 忽略文件
```

## 🔧 安装依赖

### 系统要求
- Python 3.9+
- CUDA 11.8+ (GPU 推理)
- 8GB+ GPU 内存（小模型）

### 依赖安装
```bash
pip install torch vllm evalscope modelscope datasets
```

或从源码安装：
```bash
git clone https://github.com/maxuan1798/evalscope_toolkit.git
cd evalscope_toolkit
pip install -e .
```

## 📊 核心架构

```
1. 配置创建 (EvalConfig)
   ↓
2. 数据集准备 (DatasetManager)
   - 下载数据集
   - 验证缓存
   ↓
3. 模型评估循环
   ├─ 启动 vLLM 服务 (VLLMService)
   ├─ 数据集评估循环
   │  ├─ 调用 evalscope 评估
   │  └─ 记录结果和时间
   └─ 停止 vLLM 服务
   ↓
4. 生成报告
   - evaluation_summary.json
   - overall_evaluation_times.log
```

## 📝 配置示例

### 基础配置
```python
config = EvalConfig(
    models=["unsloth/Llama-3.2-3B-Instruct"],
    datasets=["gsm8k"],
)
```

### 高级配置
```python
config = EvalConfig(
    # 模型和数据集
    models=["model1", "model2", "/path/to/local/model"],
    datasets=["gsm8k", "humaneval", "mmlu"],
    
    # GPU 配置
    gpus="0,1",                      # 使用的 GPU
    tensor_parallel_size=2,          # 张量并行大小
    gpu_memory_utilization=0.6,      # GPU 内存利用率
    
    # 评估参数
    eval_batch_size=32,              # 批次大小
    max_new_tokens=2048,             # 最大生成 token 数
    temperature=0.0,                 # 采样温度
    
    # 路径配置
    data_root=Path("./data"),        # 数据缓存目录
    log_root=Path("./log"),          # 日志目录
)
```

## 🎁 主要改进

### 相比原版 Notebook

| 特性 | 原版 | Evalscope Toolkit |
|------|------|-------------------|
| 代码结构 | 500+ 行单文件 | 模块化 7 个文件 |
| 可复用性 | ❌ | ✅ 可作为包导入 |
| 配置管理 | 分散在各处 | 统一配置类 |
| 错误处理 | 基础 | 完善的异常处理 |
| 命令行支持 | ❌ | ✅ |
| 文档 | 基础 | 完整文档 + 示例 |
| 超时策略 | 固定 300s | 智能超时 (3-10min) |

### 核心优化

✅ **简化的启动流程** - 智能超时策略，远程模型 10 分钟，本地模型 3 分钟  
✅ **更好的错误提示** - 失败时显示关键日志，快速定位问题  
✅ **自动化程度高** - 自动下载数据集、管理服务、生成报告  
✅ **清晰的进度反馈** - 实时显示评估进度和剩余时间  

## 📂 输出结果

评估结果保存在 `log/outputs_<用户>_<实例ID>/` 目录：

```
log/outputs_user_111412531924263/
├── evaluation_summary.json          # 评估摘要
├── overall_evaluation_times.log     # 总体时间日志
├── vllm_<hash>.log                  # vLLM 服务日志
└── model-name/
    ├── evaluation_times.log         # 模型评估时间
    ├── gsm8k/
    │   ├── result.json              # 评估结果
    │   └── predictions.jsonl        # 预测结果
    └── humaneval/
        └── ...
```

## 🛠️ 常见问题

### 端口被占用
程序会自动寻找可用端口，无需手动处理。

### GPU 内存不足
降低 `gpu_memory_utilization` 值：
```python
config = EvalConfig(
    gpu_memory_utilization=0.4,  # 降低到 40%
)
```

### 模型下载慢
使用本地模型或设置 HuggingFace 镜像：
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### vLLM 启动失败
查看日志文件 `vllm_<hash>.log`，失败时会自动显示最后 50 行。



## 🎯 核心模块说明

| 模块 | 类 | 功能 |
|------|-----|------|
| `config.py` | `EvalConfig` | 统一配置管理，环境变量设置 |
| `dataset_manager.py` | `DatasetManager` | 数据集下载、缓存和验证 |
| `vllm_service.py` | `VLLMService` | vLLM 服务生命周期管理 |
| `evaluator.py` | `Evaluator` | 评估流程编排和执行 |
| `utils.py` | - | 依赖检查、安装等工具函数 |
| `cli.py` | - | 命令行接口 |

## 🚀 使用示例

### 示例 1: 单模型单数据集
```python
from evalscope_toolkit import EvalConfig, Evaluator

config = EvalConfig(
    models=["unsloth/Llama-3.2-3B-Instruct"],
    datasets=["gsm8k"],
)

evaluator = Evaluator(config)
results = evaluator.run()
```

### 示例 2: 多模型多数据集
```python
config = EvalConfig(
    models=[
        "unsloth/Llama-3.2-3B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
    ],
    datasets=["gsm8k", "humaneval", "mmlu"],
    gpus="0,1",
)

evaluator = Evaluator(config)
results = evaluator.run()
```

## 🌟 最佳实践

### GPU 内存管理
```python
# 小模型 (<7B)
config = EvalConfig(gpu_memory_utilization=0.8)

# 中等模型 (7B-13B)  
config = EvalConfig(gpu_memory_utilization=0.6)

# 大模型 (>13B)
config = EvalConfig(gpu_memory_utilization=0.4)
```

### 多 GPU 配置
```python
# 单模型多 GPU (张量并行)
config = EvalConfig(gpus="0,1", tensor_parallel_size=2)

# 单 GPU
config = EvalConfig(gpus="0")
```

## 🎯 未来规划

- [ ] 支持更多评估数据集
- [ ] 添加自定义数据集支持
- [ ] 并行评估多个模型
- [ ] 结果可视化和对比分析
- [ ] Web UI 界面
- [ ] Docker 容器化

## 📞 获取帮助

- **GitHub Issues**: [报告问题或功能请求](https://github.com/maxuan1798/evalscope_toolkit/issues)
- **文档**: 查看 README.md
- **示例**: `eval.ipynb`

## 🤝 贡献

欢迎贡献代码、报告问题或提出建议！

```bash
git clone https://github.com/maxuan1798/evalscope_toolkit.git
cd evalscope_toolkit
pip install -e .
```

## 📄 许可证

本项目采用 MIT 许可证。

## 🙏 致谢

- [vLLM](https://github.com/vllm-project/vllm) - 高性能 LLM 推理引擎
- [Evalscope](https://github.com/modelscope/evalscope) - 模型评估框架
- [ModelScope](https://modelscope.cn/) - 模型和数据集平台

---

**版本**: 1.0.0  
**更新时间**: 2025-11-14  
**维护者**: [@maxuan1798](https://github.com/maxuan1798)

如果这个项目对您有帮助，请给个 ⭐️ Star！


