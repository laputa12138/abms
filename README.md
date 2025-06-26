# RAG 多智能体报告生成系统

## 项目概述

本项目是一个基于本地PDF文档集的多智能体报告生成系统。它利用本地部署的嵌入（Embedding）模型和大型语言模型（LLM），通过一系列智能体（Agent）的协作，根据用户指定的主题，自动生成结构化报告。所有模型调用均通过Xinference API进行，确保了本地化处理和数据私密性。

系统的核心功能包括：
1.  **PDF文档处理**：解析用户提供的PDF文档，提取文本内容，并进行分块。
2.  **向量化与存储**：使用本地嵌入模型将文本块转换为向量，并存储在本地FAISS向量数据库中以便快速检索。
3.  **多智能体协作**：
    *   **主题分析智能体 (TopicAnalyzerAgent)**：解析用户输入的主题，进行语义理解、泛化，并生成中英文关键词，以提高后续检索的召回率和准确性。
    *   **大纲生成智能体 (OutlineGeneratorAgent)**：基于分析后的主题和关键词，利用LLM生成报告的初步大纲（中文Markdown格式）。
    *   **内容检索智能体 (ContentRetrieverAgent)**：根据大纲的每个章节和相关关键词，从向量数据库中进行向量搜索，并可选地使用Reranker模型优化排序，检索最相关的内容片段。
    *   **章节撰写智能体 (ChapterWriterAgent)**：为大纲中的每个章节，利用LLM基于检索到的内容片段撰写详细的中文初稿。
    *   **评估智能体 (EvaluatorAgent)**：调用LLM对生成的章节内容进行质量评估，评估维度包括相关性、流畅性、完整性、准确性等，并给出JSON格式的评分和具体的中文反馈。
    *   **精炼智能体 (RefinerAgent)**：根据评估智能体的反馈，利用LLM对生成的文稿进行修改和润色，以提升报告质量。此过程可迭代进行。
    *   **报告编译智能体 (ReportCompilerAgent)**：将所有经过优化和完善的章节内容，按照大纲结构整合成最终的Markdown报告，可选择性添加引言和目录。
4.  **本地化模型部署**：所有核心的AI能力（嵌入、LLM、重排序）均依赖通过Xinference部署的本地模型API。

## 功能特性

*   **本地化处理**：所有文档处理和模型调用均在本地（通过Xinference API）完成。
*   **多文档支持**：能够处理用户提供的多个PDF文档作为知识库。
*   **中英文关键词**：主题分析阶段生成中英文关键词，增强对多语言文档的适应性。
*   **自动化报告流程**：从主题输入到报告输出，大部分流程自动化，包含迭代式的内容优化。
*   **可配置与可扩展**：模型API地址、模型名称、文本分块参数、迭代次数等均可配置；智能体和处理流程易于扩展和修改。
*   **中文报告生成**：专注于生成中文报告，所有提示（Prompts）和LLM交互均优化以确保输出为中文。

## 文件结构说明

```
.
├── agents/                     # 智能体实现
│   ├── __init__.py
│   ├── base_agent.py           # Agent基类
│   ├── chapter_writer_agent.py # 章节撰写Agent
│   ├── content_retriever_agent.py # 内容检索Agent
│   ├── evaluator_agent.py      # 评估Agent
│   ├── outline_generator_agent.py # 大纲生成Agent
│   ├── refiner_agent.py        # 精炼Agent
│   ├── report_compiler_agent.py # 报告编译Agent
│   └── topic_analyzer_agent.py # 主题分析Agent
├── config/                     # 配置文件夹
│   ├── __init__.py
│   └── settings.py             # 项目配置 (API地址, 模型名, 默认参数)
├── core/                       # 核心逻辑模块
│   ├── __init__.py
│   ├── document_processor.py   # PDF处理和文本分块
│   ├── embedding_service.py    # Embedding模型服务接口
│   ├── llm_service.py          # LLM服务接口
│   ├── reranker_service.py     # Reranker模型服务接口
│   └── vector_store.py         # FAISS向量存储与检索
├── data/                       # (Git忽略) 存放用户上传的PDF文档
│   └── .gitkeep                #  (确保目录被Git追踪)
├── output/                     # (Git忽略) 存放生成的报告
│   └── .gitkeep                #  (确保目录被Git追踪)
├── pipelines/                  # 工作流编排
│   ├── __init__.py
│   └── report_generation_pipeline.py # 报告生成主流程
├── utils/                      # (可选) 通用工具函数
│   └── __init__.py
├── .gitignore                  # Git忽略配置
├── main.py                     # 项目主入口脚本 (命令行接口)
├── README.md                   # 项目说明（本文档）
└── requirements.txt            # Python依赖包列表
```

## 环境配置

1.  **Python环境**: 推荐使用 Python 3.8 或更高版本。建议创建虚拟环境：
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    # venv\Scripts\activate   # Windows
    ```

2.  **安装依赖**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Xinference服务**:
    *   确保您的Xinference服务正在运行。
    *   确保已通过Xinference成功部署了所需的LLM、Embedding模型和（可选的）Reranker模型。默认模型名称在 `config/settings.py` 中指定，可以通过命令行参数覆盖：
        *   LLM: `qwen3`
        *   Embedding: `Qwen3-Embedding-0.6B`
        *   Reranker: `Qwen3-Reranker-0.6B`
    *   Xinference API的URL默认为 `http://124.128.251.61:1874`。如果您的服务地址不同，可以通过设置环境变量 `XINFERENCE_API_URL` 或在运行时使用 `--xinference_url` 参数来修改。

## 使用方法

通过 `main.py` 脚本使用命令行参数来运行报告生成系统。

**基本命令格式**:
```bash
python main.py --topic "您的报告主题" --pdfs "路径/到/文档1.pdf,路径/到/文档2.pdf" [其他可选参数]
```

**主要参数说明**:

*   `--topic "主题字符串"` (必需): 您希望生成的报告主题。
*   `--pdfs "pdf路径1,pdf路径2,..."` (必需): 一个或多个PDF文件的路径，以逗号分隔，作为报告的知识来源。
*   `--output_path "输出文件路径.md"` (可选): 生成的Markdown报告的保存路径。
    *   默认: `output/report_YYYYMMDD_HHMMSS.md` （例如 `output/report_20231027_143000.md`）
*   `--report_title "自定义报告标题"` (可选): 为生成的报告指定一个自定义标题。
    *   默认: 根据 `--topic` 自动生成，格式为 `关于“{主题}”的分析报告`。
*   `--xinference_url "URL"` (可选): Xinference API服务器的URL。
    *   默认: `http://124.128.251.61:1874` (来自 `config/settings.py`)
*   `--llm_model "模型名"` (可选): Xinference中部署的LLM模型名称。
    *   默认: `qwen3` (来自 `config/settings.py`)
*   `--embedding_model "模型名"` (可选): Xinference中部署的Embedding模型名称。
    *   默认: `Qwen3-Embedding-0.6B` (来自 `config/settings.py`)
*   `--reranker_model "模型名"` (可选): Xinference中部署的Reranker模型名称。如果不想使用Reranker，可以将其设置为空字符串 `""` 或 `"None"`。
    *   默认: `Qwen3-Reranker-0.6B` (来自 `config/settings.py`)
*   `--max_refinement_iterations <整数>` (可选): 每个章节内容的最大精炼（优化）迭代次数。
    *   默认: `1` (来自 `config/settings.py`)

**示例**:
```bash
python main.py \
    --topic "人工智能在医疗领域的应用与挑战" \
    --pdfs "data/paper1_ai_in_medicine.pdf,data/report_healthcare_ethics.pdf" \
    --output_path "output/AI_医疗报告.md" \
    --report_title "人工智能驱动的医疗革新：应用、挑战与未来展望" \
    --max_refinement_iterations 2
```

此命令将：
1.  以“人工智能在医疗领域的应用与挑战”为主题。
2.  使用 `data/paper1_ai_in_medicine.pdf` 和 `data/report_healthcare_ethics.pdf` 作为知识源。
3.  将生成的报告保存为 `output/AI_医疗报告.md`。
4.  报告的标题将是“人工智能驱动的医疗革新：应用、挑战与未来展望”。
5.  每个章节将进行最多2轮的评估和精炼。
6.  使用默认的Xinference URL和模型。

## 高级配置与定制

*   **修改默认参数**: `config/settings.py` 文件中包含了各类默认参数（如模型名称、文本分块大小、API URL等）。您可以直接修改此文件，或通过设置相应的环境变量来覆盖这些默认值。
*   **调整Agent Prompts**: 每个Agent的核心逻辑很大程度上由其Prompt驱动。您可以在各自的Agent实现文件（位于 `agents/` 目录下）中找到并修改这些Prompt模板，以调整其行为和输出风格。例如，修改 `agents/chapter_writer_agent.py` 中的 `DEFAULT_PROMPT_TEMPLATE` 可以改变章节内容的写作风格。
*   **文本分块策略**: `core/document_processor.py` 中的 `DocumentProcessor` 类负责文本分块。您可以通过修改 `config/settings.py` 中的 `DEFAULT_CHUNK_SIZE` 和 `DEFAULT_CHUNK_OVERLAP` (或相应的环境变量) 来调整分块大小和重叠，以适应不同类型的文档和检索需求。
*   **扩展Agent功能**: 可以通过创建新的Agent类（继承自 `agents/base_agent.py:BaseAgent`）并将其集成到 `pipelines/report_generation_pipeline.py` 中来扩展系统功能。

## 注意事项

*   **模型依赖**: 本系统的性能和输出质量高度依赖于所使用的LLM、Embedding和Reranker模型的质量。
*   **计算资源**: 运行本地模型（尤其是大型LLM）需要相当的计算资源（CPU、内存，对于某些模型可能需要GPU）。请确保您的Xinference部署环境满足这些要求。
*   **PDF质量**: 从扫描版或包含大量复杂布局、图片的PDF中提取文本的效果可能不佳，这会影响后续所有步骤的质量。尽量使用文本可选的、结构清晰的PDF文档。
*   **首次运行**: 首次运行时，FAISS索引的构建和模型的加载可能需要一些时间。

## 未来可能的改进方向

*   更复杂的RAG策略（例如，多查询、HyDE、CoT）。
*   支持更多文档类型（如.docx, .txt, .html）。
*   引入知识图谱增强检索和内容生成。
*   提供Web界面进行交互。
*   更细致的错误处理和用户反馈机制。
*   对生成报告的自动事实校验。
```
