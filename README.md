# RAG 多智能体报告生成系统

## 项目概述

本项目是一个基于本地文档集（支持PDF, DOCX, TXT格式）的多智能体报告生成系统。它利用本地部署的嵌入（Embedding）模型和大型语言模型（LLM），通过一系列智能体（Agent）的协作，根据用户指定的主题，自动生成结构化报告。系统采用**父子分块 (Parent-Child Chunking)**策略处理长文档，并通过**混合检索 (Hybrid Search)**（向量检索 + BM25关键词检索）技术提高召回内容的相关性和上下文丰富度。所有模型调用均通过Xinference API进行，确保了本地化处理和数据私密性。

系统的核心功能包括：
1.  **多格式文档处理与父子分块**：
    *   解析用户指定文件夹内的 `.pdf`, `.docx`, `.txt` 文档。
    *   对提取的文本内容进行父子分块：父块较大，提供丰富上下文；子块较小，用于精确向量检索。
2.  **向量化与存储**：使用本地嵌入模型将 **子块** 转换为向量，并存储在本地FAISS向量数据库中。同时保存子块与父块的关联信息。
3.  **多智能体协作**：
    *   **主题分析智能体 (TopicAnalyzerAgent)**：解析用户输入的主题，进行语义理解、泛化，并生成中英文关键词。
    *   **大纲生成智能体 (OutlineGeneratorAgent)**：基于分析后的主题和关键词，利用LLM生成报告的初步大纲（中文Markdown格式）。
    *   **内容检索智能体 (ContentRetrieverAgent)**：
        *   执行混合检索：结合基于子块的向量搜索和基于子块的BM25关键词搜索。
        *   通过可配置的权重参数 (`alpha`) 平衡两种检索方式的结果。
        *   检索到相关的子块后，提取其对应的 **父块** 作为上下文。
        *   可选地使用Reranker模型对检索到的父块列表进行重排序。
    *   **章节撰写智能体 (ChapterWriterAgent)**：为大纲中的每个章节，利用LLM基于检索到的父块内容撰写详细的中文初稿。
    *   **评估智能体 (EvaluatorAgent)**：调用LLM对生成的章节内容进行质量评估（相关性、流畅性、完整性、准确性），返回JSON格式的评分和中文反馈。
    *   **精炼智能体 (RefinerAgent)**：根据评估反馈，利用LLM对文稿进行修改和润色。此过程可迭代进行。
    *   **报告编译智能体 (ReportCompilerAgent)**：将所有优化后的章节内容，按大纲结构整合成最终的Markdown报告，可选添加引言和目录。
4.  **本地化模型部署**：所有核心AI能力均依赖通过Xinference部署的本地模型API。

## 功能特性

*   **本地化处理**：所有文档处理和模型调用均在本地（通过Xinference API）完成。
*   **多文档类型支持**：自动扫描指定文件夹，处理 `.pdf`, `.docx`, `.txt` 文件。
*   **父子分块**：优化长文档处理，通过子块精确检索，父块补充上下文。
*   **混合检索**：结合向量检索的语义相似性和关键词检索的字面匹配能力，提高检索质量。
*   **中英文关键词**：主题分析阶段生成中英文关键词，增强检索适应性。
*   **自动化报告流程**：从主题输入到报告输出，大部分流程自动化，包含迭代式内容优化。
*   **高度可配置**：模型API、模型名称、分块参数、检索参数、迭代次数等均可通过配置文件或命令行参数进行调整。
*   **中文报告生成**：专注于生成中文报告，所有提示（Prompts）和LLM交互均优化以确保输出为中文。

## 文件结构说明

```
.
├── agents/                     # 智能体实现 (各Agent具体功能如上所述)
│   ├── __init__.py
│   ├── base_agent.py
│   ├── chapter_writer_agent.py
│   ├── content_retriever_agent.py # 已更新以支持混合检索和父子块
│   ├── evaluator_agent.py
│   ├── outline_generator_agent.py
│   ├── refiner_agent.py
│   ├── report_compiler_agent.py
│   └── topic_analyzer_agent.py
├── config/
│   ├── __init__.py
│   └── settings.py             # 项目配置 (API, 模型名, 分块/检索默认参数等)
├── core/
│   ├── __init__.py
│   ├── document_processor.py   # 已更新以支持多文档类型和父子分块
│   ├── embedding_service.py
│   ├── llm_service.py
│   ├── reranker_service.py
│   └── vector_store.py         # 已更新以支持存储父子块信息
├── data/                       # (Git忽略) 存放用户上传的源文档
│   └── .gitkeep
├── output/                     # (Git忽略) 存放生成的报告
│   └── .gitkeep
├── pipelines/
│   ├── __init__.py
│   └── report_generation_pipeline.py # 已更新以集成新功能流程
├── utils/                      # (可选) 通用工具函数
│   └── __init__.py
├── .gitignore
├── main.py                     # 项目主入口 (命令行接口, 已更新参数)
├── README.md                   # 项目说明（本文档）
└── requirements.txt            # Python依赖 (已更新)
```

## 环境配置

1.  **Python环境**: 推荐 Python 3.8+。建议使用虚拟环境。
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    # venv\Scripts\activate   # Windows
    ```

2.  **安装依赖**:
    ```bash
    pip install -r requirements.txt
    ```
    (如果遇到 `nltk` 相关资源下载问题，可能需要手动在Python环境中执行 `import nltk; nltk.download('punkt')` 来下载必要的分词模型。)

3.  **Xinference服务**:
    *   确保Xinference服务正在运行，并且已成功部署所需的LLM、Embedding和（可选的）Reranker模型。
    *   默认模型名称和Xinference API URL在 `config/settings.py` 中定义，可通过命令行参数覆盖。

## 使用方法

通过 `main.py` 脚本和命令行参数运行系统。

**基本命令格式**:
```bash
python main.py --topic "您的报告主题" --data_path "存放源文档的文件夹路径" [其他可选参数]
```

**主要参数说明**:

*   `--topic "主题字符串"` (必需): 您希望生成的报告主题。
*   `--data_path "文件夹路径"` (可选): 存放源文档的文件夹路径。系统将扫描此路径下所有支持的文档（PDF, DOCX, TXT）。
    *   默认: `./data/`
*   `--output_path "输出文件路径.md"` (可选): 生成的Markdown报告的保存路径。
    *   默认: `output/report_YYYYMMDD_HHMMSS.md`
*   `--report_title "自定义报告标题"` (可选): 为报告指定自定义标题。
    *   默认: 根据 `--topic` 自动生成。

**Xinference与模型配置 (分组: Xinference and Model Configuration)**:
*   `--xinference_url "URL"`: Xinference API服务器URL。 (默认: `config.settings.XINFERENCE_API_URL`)
*   `--llm_model "模型名"`: LLM模型名称。 (默认: `config.settings.DEFAULT_LLM_MODEL_NAME`)
*   `--embedding_model "模型名"`: Embedding模型名称。 (默认: `config.settings.DEFAULT_EMBEDDING_MODEL_NAME`)
*   `--reranker_model "模型名"`: Reranker模型名称。设为 'None' 或空字符串禁用。 (默认: `config.settings.DEFAULT_RERANKER_MODEL_NAME`)

**文档处理 - 分块参数 (分组: Document Processing - Chunking Parameters)**:
*   `--parent_chunk_size <整数>`: 父块目标字符数。 (默认: `config.settings.DEFAULT_PARENT_CHUNK_SIZE`)
*   `--parent_chunk_overlap <整数>`: 父块重叠字符数。 (默认: `config.settings.DEFAULT_PARENT_CHUNK_OVERLAP`)
*   `--child_chunk_size <整数>`: 子块目标字符数。 (默认: `config.settings.DEFAULT_CHILD_CHUNK_SIZE`)
*   `--child_chunk_overlap <整数>`: 子块重叠字符数。 (默认: `config.settings.DEFAULT_CHILD_CHUNK_OVERLAP`)

**检索参数 (分组: Retrieval Parameters)**:
*   `--vector_top_k <整数>`: 向量搜索召回的文档数。 (默认: `config.settings.DEFAULT_VECTOR_STORE_TOP_K`)
*   `--keyword_top_k <整数>`: 关键词搜索(BM25)召回的文档数。 (默认: `config.settings.DEFAULT_KEYWORD_SEARCH_TOP_K`)
*   `--hybrid_search_alpha <浮点数>`: 混合搜索的权重因子 (0.0 纯关键词, 1.0 纯向量)。 (默认: `config.settings.DEFAULT_HYBRID_SEARCH_ALPHA`)
*   `--final_top_n_retrieval <整数>`: 检索和重排序后，最终用于章节生成的文档数。 (默认: 等同于 `vector_top_k`)

**流程执行参数 (分组: Pipeline Execution Parameters)**:
*   `--max_refinement_iterations <整数>`: 每个章节内容的最大精炼迭代次数。 (默认: `config.settings.DEFAULT_MAX_REFINEMENT_ITERATIONS`)

**示例**:
```bash
python main.py \
    --topic "AI在教育领域的创新应用与伦理考量" \
    --data_path "./my_research_papers/" \
    --output_path "output/AI_教育报告_v2.md" \
    --report_title "人工智能赋能教育：创新、实践与伦理边界" \
    --parent_chunk_size 2000 \
    --child_chunk_size 400 \
    --hybrid_search_alpha 0.6 \
    --max_refinement_iterations 1
```
此命令将：
1.  以“AI在教育领域的创新应用与伦理考量”为主题。
2.  从 `./my_research_papers/` 文件夹中读取所有支持的文档。
3.  生成的报告保存为 `output/AI_教育报告_v2.md`，标题为指定内容。
4.  父块大小设置为2000字符，子块400字符。
5.  混合检索时，向量检索权重0.6，关键词检索权重0.4。
6.  每个章节进行1轮评估和精炼。
7.  使用默认的Xinference URL和模型（除非在`settings.py`中修改或通过环境变量设置）。

## 高级配置与定制

*   **修改默认参数**: `config/settings.py` 文件包含了各类默认参数。您可直接修改此文件，或通过设置相应的环境变量，或通过命令行参数覆盖。
*   **调整Agent Prompts**: 各Agent的Prompt位于其各自的实现文件中（`agents/`目录）。修改这些Prompt可以调整Agent的行为和输出风格。
*   **文本分块与检索策略**: `DocumentProcessor`（父子分块逻辑）、`VectorStore`（向量存储）和 `ContentRetrieverAgent`（混合检索逻辑）是核心。您可以调整这些模块的内部实现或参数来改变数据处理和检索方式。

## 注意事项

*   **模型依赖**: 系统性能高度依赖于所用AI模型的质量。
*   **计算资源**: 本地运行模型（尤其是LLM）需要充足的计算资源。
*   **文档质量**: 源文档的文本质量直接影响提取和分块效果。扫描版PDF或复杂布局文档可能效果不佳。
*   **NLTK资源**: 首次运行时，如果`nltk`的`punkt`分词模型未下载，程序会尝试自动下载。如自动下载失败，请参照“环境配置”部分手动下载。
*   **BM25分词**: 当前BM25的关键词检索使用简单的空格分词。对于中文等需要专业分词的语言，效果可能有限。未来可集成如 `jieba` 等中文分词库来提升关键词检索的准确性。

## 未来可能的改进方向

*   集成更专业的中文分词库 (如 `jieba`) 用于BM25。
*   实现更复杂的RAG策略（如多查询、HyDE）。
*   提供Web界面。
*   对生成报告的自动事实校验和引用生成。
```
