import os

# ==============================================================================
# Xinference 服务配置 (Xinference Service Configuration)
# ==============================================================================
XINFERENCE_API_URL = os.getenv("XINFERENCE_API_URL", "http://124.128.251.61:1874") # Xinference API 服务器 URL

# ==============================================================================
# 大语言模型 (LLM) 配置 (Large Language Model Configuration)
# ==============================================================================
DEFAULT_LLM_MODEL_NAME = os.getenv("DEFAULT_LLM_MODEL_NAME", "qwen3") # 默认 LLM 模型名称
DEFAULT_LLM_MAX_TOKENS = int(os.getenv("DEFAULT_LLM_MAX_TOKENS", "14000")) # LLM 生成时最大 token 数量
DEFAULT_LLM_TEMPERATURE = float(os.getenv("DEFAULT_LLM_TEMPERATURE", "0.6")) # LLM 生成温度
DEFAULT_LLM_TOP_P = float(os.getenv("DEFAULT_LLM_TOP_P", "0.95")) # LLM nucleus sampling (top-p) 参数
DEFAULT_LLM_ENABLE_THINKING = os.getenv("DEFAULT_LLM_ENABLE_THINKING", "True").lower() == "true" # 是否启用 LLM 的 "思考" 模式 (如果模型支持)
DEFAULT_LLM_TOP_K = int(os.getenv("DEFAULT_LLM_TOP_K", "20")) # LLM top-k 采样参数
DEFAULT_LLM_MIN_P = float(os.getenv("DEFAULT_LLM_MIN_P", "0")) # LLM min-p 采样参数 (一些模型可能支持)

# ==============================================================================
# 词嵌入模型配置 (Embedding Model Configuration)
# ==============================================================================
DEFAULT_EMBEDDING_MODEL_NAME = os.getenv("DEFAULT_EMBEDDING_MODEL_NAME", "Qwen3-Embedding-0.6B") # 默认词嵌入模型名称

# ==============================================================================
# Reranker 模型配置 (Reranker Model Configuration)
# ==============================================================================
DEFAULT_RERANKER_MODEL_NAME = os.getenv("DEFAULT_RERANKER_MODEL_NAME", "Qwen3-Reranker-0.6B") # 默认 Reranker 模型名称
DEFAULT_RERANKER_BATCH_SIZE = int(os.getenv("DEFAULT_RERANKER_BATCH_SIZE", "2")) # Reranker 处理文档时的批次大小 (调小默认值)
DEFAULT_RERANKER_MAX_TEXT_LENGTH = int(os.getenv("DEFAULT_RERANKER_MAX_TEXT_LENGTH", "512")) # 发送给 Reranker 的文档最大字符长度 (调小默认值)

# ==============================================================================
# 文档处理配置 (Document Processing Configuration)
# ==============================================================================
# --- 通用分块设置 (General Chunking Settings) ---
# (如果未使用父子分块，则为后备设置)
DEFAULT_CHUNK_SIZE = int(os.getenv("DEFAULT_CHUNK_SIZE", "1000")) # 通用分块大小 (字符数)
DEFAULT_CHUNK_OVERLAP = int(os.getenv("DEFAULT_CHUNK_OVERLAP", "100")) # 通用分块重叠大小 (字符数)

# --- 父子分块配置 (Parent-Child Chunking Configuration) ---
# 父块旨在包含更丰富的上下文 (例如段落)
DEFAULT_PARENT_CHUNK_SIZE = int(os.getenv("DEFAULT_PARENT_CHUNK_SIZE", "4000")) # 父块目标字符数
DEFAULT_PARENT_CHUNK_OVERLAP = int(os.getenv("DEFAULT_PARENT_CHUNK_OVERLAP", "200")) # 父块重叠字符数
# 子块旨在包含更小、更集中的片段 (例如句子或少量句子)
DEFAULT_CHILD_CHUNK_SIZE = int(os.getenv("DEFAULT_CHILD_CHUNK_SIZE", "500"))  # 子块目标字符数
DEFAULT_CHILD_CHUNK_OVERLAP = int(os.getenv("DEFAULT_CHILD_CHUNK_OVERLAP", "50"))   # 子块重叠字符数
# 注意: 分隔符可用于更语义化的分块 (例如, "\n\n" 代表段落)。
# 如果使用 NLTK 进行句子切分，这可能不直接用于子块，但可用于父块或作为后备。
# 为简单起见，目前主要依赖大小进行分块。

# --- 支持的文档类型 (Supported Document Types) ---
SUPPORTED_DOC_EXTENSIONS = [".pdf", ".docx", ".txt"] # 支持处理的文档扩展名

# ==============================================================================
# 向量存储配置 (Vector Store Configuration)
# ==============================================================================

DEFAULT_VECTOR_STORE_TOP_K = int(os.getenv("DEFAULT_VECTOR_STORE_TOP_K", "20")) # 向量搜索时检索的文档数量

DEFAULT_VECTOR_STORE_PATH = os.getenv("DEFAULT_VECTOR_STORE_PATH", "/home/ISTIC_0/abms/vector_store") # 向量存储索引文件的默认保存路径

# ==============================================================================
# 混合搜索与检索配置 (Hybrid Search & Retrieval Configuration)
# ==============================================================================
# 用于混合向量搜索和关键字搜索分数的 Alpha 参数。
# Alpha = 1.0 表示纯向量搜索，Alpha = 0.0 表示纯关键字搜索。
DEFAULT_HYBRID_SEARCH_ALPHA = float(os.getenv("DEFAULT_HYBRID_SEARCH_ALPHA", "0.5"))
# 融合前关键字搜索 (BM25) 的 Top K 数量。
DEFAULT_KEYWORD_SEARCH_TOP_K = int(os.getenv("DEFAULT_KEYWORD_SEARCH_TOP_K", "20"))
# RAG检索后，送入LLM生成答案的最终文档数量。
DEFAULT_RETRIEVAL_FINAL_TOP_N = int(os.getenv("DEFAULT_RETRIEVAL_FINAL_TOP_N", "20"))
# 检索结果的最低分数阈值 (例如，基于相似度分数, 0.0 到 1.0)。低于此阈值的文档将被丢弃。
# 注意: FAISS L2距离分数越低越好。BM25 和 Reranker 分数越高越好。
# 此阈值将在 RetrievalService 中应用于归一化后的混合分数或Reranker分数。
DEFAULT_RETRIEVAL_MIN_SCORE_THRESHOLD = float(os.getenv("DEFAULT_RETRIEVAL_MIN_SCORE_THRESHOLD", "0.7"))


# ==============================================================================
# Pipeline (工作流) 配置 (Pipeline Configuration)
# ==============================================================================
DEFAULT_MAX_REFINEMENT_ITERATIONS = int(os.getenv("DEFAULT_MAX_REFINEMENT_ITERATIONS", "5")) # 每个章节内容的最大精炼迭代次数
DEFAULT_PIPELINE_MAX_WORKFLOW_ITERATIONS = int(os.getenv("DEFAULT_PIPELINE_MAX_WORKFLOW_ITERATIONS", "500")) # 工作流最大迭代次数，防止无限循环
DEFAULT_EVALUATOR_REFINEMENT_THRESHOLD = int(os.getenv("DEFAULT_EVALUATOR_REFINEMENT_THRESHOLD", "85")) # Evaluator Agent 评估分数阈值，低于此分数则需要精炼

# ==============================================================================
# 日志配置 (Logging Configuration)
# ==============================================================================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper() # 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)

# ==============================================================================
# Agent 默认 Prompt 模板 (Agent Default Prompt Templates)
# ==============================================================================

# --- TopicAnalyzerAgent ---
DEFAULT_TOPIC_ANALYZER_PROMPT = """你是一个主题分析专家。请分析以下用户提供的主题，对其进行理解、扩展和泛化，并生成相关的中文和英文关键词/主题概念，以便于后续的文档检索。请确保关键词的全面性。

用户主题：'{user_topic}'

请严格按照以下JSON格式返回结果，不要添加任何额外的解释或说明文字：
{{
  "generalized_topic_cn": "泛化后的中文主题",
  "generalized_topic_en": "Generalized English Topic",
  "keywords_cn": ["中文关键词1", "中文关键词2", "中文关键词3"],
  "keywords_en": ["English Keyword1", "English Keyword2", "English Keyword3"]
}}
"""

# --- OutlineGeneratorAgent ---
DEFAULT_OUTLINE_GENERATOR_PROMPT = """你是一个报告大纲撰写助手。请根据以下主题和关键词，生成一份详细的中文报告大纲。
大纲应包含主要章节和子章节（如果适用）。请确保大纲结构清晰、逻辑连贯，并覆盖主题的核心方面。

主题：
{topic_cn} (英文参考: {topic_en})

关键词：
中文: {keywords_cn}
英文: {keywords_en}

请以Markdown列表格式返回大纲。例如：
- 章节一：介绍
  - 1.1 背景
  - 1.2 研究意义
- 章节二：主要发现
  - 2.1 发现点A
  - 2.2 发现点B
- 章节三：结论

输出的大纲内容：
"""

# --- ChapterWriterAgent ---
DEFAULT_CHAPTER_WRITER_PROMPT = """你是一位专业的报告撰写员。请根据以下报告章节标题和相关的参考资料，撰写详细、流畅、专业、连贯的中文章节内容。

章节标题：
{chapter_title}

参考资料：
{retrieved_content_formatted}

撰写要求：
1. 内容需与章节标题紧密相关，并充分、合理地利用提供的参考资料。
2. 避免直接复制粘贴参考资料，而是要理解、整合信息，并用自己的话语有条理地表达出来。
3. 当你引用参考资料中的具体信息时，必须在引用的内容末尾明确注明来源。引用格式为： `[引用来源：【文档名称】 - “【原文片段的开头部分摘要】”，资料编号：【资料编号】]`。
   例如：一项研究表明ABMS系统在复杂战场环境中的关键作用 [引用来源：战术数据链安全增强研究V2.pdf - “ABMS（Advanced Battle Management System）作为美国空军未来作战的核心概念...”，资料编号：some_doc_name-p1] 。
   请确保“原文片段的开头部分摘要”是原文片段最开始的、能够体现核心内容的连续文字，长度约20-30字。
4. 输出的章节内容应具有良好的可读性和专业性。

撰写的章节内容（纯文本，不需要包含章节标题本身）：
"""

# --- EvaluatorAgent ---
DEFAULT_EVALUATOR_PROMPT = """你是一位资深的报告评审员。请根据以下标准评估提供的报告内容：
1.  **相关性**：内容是否紧扣主题和章节要求？信息是否与讨论的核心问题直接相关？
2.  **流畅性**：语句是否通顺自然？段落之间过渡是否平滑？逻辑是否清晰？
3.  **完整性**：信息是否全面？论点是否得到了充分的论证和支持？是否涵盖了应有的关键点？
4.  **准确性**：所陈述的事实、数据和信息是否准确无误？（请基于常识或普遍接受的知识进行判断，除非提供了特定领域的参考标准）

章节标题： {chapter_title}

待评估内容：
---
{content_to_evaluate}
---

请对以上内容进行综合评估，并严格按照以下JSON格式返回你的评分和反馈意见。不要添加任何额外的解释或说明文字。
总评分范围为0-100分。反馈意见应具体指出优点和需要改进的地方。

JSON输出格式：
{{
  "score": <总评分，整数>,
  "feedback_cn": "具体的中文反馈意见，包括优点和改进建议。",
  "evaluation_criteria_met": {{
    "relevance": "<关于相关性的简短评价，例如：高/中/低，具体说明>",
    "fluency": "<关于流畅性的简短评价，例如：优秀/良好/一般/较差，具体说明>",
    "completeness": "<关于完整性的简短评价，例如：非常全面/基本全面/部分缺失/严重缺失，具体说明>",
    "accuracy": "<关于准确性的简短评价（基于常识），例如：高/待核实/部分存疑/低，具体说明>"
    }}
}}
"""

# --- RefinerAgent ---
DEFAULT_REFINER_PROMPT = """你是一位报告修改专家。请根据以下原始内容和评审反馈，对内容进行修改和完善，输出修改后的中文版本。
你的目标是解决反馈中指出的问题，并提升内容的整体质量，包括相关性、流畅性、完整性和准确性。

章节标题：{chapter_title}

原始内容：
---
{original_content}
---

评审反馈：
---
{evaluation_feedback}
---

请仔细阅读评审反馈，理解需要改进的关键点。
在修改时，请尽量保留原始内容的合理部分，重点针对反馈中提出的不足之处进行优化。
如果反馈中包含具体的修改建议，请优先考虑采纳。

修改后的内容（纯文本）：
"""

# ==============================================================================
# 使用示例 (Example of how to use these settings)
# ==============================================================================
# from config.settings import XINFERENCE_API_URL, DEFAULT_LLM_MODEL_NAME
#
# client = Client(XINFERENCE_API_URL)
# model = client.get_model(DEFAULT_LLM_MODEL_NAME)

if __name__ == '__main__':
    print("--- Xinference 服务配置 ---")
    print(f"XINFERENCE_API_URL: {XINFERENCE_API_URL}")

    print("\n--- 大语言模型 (LLM) 配置 ---")
    print(f"DEFAULT_LLM_MODEL_NAME: {DEFAULT_LLM_MODEL_NAME}")
    print(f"DEFAULT_LLM_MAX_TOKENS: {DEFAULT_LLM_MAX_TOKENS}")
    print(f"DEFAULT_LLM_TEMPERATURE: {DEFAULT_LLM_TEMPERATURE}")
    print(f"DEFAULT_LLM_TOP_P: {DEFAULT_LLM_TOP_P}")
    print(f"DEFAULT_LLM_ENABLE_THINKING: {DEFAULT_LLM_ENABLE_THINKING}")
    print(f"DEFAULT_LLM_TOP_K: {DEFAULT_LLM_TOP_K}")
    print(f"DEFAULT_LLM_MIN_P: {DEFAULT_LLM_MIN_P}")

    print("\n--- 词嵌入模型配置 ---")
    print(f"DEFAULT_EMBEDDING_MODEL_NAME: {DEFAULT_EMBEDDING_MODEL_NAME}")

    print("\n--- Reranker 模型配置 ---")
    print(f"DEFAULT_RERANKER_MODEL_NAME: {DEFAULT_RERANKER_MODEL_NAME}")
    print(f"DEFAULT_RERANKER_BATCH_SIZE: {DEFAULT_RERANKER_BATCH_SIZE}")
    print(f"DEFAULT_RERANKER_MAX_TEXT_LENGTH: {DEFAULT_RERANKER_MAX_TEXT_LENGTH}")

    print("\n--- 文档处理配置 ---")
    print(f"DEFAULT_CHUNK_SIZE (通用): {DEFAULT_CHUNK_SIZE}")
    print(f"DEFAULT_CHUNK_OVERLAP (通用): {DEFAULT_CHUNK_OVERLAP}")
    print(f"DEFAULT_PARENT_CHUNK_SIZE: {DEFAULT_PARENT_CHUNK_SIZE}")
    print(f"DEFAULT_PARENT_CHUNK_OVERLAP: {DEFAULT_PARENT_CHUNK_OVERLAP}")
    print(f"DEFAULT_CHILD_CHUNK_SIZE: {DEFAULT_CHILD_CHUNK_SIZE}")
    print(f"DEFAULT_CHILD_CHUNK_OVERLAP: {DEFAULT_CHILD_CHUNK_OVERLAP}")
    print(f"SUPPORTED_DOC_EXTENSIONS: {SUPPORTED_DOC_EXTENSIONS}")

    print("\n--- 向量存储配置 ---")
    print(f"DEFAULT_VECTOR_STORE_TOP_K: {DEFAULT_VECTOR_STORE_TOP_K}")
    print(f"DEFAULT_VECTOR_STORE_PATH: {DEFAULT_VECTOR_STORE_PATH}")

    print("\n--- 混合搜索与检索配置 ---")
    print(f"DEFAULT_HYBRID_SEARCH_ALPHA: {DEFAULT_HYBRID_SEARCH_ALPHA}")
    print(f"DEFAULT_KEYWORD_SEARCH_TOP_K: {DEFAULT_KEYWORD_SEARCH_TOP_K}")
    print(f"DEFAULT_RETRIEVAL_FINAL_TOP_N: {DEFAULT_RETRIEVAL_FINAL_TOP_N}")
    print(f"DEFAULT_RETRIEVAL_MIN_SCORE_THRESHOLD: {DEFAULT_RETRIEVAL_MIN_SCORE_THRESHOLD}")

    print("\n--- Pipeline (工作流) 配置 ---")
    print(f"DEFAULT_MAX_REFINEMENT_ITERATIONS: {DEFAULT_MAX_REFINEMENT_ITERATIONS}")
    print(f"DEFAULT_PIPELINE_MAX_WORKFLOW_ITERATIONS: {DEFAULT_PIPELINE_MAX_WORKFLOW_ITERATIONS}")
    print(f"DEFAULT_EVALUATOR_REFINEMENT_THRESHOLD: {DEFAULT_EVALUATOR_REFINEMENT_THRESHOLD}")

    print("\n--- 日志配置 ---")
    print(f"LOG_LEVEL: {LOG_LEVEL}")

    print("\n--- Agent 默认 Prompt 模板 ---")
    print(f"DEFAULT_TOPIC_ANALYZER_PROMPT (first 50 chars): {DEFAULT_TOPIC_ANALYZER_PROMPT[:50]}...")
    print(f"DEFAULT_OUTLINE_GENERATOR_PROMPT (first 50 chars): {DEFAULT_OUTLINE_GENERATOR_PROMPT[:50]}...")
    print(f"DEFAULT_CHAPTER_WRITER_PROMPT (first 50 chars): {DEFAULT_CHAPTER_WRITER_PROMPT[:50]}...")
    print(f"DEFAULT_EVALUATOR_PROMPT (first 50 chars): {DEFAULT_EVALUATOR_PROMPT[:50]}...")
    print(f"DEFAULT_REFINER_PROMPT (first 50 chars): {DEFAULT_REFINER_PROMPT[:50]}...")
