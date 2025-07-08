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

# --- Agent Specific Retrieval Settings ---
# For OutlineGeneratorAgent: Number of documents for initial outline context
DEFAULT_OUTLINE_GENERATION_RETRIEVAL_TOP_N = int(os.getenv("DEFAULT_OUTLINE_GENERATION_RETRIEVAL_TOP_N", "5"))
# For GlobalContentRetrieverAgent: Number of documents per chapter
DEFAULT_GLOBAL_RETRIEVAL_TOP_N_PER_CHAPTER = int(os.getenv("DEFAULT_GLOBAL_RETRIEVAL_TOP_N_PER_CHAPTER", "2")) # Reduced default

# --- Query Generation/Expansion Settings ---
# Max expanded queries from TopicAnalyzerAgent (LLM prompt also guides this)
DEFAULT_MAX_EXPANDED_QUERIES_TOPIC = int(os.getenv("DEFAULT_MAX_EXPANDED_QUERIES_TOPIC", "7"))
# Max heuristic queries for GlobalContentRetrieverAgent per chapter
DEFAULT_MAX_CHAPTER_QUERIES_GLOBAL_RETRIEVAL = int(os.getenv("DEFAULT_MAX_CHAPTER_QUERIES_GLOBAL_RETRIEVAL", "3"))
# Max heuristic queries for ContentRetrieverAgent per chapter
DEFAULT_MAX_CHAPTER_QUERIES_CONTENT_RETRIEVAL = int(os.getenv("DEFAULT_MAX_CHAPTER_QUERIES_CONTENT_RETRIEVAL", "4"))


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
DEFAULT_TOPIC_ANALYZER_PROMPT = """你是一个深度主题分析专家和研究助理。请对以下用户提供的主题进行深入分析和解构。你的任务是：
1.  **泛化主题**：对用户主题进行适当的泛化，以覆盖更广阔的背景。
2.  **提取核心关键词/概念**：识别并列出与主题最相关的中文和英文核心关键词或概念。
3.  **识别核心研究问题**：根据主题，推断出1-3个该主题下可能探讨的核心研究问题。这些问题应该是开放性的，能够引导深入研究。
4.  **推断潜在研究方法/视角**：根据主题，推断出可能适用的1-2种主要研究方法、分析视角或理论框架。
5.  **生成扩展查询建议**：基于以上分析，生成{max_expanded_queries}个相关的、多样化的搜索查询建议，用于后续的深入信息检索。这些查询应该有助于全面覆盖主题的各个方面。

用户主题：'{user_topic}'

请严格按照以下JSON格式返回结果，不要添加任何额外的解释或说明文字：
{{
  "generalized_topic_cn": "泛化后的中文主题",
  "generalized_topic_en": "Generalized English Topic",
  "keywords_cn": ["中文关键词1", "中文关键词2", "中文关键词3"],
  "keywords_en": ["English Keyword1", "English Keyword2", "English Keyword3"],
  "core_research_questions_cn": [
    "核心研究问题1？",
    "核心研究问题2？"
  ],
  "potential_methodologies_cn": [
    "潜在研究方法或视角1",
    "潜在研究方法或视角2"
  ],
  "expanded_queries": [
    "扩展查询1",
    "扩展查询2",
    "扩展查询3",
    "扩展查询4",
    "扩展查询5"
  ]
}}
"""

# --- OutlineGeneratorAgent ---
DEFAULT_OUTLINE_GENERATOR_PROMPT = """你是一个报告大纲撰写助手。请根据以下主题、关键词以及提供的参考资料，生成一份详细的中文报告大纲。
大纲应包含主要章节和子章节（如果适用）。请确保大纲结构清晰、逻辑连贯，并覆盖主题的核心方面。
如果参考资料与主题相关，请尝试在大纲中体现参考资料中的关键信息点，并确保这些章节有据可循。

主题：
{topic_cn} (英文参考: {topic_en})

关键词：
中文: {keywords_cn}
英文: {keywords_en}

参考资料：
---
{retrieved_context}
---

请以Markdown列表格式返回大纲。例如：
- 章节一：介绍
  - 1.1 背景
  - 1.2 研究意义
- 章节二：主要发现 (可能基于参考资料)
  - 2.1 发现点A (来自资料1)
  - 2.2 发现点B (来自资料2)
- 章节三：结论

输出的大纲内容：
"""

# --- ChapterWriterAgent ---
DEFAULT_SINGLE_SNIPPET_WRITER_PROMPT = """你是一位专业的报告撰写员。你的核心任务是根据下方提供的“单一段落参考资料”，以及相关的上下文信息（全局主题、关键术语定义、章节标题），撰写一段相关的中文描述性内容。

报告全局主题：
{report_global_theme}

关键术语定义：
{key_terms_definitions_formatted}

当前章节标题：
{chapter_title}

单一段落参考资料：
\"\"\"
{single_document_snippet}
\"\"\"

**极其重要的撰写要求**：
1.  **绝对忠实原文**：生成的所有内容**必须严格直接基于**提供的“单一段落参考资料”。
2.  **禁止编造与猜测**：**严禁**在参考资料没有明确提及的情况下，自行添加、编造或猜测任何事实、数据、百分比、日期、预测、结论或任何其他类型的信息。
3.  **信息缺失处理**：如果“单一段落参考资料”不足以全面阐述某一要点，或无法回答章节标题所暗示的某些方面，请**直接在生成内容中指出信息不足或未在当前资料中提及**，而不是尝试补充不存在的信息。
4.  **围绕参考资料**：内容应是对参考资料的阐述、总结、解释或基于其直接信息的扩展。
5.  **避免复制**：请用自己的话语进行表述，避免直接、大段地复制粘贴参考资料原文，除非是必要的专有名词或直接引语。
6.  **上下文关联**：确保内容与章节标题和报告全局主题紧密相关，并正确使用“关键术语定义”中提供的术语。
7.  **专业与流畅**：语句表达需专业、通顺、逻辑清晰。
8.  **独立片段**：当前任务是针对此单一参考资料生成描述，后续会有专门步骤进行整合。

撰写的段落内容（纯文本，不需要包含章节标题本身）：
"""

DEFAULT_CHAPTER_INTEGRATION_PROMPT = """你是一位高级报告编辑和内容审查专家。你的核心任务是将以下多个独立生成的“初步文本块”整合并精炼成一篇连贯、流畅、结构清晰且**内容绝对忠实于原始引用**的完整中文章节。每个初步文本块都包含其原始内容和对应的引用溯源信息。

报告全局主题：
{report_global_theme}

关键术语定义：
{key_terms_definitions_formatted}

当前章节标题：
{chapter_title}

待整合的文本块列表（每个文本块后紧跟其引用标记）：
---
{preliminary_content_blocks_formatted}
---
（注意：每个文本块末尾的 `[引用来源：...]` 是其溯源信息，代表了该文本块内容的直接依据。）

**极其重要的整合与审查要求**：
1.  **绝对忠实引用**：整合后的章节内容中，任何非连接性的实质性信息（事实、数据、观点、论述等）都**必须能够明确追溯到**其原始文本块及其附带的引用标记。
2.  **严禁引入新信息**：在整合过程中，**严禁添加任何在原始文本块中未曾出现过的新事实、新数据、新观点或外部知识**。你的工作是基于所给材料的整合、组织和润色，而非内容创作或扩展。
3.  **引用完整性与准确性**：每个 `[引用来源：...]` 标记**必须完整保留**，并且**准确地紧跟与其所支持的文本内容之后**。不得修改、删除这些引用标记，也不得使其与其对应的文本内容分离。
4.  **信息准确传递**：确保每个初步文本块中的核心信息和观点在整合后的章节中得到准确、无失真的体现。
5.  **提升流畅与连贯**：在确保上述前提下，消除重复，优化段落和句子结构，使过渡自然平滑，逻辑清晰。
6.  **主题与术语**：确保整合后的章节内容与报告全局主题、当前章节标题高度一致，并正确使用“关键术语定义”中的术语。
7.  **专业风格**：确保最终输出的章节在语言风格和专业程度上保持一致。

整合后的完整章节内容（纯文本，不需要包含章节标题本身）：
"""

# --- IntroductionGenerator (Special prompt for ChapterWriterAgent when writing introduction) ---
INTRODUCTION_GENERATOR_PROMPT_TEMPLATE = """你是一位专业的报告引言撰写专家。你的任务是为一份关于特定主题的报告撰写一个全面且引人入胜的引言。

报告标题：{report_title}
用户原始主题：{user_topic}

已分析的核心信息：
- 报告主题（泛化后）：{generalized_topic_cn}
- 核心研究问题：
{core_research_questions_formatted}
- 主要概念/讨论对象：
{main_concepts_formatted}
- 潜在研究方法/视角：
{potential_methodologies_formatted}

报告大纲（主要章节标题）：
---
{outline_overview}
---

引言撰写要求：
1.  **背景引入**：简要介绍报告主题的背景和重要性，点明研究的出发点。
2.  **问题提出**：清晰阐述报告旨在探讨或解决的核心研究问题。
3.  **范围与目的**：明确报告的研究范围、主要讨论的概念/对象，以及期望达成的研究目的或价值。
4.  **结构概述**：简要介绍报告的主要章节结构和各部分的核心内容。
5.  **语言要求**：语言应专业、简洁、流畅，能够吸引读者并清晰传达报告的核心议题。
6.  **内容依据**：引言的撰写应严格基于以上提供的核心信息和报告大纲，确保与报告整体内容一致。请不要引入未提及的新概念或问题。
7.  **避免空话套话**：内容应具体、言之有物，避免使用过于宽泛和模板化的表述。

请基于以上信息，撰写引言内容（纯文本）：
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
最后，严格按照以下JSON格式返回修改后的内容，不要在json外添加任何额外的解释或说明文字：

JSON输出格式：
{{
  "refined_content": "修改后的章节内容，确保内容流畅、专业且符合评审要求。",
  "modification_notes": "对修改内容的简要说明，指出主要改动和改进点。"
}}

"""

# ==============================================================================
# 使用示例 (Example of how to use these settings)
# ==============================================================================
# from config.settings import XINFERENCE_API_URL, DEFAULT_LLM_MODEL_NAME
#
# client = Client(XINFERENCE_API_URL)
# model = client.get_model(DEFAULT_LLM_MODEL_NAME)

# --- ChapterWriterAgent - Relevance Check ---
DEFAULT_RELEVANCE_CHECK_PROMPT = """你是一个内容相关性判断助手。
请根据以下提供的章节标题和文档片段，判断此文档片段是否与章节标题紧密相关，并且有助于撰写该章节的内容。

章节标题：
{chapter_title}

文档片段：
---
{document_text}
---

请严格按照以下JSON格式返回你的判断结果，不要添加任何额外的解释或说明文字：
{{
  "is_relevant": <true 或 false>
}}
"""

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
    print(f"DEFAULT_SINGLE_SNIPPET_WRITER_PROMPT (first 50 chars): {DEFAULT_SINGLE_SNIPPET_WRITER_PROMPT[:50]}...")
    print(f"DEFAULT_CHAPTER_INTEGRATION_PROMPT (first 50 chars): {DEFAULT_CHAPTER_INTEGRATION_PROMPT[:50]}...")
    print(f"DEFAULT_EVALUATOR_PROMPT (first 50 chars): {DEFAULT_EVALUATOR_PROMPT[:50]}...")
    print(f"DEFAULT_REFINER_PROMPT (first 50 chars): {DEFAULT_REFINER_PROMPT[:50]}...")
    print(f"DEFAULT_RELEVANCE_CHECK_PROMPT (first 50 chars): {DEFAULT_RELEVANCE_CHECK_PROMPT[:50]}...")


    print("\n--- Agent Specific Retrieval Settings ---")
    print(f"DEFAULT_OUTLINE_GENERATION_RETRIEVAL_TOP_N: {DEFAULT_OUTLINE_GENERATION_RETRIEVAL_TOP_N}")
    print(f"DEFAULT_GLOBAL_RETRIEVAL_TOP_N_PER_CHAPTER: {DEFAULT_GLOBAL_RETRIEVAL_TOP_N_PER_CHAPTER}")

    print("\n--- Query Generation/Expansion Settings ---")
    print(f"DEFAULT_MAX_EXPANDED_QUERIES_TOPIC: {DEFAULT_MAX_EXPANDED_QUERIES_TOPIC}")
    print(f"DEFAULT_MAX_CHAPTER_QUERIES_GLOBAL_RETRIEVAL: {DEFAULT_MAX_CHAPTER_QUERIES_GLOBAL_RETRIEVAL}")
    print(f"DEFAULT_MAX_CHAPTER_QUERIES_CONTENT_RETRIEVAL: {DEFAULT_MAX_CHAPTER_QUERIES_CONTENT_RETRIEVAL}")


# --- OutlineRefinementAgent ---
DEFAULT_OUTLINE_REFINEMENT_PROMPT_CN = """\
你是大纲评审和优化专家。你的任务是审查所提供的报告大纲，并提出具体的改进建议。
报告主题：{topic_description}

当前大纲 (Markdown 格式):
---
{current_outline_md}
---

当前大纲 (解析后的结构及ID):
---
{parsed_outline_json}
---

全局检索信息 (基于章节标题的初步上下文):
---
{global_retrieved_info_summary}
---

请根据当前大纲和全局检索信息，提出优化建议，使大纲在逻辑性、全面性、连贯性和结构性方面更佳。
请关注：
- 检索到的信息是否暗示了缺失的子主题或新的相关章节。
- 是否有章节看起来缺乏足够的支撑信息，可能表明它们过于专门或可以合并。
- 不同章节的信息是否高度重叠，可能表明需要合并或重组。

考虑以下类型的更改：
- 增加新的章节或子章节，尤其是在检索内容暗示信息缺失的地方。
- 删除冗余、不相关或过于细化的章节或子章节。
- 修改章节或子章节的标题，使其更清晰、简洁或更具影响力。
- 重新排序章节或子章节，以获得更好的流程和逻辑进展。
- 合并过于相似或内容重叠的章节。
-拆分过于宽泛或涵盖多个不同主题的章节。
- 调整章节的级别（缩进）以实现正确的层级结构。

约束条件 (如有):
- 最大章节数: {max_chapters}
- 最少章节数: {min_chapters}

请以 JSON 操作列表的形式提供你的建议。每个操作都应该是一个包含 "action" 键和其他必要键的对象。
支持的操作及其格式:
1.  `{{ "action": "add", "title": "新章节标题", "level": <层级编号>, "after_id": "<在此ID之后的章节ID或null>" }}` (如果 after_id 为 null, 则附加到该层级末尾或整个大纲末尾)
2.  `{{ "action": "delete", "id": "<要删除的章节ID>" }}`
3.  `{{ "action": "modify_title", "id": "<要修改的章节ID>", "new_title": "修改后的标题" }}`
4.  `{{ "action": "modify_level", "id": "<要修改的章节ID>", "new_level": <新层级编号> }}`
5.  `{{ "action": "move", "id": "<要移动的章节ID>", "after_id": "<移动到此ID之后的章节ID或null>" }}` (如果 after_id 为 null, 则移动到其层级开头或整个大纲开头)
6.  `{{ "action": "merge", "primary_id": "<合并目标章节ID>", "secondary_id": "<被合并并删除的章节ID>", "new_title_for_primary": "可选的新标题" }}`
7.  `{{ "action": "split", "id": "<要拆分的章节ID>", "new_chapters": [{{ "title": "部分1", "level": <层级编号> }}, {{ "title": "部分2", "level": <层级编号> }}] }}` (ID为'id'的原始章节将被删除, 新章节将获得新ID)

如果不需要优化，请返回一个空的 JSON 列表: `[]`。

建议优化的 JSON 输出:
"""
