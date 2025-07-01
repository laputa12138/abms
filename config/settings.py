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
DEFAULT_RERANKER_BATCH_SIZE = int(os.getenv("DEFAULT_RERANKER_BATCH_SIZE", "1")) # Reranker 处理文档时的批次大小
DEFAULT_RERANKER_MAX_TEXT_LENGTH = int(os.getenv("DEFAULT_RERANKER_MAX_TEXT_LENGTH", "5000")) # 发送给 Reranker 的文档最大字符长度 (超长会截断)

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
DEFAULT_VECTOR_STORE_TOP_K = int(os.getenv("DEFAULT_VECTOR_STORE_TOP_K", "10")) # 向量搜索时检索的文档数量
DEFAULT_VECTOR_STORE_PATH = os.getenv("DEFAULT_VECTOR_STORE_PATH", "/home/ISTIC_0/abms/vector_store") # 向量存储索引文件的默认保存路径

# ==============================================================================
# 混合搜索与检索配置 (Hybrid Search & Retrieval Configuration)
# ==============================================================================
# 用于混合向量搜索和关键字搜索分数的 Alpha 参数。
# Alpha = 1.0 表示纯向量搜索，Alpha = 0.0 表示纯关键字搜索。
DEFAULT_HYBRID_SEARCH_ALPHA = float(os.getenv("DEFAULT_HYBRID_SEARCH_ALPHA", "0.5"))
# 融合前关键字搜索 (BM25) 的 Top K 数量。
DEFAULT_KEYWORD_SEARCH_TOP_K = int(os.getenv("DEFAULT_KEYWORD_SEARCH_TOP_K", "10"))
# RAG检索后，送入LLM生成答案的最终文档数量。
DEFAULT_RETRIEVAL_FINAL_TOP_N = int(os.getenv("DEFAULT_RETRIEVAL_FINAL_TOP_N", "7"))
# 检索结果的最低分数阈值 (例如，基于相似度分数, 0.0 到 1.0)。低于此阈值的文档将被丢弃。
# 注意: FAISS L2距离分数越低越好。BM25 和 Reranker 分数越高越好。
# 此阈值将在 RetrievalService 中应用于归一化后的混合分数或Reranker分数。
DEFAULT_RETRIEVAL_MIN_SCORE_THRESHOLD = float(os.getenv("DEFAULT_RETRIEVAL_MIN_SCORE_THRESHOLD", "0.2"))


# ==============================================================================
# Pipeline (工作流) 配置 (Pipeline Configuration)
# ==============================================================================
DEFAULT_MAX_REFINEMENT_ITERATIONS = int(os.getenv("DEFAULT_MAX_REFINEMENT_ITERATIONS", "1")) # 每个章节内容的最大精炼迭代次数
DEFAULT_PIPELINE_MAX_WORKFLOW_ITERATIONS = int(os.getenv("DEFAULT_PIPELINE_MAX_WORKFLOW_ITERATIONS", "300")) # 工作流最大迭代次数，防止无限循环
DEFAULT_EVALUATOR_REFINEMENT_THRESHOLD = int(os.getenv("DEFAULT_EVALUATOR_REFINEMENT_THRESHOLD", "80")) # Evaluator Agent 评估分数阈值，低于此分数则需要精炼

# ==============================================================================
# 日志配置 (Logging Configuration)
# ==============================================================================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper() # 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)


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
