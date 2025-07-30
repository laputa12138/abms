python main.py \
    --topic "美国空军的情报、监视和侦察（Intelligence, Surveillance, and Reconnaissance,ISR）活动的总体情况调研分析" \
    --data_path "./data/" \
    --output_path "output/ISR-v23-#51-针对超长子块合并问题使用递归处理+修改了超参数+rerank_score=0.6.md" \
    --index_name "abms-V2" \
    --vector_store_path "./my_vector_indexes/"
# python main.py \
#     --topic "美国空军人工智能相关技术现状调研、技术成熟度以及未来发展态势评估" \
#     --data_path "./data/" \
#     --output_path "output/ADTC-v16-#43-修改了LLM json解析部分.md" \
#     --index_name "abms-V2" \
#     --vector_store_path "./my_vector_indexes/" \
#     --log_level DEBUG 
    # --force_reindex