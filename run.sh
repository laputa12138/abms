python main.py \
    --topic "美国空军的情报、监视和侦察（Intelligence, Surveillance, and Reconnaissance,ISR）活动的总体情况调研分析" \
    --data_path "./data/" \
    --output_path "output/ISR-v18-#45-修改了多轮检索策略.md" \
    --index_name "abms-V2" \
    --vector_store_path "./my_vector_indexes/" \
    --log_level DEBUG 
# python main.py \
#     --topic "美国空军人工智能相关技术现状调研、技术成熟度以及未来发展态势评估" \
#     --data_path "./data/" \
#     --output_path "output/ADTC-v16-#43-修改了LLM json解析部分.md" \
#     --index_name "abms-V2" \
#     --vector_store_path "./my_vector_indexes/" \
#     --log_level DEBUG 
    # --force_reindex