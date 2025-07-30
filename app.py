import gradio as gr
import subprocess
import threading
import time
import sys
from main import main as run_main

def run_script(topic, data_path, output_path, report_title, xinference_url, llm_model, embedding_model, reranker_model, parent_chunk_size, parent_chunk_overlap, child_chunk_size, child_chunk_overlap, vector_top_k, keyword_top_k, final_top_n_retrieval, max_refinement_iterations, max_workflow_iterations, vector_store_path, index_name, force_reindex, log_level, debug, log_path):

    # Construct the command
    command = [
        "python", "main.py",
        "--topic", topic,
        "--data_path", data_path,
        "--output_path", output_path,
        "--report_title", report_title,
        "--xinference_url", xinference_url,
        "--llm_model", llm_model,
        "--embedding_model", embedding_model,
        "--reranker_model", reranker_model,
        "--parent_chunk_size", str(parent_chunk_size),
        "--parent_chunk_overlap", str(parent_chunk_overlap),
        "--child_chunk_size", str(child_chunk_size),
        "--child_chunk_overlap", str(child_chunk_overlap),
        "--vector_top_k", str(vector_top_k),
        "--keyword_top_k", str(keyword_top_k),
        "--final_top_n_retrieval", str(final_top_n_retrieval),
        "--max_refinement_iterations", str(max_refinement_iterations),
        "--max_workflow_iterations", str(max_workflow_iterations),
        "--vector_store_path", vector_store_path,
        "--index_name", index_name,
        "--log_level", log_level,
        "--log_path", log_path
    ]
    if force_reindex:
        command.append("--force_reindex")
    if debug:
        command.append("--debug")

    # Start the process
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True
    )

    # Stream output
    log_output = ""
    workflow_status = "Starting..."
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            log_output += output
            if output.startswith("[WF_LOG]"):
                workflow_status = output.strip()
            yield log_output, workflow_status

    # Final status
    if process.poll() == 0:
        yield log_output, "Completed"
    else:
        yield log_output, f"Error (code {process.poll()})"


from config import settings

with gr.Blocks() as iface:
    gr.Markdown("# RAG Multi-Agent Report Generation System")

    with gr.Tab("Run Report"):
        with gr.Row():
            with gr.Column(scale=1):
                topic = gr.Textbox(label="Topic", value="美国空军的情报、监视和侦察（Intelligence, Surveillance, and Reconnaissance,ISR）活动的总体情况调研分析")
                data_path = gr.Textbox(label="Data Path", value="./data/")
                output_path = gr.Textbox(label="Output Path", value="output/ISR-v24-#55-webui.md")
                report_title = gr.Textbox(label="Report Title", placeholder="Optional")

                with gr.Accordion("Xinference and Model Configuration", open=False):
                    xinference_url = gr.Textbox(label="Xinference URL", value=settings.XINFERENCE_API_URL)
                    llm_model = gr.Textbox(label="LLM Model", value=settings.DEFAULT_LLM_MODEL_NAME)
                    embedding_model = gr.Textbox(label="Embedding Model", value=settings.DEFAULT_EMBEDDING_MODEL_NAME)
                    reranker_model = gr.Textbox(label="Reranker Model", value=settings.DEFAULT_RERANKER_MODEL_NAME)

                with gr.Accordion("Document Processing", open=False):
                    parent_chunk_size = gr.Slider(label="Parent Chunk Size", minimum=500, maximum=5000, value=settings.DEFAULT_PARENT_CHUNK_SIZE, step=100)
                    parent_chunk_overlap = gr.Slider(label="Parent Chunk Overlap", minimum=0, maximum=1000, value=settings.DEFAULT_PARENT_CHUNK_OVERLAP, step=50)
                    child_chunk_size = gr.Slider(label="Child Chunk Size", minimum=100, maximum=1000, value=settings.DEFAULT_CHILD_CHUNK_SIZE, step=50)
                    child_chunk_overlap = gr.Slider(label="Child Chunk Overlap", minimum=0, maximum=200, value=settings.DEFAULT_CHILD_CHUNK_OVERLAP, step=10)

                with gr.Accordion("Retrieval Parameters", open=False):
                    vector_top_k = gr.Slider(label="Vector Top K", minimum=1, maximum=50, value=settings.DEFAULT_VECTOR_STORE_TOP_K, step=1)
                    keyword_top_k = gr.Slider(label="Keyword Top K", minimum=1, maximum=50, value=settings.DEFAULT_KEYWORD_SEARCH_TOP_K, step=1)
                    final_top_n_retrieval = gr.Slider(label="Final Top N Retrieval", minimum=1, maximum=50, value=settings.DEFAULT_RETRIEVAL_FINAL_TOP_N, step=1)

                with gr.Accordion("Pipeline Execution", open=False):
                    max_refinement_iterations = gr.Slider(label="Max Refinement Iterations", minimum=0, maximum=10, value=settings.DEFAULT_MAX_REFINEMENT_ITERATIONS, step=1)
                    max_workflow_iterations = gr.Slider(label="Max Workflow Iterations", minimum=10, maximum=1000, value=settings.DEFAULT_PIPELINE_MAX_WORKFLOW_ITERATIONS, step=5)

                with gr.Accordion("Vector Store and Indexing", open=False):
                    vector_store_path = gr.Textbox(label="Vector Store Path", value="./my_vector_indexes/")
                    index_name = gr.Textbox(label="Index Name", value="abms-V2")
                    force_reindex = gr.Checkbox(label="Force Re-index")

                with gr.Accordion("Logging", open=False):
                    log_level = gr.Dropdown(label="Log Level", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], value=settings.LOG_LEVEL)
                    debug = gr.Checkbox(label="Debug Mode")
                    log_path = gr.Textbox(label="Log Path", value="./logs/")

            with gr.Column(scale=2):
                run_button = gr.Button("Run Report Generation", variant="primary")
                status = gr.Textbox(label="Workflow State", value="Idle", interactive=False)

                with gr.Tabs():
                    with gr.TabItem("Logs"):
                        log_output = gr.Textbox(label="Logs", lines=20, max_lines=20, autoscroll=True, interactive=False)
                    with gr.TabItem("Result"):
                        result_display = gr.Markdown(label="Generated Report")

    run_button.click(
        fn=run_script,
        inputs=[
            topic, data_path, output_path, report_title,
            xinference_url, llm_model, embedding_model, reranker_model,
            parent_chunk_size, parent_chunk_overlap, child_chunk_size, child_chunk_overlap,
            vector_top_k, keyword_top_k, final_top_n_retrieval,
            max_refinement_iterations, max_workflow_iterations,
            vector_store_path, index_name, force_reindex,
            log_level, debug, log_path
        ],
        outputs=[log_output, status]
    )

import argparse

parser = argparse.ArgumentParser(description="Gradio Web UI for RAG Report Generation")
parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to run the Gradio app on")
parser.add_argument("--port", type=int, default=7860, help="Port to run the Gradio app on")
args, unknown = parser.parse_known_args()

iface.launch(server_name=args.host, server_port=args.port,share=True)
