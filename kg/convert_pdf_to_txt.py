import pymupdf  # PyMuPDF库
import os
import sys

def convert_pdfs_to_txt(source_directory, target_directory):
    """
    遍历源目录中的所有PDF文件，将它们的内容转换为TXT文件，
    并保存在目标目录中。
    """
    # 确保目标目录存在
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)
        print(f"创建目标目录: {target_directory}")

    print(f"开始从 '{source_directory}' 转换PDF文件...")
    
    # 遍历源目录中的所有文件
    for filename in os.listdir(source_directory):
        if filename.lower().endswith('.pdf'):
            source_path = os.path.join(source_directory, filename)
            print(f"处理文件: {filename}")
            # 构造目标文件名，将.pdf后缀替换为.txt
            target_filename = os.path.splitext(filename)[0] + ".txt"
            target_path = os.path.join(target_directory, target_filename)

            try:
                # 打开PDF文件
                doc = pymupdf.open(source_path)
                full_text = ""
                # 逐页提取文本并拼接
                for page in doc:
                    full_text += page.get_text()
                
                doc.close()

                # 将提取的文本写入TXT文件
                with open(target_path, 'w', encoding='utf-8') as f:
                    f.write(full_text)
                
                print(f"成功转换: {filename} -> {target_filename}")

            except Exception as e:
                print(f"转换失败: {filename}。错误: {e}", file=sys.stderr)

    print("所有PDF文件转换完成。")

if __name__ == '__main__':
    # --- 请根据您的实际路径修改以下两个变量 ---
    
    # 存放您原始PDF文档的目录
    raw_pdf_dir = '/home/ISTIC_0/abms/data' 
    
    # GraphRAG项目用于读取输入的目录 (通常是项目根目录下的 'input' 文件夹)
    graphrag_input_dir = './input' 
    
    # 检查源目录是否存在
    if not os.path.isdir(raw_pdf_dir):
        print(f"错误：源目录 '{raw_pdf_dir}' 不存在。请创建该目录并放入您的PDF文件。", file=sys.stderr)
    else:
        convert_pdfs_to_txt(raw_pdf_dir, graphrag_input_dir)