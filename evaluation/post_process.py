import json
import re
import os
import glob
from tqdm import tqdm  # 如果没安装，可以 pip install tqdm，或者把相关代码删掉，不影响运行

# ================= 配置区域 =================

# ===========================================

def clean_answer_aggressive(text):
    
    # 1. 移除 <think> 标签块
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    
    # 2. 移除所有换行符 (解决你的痛点)
    text = text.replace('\n', '')
    
    # 3. 移除所有空格 (让句子紧凑)
    # 如果你想保留句子间的空格，注释掉下面这行即可
    text = text.replace(' ', '')
    
    return text.strip()

def process_and_save(input_dir, output_dir):
    # 1. 检查并创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"已创建输出目录: {output_dir}")
    else:
        print(f"输出目录已存在: {output_dir}")

    # 2. 获取所有 json 文件
    json_files = glob.glob(os.path.join(input_dir, "*.json"))
    print(f"共发现 {len(json_files)} 个 JSON 文件，准备处理...")

    # 3. 遍历处理
    count = 0
    # 使用 tqdm 显示进度条，如果没有安装 tqdm，直接用 for file_path in json_files: 即可
    for file_path in tqdm(json_files, desc="Processing"):
        file_name = os.path.basename(file_path)
        save_path = os.path.join(output_dir, file_name)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # 兼容处理：尝试读取标准 JSON，失败则尝试 JSONL
                is_jsonl = False
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    f.seek(0)
                    data = [json.loads(line) for line in f]
                    is_jsonl = True
            
            items = data['dialogue']


            # === 执行清洗操作 ===
            for item in items:
                # 检查是否存在 answer 字段
                if 'answer' in item:
                    original = item['answer']
                    # print(original)
                    cleaned = clean_answer_aggressive(original)
                    
                    # 直接修改 answer 字段的内容
                    item['answer'] = cleaned
                    
                    # 【可选】如果你想保留原始思维链做备份，可以取消下面这行的注释
                    # item['raw_answer_with_think'] = original
            data['dialogue'] = items
            # === 保存文件 ===
            with open(save_path, 'w', encoding='utf-8') as f_out:
                f_out.write(json.dumps(data, ensure_ascii=False, indent=4))
            
            count += 1

        except Exception as e:
            print(f"处理文件 {file_name} 时出错: {e}")

    print(f"\n处理完成！")
    print(f"成功清洗并保存了 {count} 个文件。")
    print(f"清洗后的文件位于: {output_dir}")

def printer(data):
    data = json.loads(data)
    for key, value in data.items():
        print(key)
        print(value)
        print('-'*100)

if __name__ == "__main__":
    import argparse

    from repo_paths import CLEANED_RESULT_DIR, RESULTS_DIR

    parser = argparse.ArgumentParser(description="Clean answer fields in batch QA JSON outputs.")
    parser.add_argument(
        "--results_parent",
        default=str(RESULTS_DIR / "qwen3-4B-ablation"),
        help="Directory containing per-run subfolders (each with profile *.json).",
    )
    parser.add_argument(
        "--cleaned_parent",
        default=str(CLEANED_RESULT_DIR / "qwen3-4B-ablation_QA_ver4_cleaned"),
        help="Parent directory for cleaned outputs (one subfolder per run).",
    )
    args = parser.parse_args()

    home_dir = args.results_parent
    cleaned_root = args.cleaned_parent
    os.makedirs(cleaned_root, exist_ok=True)

    for idir in os.listdir(home_dir):
        input_dir = os.path.join(home_dir, idir)
        if not os.path.isdir(input_dir):
            continue
        parts = idir.split("_")
        idx = parts[1] if len(parts) > 1 else idir
        out_sub = os.path.join(cleaned_root, idx)
        if os.path.isdir(out_sub) and os.listdir(out_sub):
            continue
        process_and_save(input_dir, out_sub)
