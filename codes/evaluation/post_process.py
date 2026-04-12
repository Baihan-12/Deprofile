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
    # 1. 输入文件夹路径
    home_dir = '/hpc_stor03/sjtu_home/baihan.li/deprofile/ACL_agent/evaluation/results/qwen3-4B-ablation'
    output_dir = os.path.join(home_dir+'_QA_ver3_cleaned')
    os.makedirs(output_dir, exist_ok=True)


    for idir in os.listdir(home_dir):
        if idir in os.listdir(output_dir) :
            continue
        input_dir = os.path.join(home_dir, idir)
        idx = idir.split('_')[1]
        output_dir = os.path.join("/hpc_stor03/sjtu_home/baihan.li/deprofile/ACL_agent/evaluation/cleaned_result/qwen3-4B-ablation_QA_ver4_cleaned", f'{idx}')
        process_and_save(input_dir, output_dir)

    