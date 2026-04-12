import requests
import json
import argparse  # 导入参数解析库
import sys

def main():
    # 1. 设置命令行参数解析
    parser = argparse.ArgumentParser(description="patient_simulation 批量测试脚本")
    
    # 添加参数：--id 或 -i，默认值为 G7
    parser.add_argument("--id", "-i", type=str, default="G7", help="指定 Baseline ID (默认: G7)")
    
    # 添加参数：--url 或 -u，默认值为 http://localhost:8002/api/run_batch
    parser.add_argument("--url", "-u", type=str, 
                        default="http://localhost:8002/api/run_batch", 
                        help="指定服务器接口地址")
    
    # 添加参数：--name 或 -n，给运行起名字
    parser.add_argument("--name", "-n", type=str, default="ablation", help="运行名称 (默认: ablation)")

    args = parser.parse_args()

    # ================= 配置逻辑 =================
    BASELINE_ID = args.id
    API_URL = args.url
    RUN_NAME = args.name

    # 读取问题列表
    try:
        with open("/hpc_stor03/sjtu_home/baihan.li/deprofile/ACL_agent/evaluation/questions/QA_ALL.json", "r", encoding="utf-8") as f:
            QUESTION_LIST = list(json.load(f).values())
    except FileNotFoundError:
        print("❌ 错误: 找不到 QA.json 文件")
        sys.exit(1)

    payload = {
        "baseline": BASELINE_ID,
        "questions": QUESTION_LIST,
        "language": "zh",
        "run_name": RUN_NAME
    }

    print(f"🚀 正在发送请求...")
    print(f"📍 目标地址: {API_URL}")
    print(f"🆔 Baseline : {BASELINE_ID}")
    print(f"🏷️ 运行名称 : {RUN_NAME}")

    try:
        response = requests.post(API_URL, json=payload)
        if response.status_code == 200:
            res = response.json()
            print("\n✅ 任务已成功启动！")
            output_path = res.get('output_directory') or res.get('output_file_pattern')
            print(f"📁 结果将保存至: {output_path}")
        else:
            print(f"\n❌ 请求失败: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"\n❌ 连接错误 (请检查 backend 是否已启动): {e}")

if __name__ == "__main__":
    main()