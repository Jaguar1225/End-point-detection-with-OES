import time
import datetime
import argparse
import json

from utils.analysis import Analysis

def parse_list(arg):
    try:
        return [eval(x) for x in arg.split(',')]
    except ValueError:
        raise argparse.ArgumentTypeError("리스트는 쉼표로 구분된 정수여야 합니다.")

def parse_bool(arg):
    try:
        return eval(arg)
    except ValueError:
        raise argparse.ArgumentTypeError("주어진 인자는 bool이 아닙니다.")
    
def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def get_params(params, args):

    if args.data_path:
        params["data_path"] = args.data_path
    if args.model_name:
        params["model_name"] = args.model_name
    if args.window_size:
        params["window_size"] = args.window_size
    if args.num_channels:
        params["num_channels"] = args.num_channels
    if args.save_dir:
        params["save_dir"] = args.save_dir
    return params

def main():
    parser = argparse.ArgumentParser(description="OriginLab 데이터 분석 프로그램")

    parser.add_argument("--data_path", type=str, default="data.csv", help="데이터 파일 경로")
    parser.add_argument("--model_name", type=str, default="KMC", help="모델 이름")
    parser.add_argument("--window_size", type=int, default=10, help="윈도우 크기")
    parser.add_argument("--num_channels", type=int, default=25, help="채널 수")

    args = parser.parse_args()

    try:
        params = load_config('./config/config.json')
        params = get_params(params, args)

        analysis = Analysis(**params)
        analysis.analysis()
    
    except Exception as e:
        print(e)
    
    

if __name__ == "__main__":
    main()
