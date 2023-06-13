import argparse
import os

# from mme.utils import Config
from mmengine.config import Config

def parse_args():
    parser = argparse.ArgumentParser(
        description='load configs and dump the entire configs')
    parser.add_argument('-c', '--config', help='config file')
    parser.add_argument('-o', '--out', help='output image')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    print("Load...")
    cfg = Config.fromfile(args.config)
    name = os.path.splitext(args.config)[0] + "_while" + os.path.splitext(args.config)[1]
    print("Dump...")
    
    if args.out is None:
        savepath = name
    if os.path.isdir(args.out):
        if 'nt' in os.name.lower():
            name = name.split('\\')[-1]
        else:
            name = name.split('/')[-1]
        savepath = os.path.join(args.out, name)
    else:
        savepath = args.out
    result = cfg.dump(savepath)
    print("Done")

if __name__ == "__main__":
    main()