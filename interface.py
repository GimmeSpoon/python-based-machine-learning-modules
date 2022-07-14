import argparse
import json
import yaml
import cmd
import sys
import os
import re

class Config(object):
    def __init__(self, dict:dict)->None:
        self.update(dict)
    
    def update(self, args):
        self.__dict__.update(args)
            
class DCS(cmd.Cmd):
    """
    cool shell
    """
    def cmdloop(self, intro=None) -> None:
        print("Type 'help' for more information.")
        return cmd.Cmd.cmdloop(self, intro)

    def do_train(self):
        pass

    def do_infer(self):
        pass

parser = argparse.ArgumentParser(
            prog="General Machine Learning Framework",
            description="General ML framework for various use. It provides CLI interface with argparser and basic training and inference actions."
         )

# Basic config
parser.add_argument('mode', type=str, choices=['train', 'infer', 'shell'], help='Configure ML types')
parser.add_argument('-s', '--silent', action='store_true', help='Skips logo if set')
parser.add_argument('-C','--cmd', action='store_true', help='Interactive cmd')
parser.add_argument('--input_path', type=argparse.FileType('r'), metavar='INPUT_PATH')
parser.add_argument('--output_path', type=str, metavar='OUTPUT_PATH')
parser.add_argument('--checkpoint_path', type=str, metavar='CHECKPOINT_PATH')

# Common Hyperparameters
parser.add_argument('-l', '--lr', type=float, default=0.001, metavar='LEARNING_RATE', help='initial learning rate', dest='learning_rate')

# You can also provide config values by file. If file path is passed, hyperparameters before will be ignored.
parser.add_argument('--config_path', type=str, default='', metavar='CONFIG_PATH')
parser.add_argument('--config_type', choices=['xml', 'json', 'yaml'], default='yaml', metavar='CONFIG_TYPE')

def load_config(path, type)->Config:
    """
    load config file into one object
    :param type:type(str)
    :return: Configuration object
    """
    config_file = open(path, 'rt')
    config = None
    if type == 'xml':
        parser.exit(message="XML is not supported yet.")
    elif type == 'json':
        config = json.load(config_file)
    elif type == 'yaml' or type == 'yml':
        config = yaml.load(config_file, Loader=yaml.SafeLoader)
    else:
        print(f"Undefined extension : {type}")
    return Config(config)

def resolve_config(args, config:Config)->argparse.Namespace:
    """
    Assing properties form config file to args
    :param config: Configurations
    :return: args(Namespace)
    """
    # Validate config properties here.
    #    
    config.update(vars(args))
    return config

if __name__ == "__main__":
    args = parser.parse_args()
    # config file handle here
    if args.config_path:
        config = resolve_config(args=args, config=load_config(path=args.config_path, type=args.config_type))
    if not args.silent:
        logo = open('logo.txt','rt').read()
    shell = DCS()
    if args.cmd:
        DCS.cmdloop(shell, logo)
    else:
        if args.mode == "train":
            DCS.onecmd(shell, '')
        elif args.mode == "infer":
            DCS.onecmd(shell)
        else:
            print(f"Invalid argument: mode= {args.mode}")