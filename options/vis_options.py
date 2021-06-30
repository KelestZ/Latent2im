import argparse
import oyaml as yaml

class VisOptions():
    def __init__(self):
        self.initialized = False
        self.parser = parser = argparse.ArgumentParser("Visualization Parser")

    def initialize(self):
        parser = self.parser
        # parser.add_argument('weight_path', default='', help="path to saved weights")
        parser.add_argument('config_file', type=argparse.FileType(mode='r'), help="configuration yml file")
        # Load model:
        parser.add_argument('--save_path_w', type=str,  help='')
        parser.add_argument('--save_path_gan', type=str,  help='')

        parser.add_argument("--gpu", default="", type=str, help='GPUs to use (leave blank for CPU only)')
        parser.add_argument('--noise_seed', type=int, default=0, help="noise seed for z samples")
        parser.add_argument('--output_dir', help="where to save output; if specified, overrides output_dir in config file")

        # New
        parser.add_argument('--attrList', type=str)
        # parser.add_argument('--attrPath', type=str, default='/home/peiye/transient_scene/annotations/attributes_celeba.txt')
        parser.add_argument('--attrPath', type=str, default='')
        # /home/peiye/celeba/attributes_celeba.txt
        self.initialized = True
        return self.parser


    def parse(self):
        # initialize parser with basic options
        if not self.initialized:
            self.initialize()

        # parse options
        opt = self.parser.parse_args()

        # get arguments specified in config file
        # and convert to a namespace
        data = yaml.load(opt.config_file, Loader=yaml.FullLoader)
        for k,v in data.items():
            if isinstance(v, dict):
                data[k] = argparse.Namespace(**v)
        data = argparse.Namespace(**data)

        self.opt = opt
        self.data = data
        return opt, data
