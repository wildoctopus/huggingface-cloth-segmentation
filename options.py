import os.path as osp
import os


class parser(object):
    def __init__(self):
        
        self.output = "./output"  # output image folder path  
        self.logs_dir = './logs'
        self.device = 'cuda:0'

opt = parser()