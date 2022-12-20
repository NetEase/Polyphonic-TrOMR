import os
from omegaconf import OmegaConf

def getconfig(configpath):
    args = OmegaConf.load(configpath)

    workspace = os.path.dirname(configpath)
    for key in args.filepaths.keys():
        args.filepaths[key] = os.path.join(workspace, args.filepaths[key])
    return args