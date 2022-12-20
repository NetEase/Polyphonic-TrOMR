import os

import argparse

from configs import getconfig
from staff2score import StaffToScore

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference single staff image')
    parser.add_argument('filepath', type=str, help='path to staff image')
    parsed_args = parser.parse_args()
    
    cofigpath = os.path.join(os.path.dirname(__file__), "workspace", "config.yaml")
    args = getconfig(cofigpath)
    
    handler = StaffToScore(args)
    predrhythms, predpitchs, predlifts = handler.predict(parsed_args.filepath)
    
    mergeds = []
    for i in range(len(predrhythms)):
        predlift = predlifts[i]
        predpitch = predpitchs[i]
        predrhythm = predrhythms[i]
        
        merge = predrhythm[0] + '+'
        for j in range(1, len(predrhythm)):
            if predrhythm[j] == "|":
                merge = merge[:-1]+predrhythm[j]
            elif "note" in predrhythm[j]:
                lift = ""
                if predlift[j] in ("lift_##", "lift_#", "lift_bb", "lift_b", "lift_N",):
                    lift = predlift[j].split("_")[-1]
                merge += predpitch[j]+lift+"_"+predrhythm[j].split('note-')[-1]+"+"
            else:
                merge += predrhythm[j]+"+"
        mergeds.append(merge[:-1])
    print(mergeds)