from oemer.ete_part1 import runModel1
from oemer.ete_part2 import runModel2
import numpy as np
import os

input_path = 'C:/Ellie/ellie2023~2024/iis/oemer/images/week3/testFiles/simple_dumky.png'
# dataDictOrigin = runModel1(input_path,outputPath = "./images/week3/",dodewarp=False,save_npy=True)

dataDict = np.load('images/week3/simple_dumky.npy',allow_pickle=True)
dataDict = dataDict.tolist()
runModel2(input_path, dataDict, outputPath="./images/week3/")


# folder_path = 'images/testing/origin'
# files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and not f.startswith('1')]

# for file in files:
#     print(folder_path+'/'+file)
#     eteTemp.main(folder_path+'/'+file)

# eteTemp.main("C:/Ellie/ellie2023~2024/iis/oemer/images/newScope0613/orch.png")