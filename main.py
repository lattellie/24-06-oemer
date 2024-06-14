from oemer.ete_part1 import runModel1
from oemer import ete_part2
import os

input_path = 'C:/Ellie/ellie2023~2024/iis/oemer/images/newScope0613/1_tch1_doubleSharp.png'
dataDict = runModel1(input_path,outputPath = "./images/week2",dodewarp=False,save_npy=True)


# folder_path = 'images/testing/origin'
# files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and not f.startswith('1')]

# for file in files:
#     print(folder_path+'/'+file)
#     eteTemp.main(folder_path+'/'+file)

# eteTemp.main("C:/Ellie/ellie2023~2024/iis/oemer/images/newScope0613/orch.png")