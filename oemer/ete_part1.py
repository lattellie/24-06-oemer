import os
import pickle
import urllib.request
from pathlib import Path
from typing import Tuple

from PIL import Image
from numpy import ndarray

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import numpy as np

from oemer import MODULE_PATH
from oemer import layers
from oemer.inference import inference
from oemer.logger import get_logger
from oemer.dewarp import estimate_coords, dewarp
from oemer.staffline_extraction import extract as staff_extract
from oemer.notehead_extraction import extract as note_extract
from oemer.note_group_extraction import extract as group_extract
from oemer.symbol_extraction import extract as symbol_extract
from oemer.rhythm_extraction import extract as rhythm_extract
from oemer.build_system import MusicXMLBuilder
from oemer.draw_teaser import teaser
from oemer.ellie_createImg import createImg

logger = get_logger(__name__)


CHECKPOINTS_URL = {
    "1st_model.onnx": "https://github.com/BreezeWhite/oemer/releases/download/checkpoints/1st_model.onnx",
    "1st_weights.h5": "https://github.com/BreezeWhite/oemer/releases/download/checkpoints/1st_weights.h5",
    "2nd_model.onnx": "https://github.com/BreezeWhite/oemer/releases/download/checkpoints/2nd_model.onnx",
    "2nd_weights.h5": "https://github.com/BreezeWhite/oemer/releases/download/checkpoints/2nd_weights.h5"
}



def clear_data() -> None:
    lls = layers.list_layers()
    for l in lls:
        layers.delete_layer(l)


def generate_pred(img_path: str, use_tf: bool = False) -> Tuple[ndarray, ndarray, ndarray, ndarray, ndarray]:
    logger.info("Extracting staffline and symbols")
    staff_symbols_map, _ = inference(
        os.path.join(MODULE_PATH, "checkpoints/unet_big"),
        img_path,
        use_tf=use_tf,
    )
    staff = np.where(staff_symbols_map==1, 1, 0)
    symbols = np.where(staff_symbols_map==2, 1, 0)

    logger.info("Extracting layers of different symbols")
    sep, _ = inference(
        os.path.join(MODULE_PATH, "checkpoints/seg_net"),
        img_path,
        manual_th=None,
        use_tf=use_tf,
    )
    stems_rests = np.where(sep==1, 1, 0)
    notehead = np.where(sep==2, 1, 0)
    clefs_keys = np.where(sep==3, 1, 0)
    # stems_rests = sep[..., 0]
    # notehead = sep[..., 1]
    # clefs_keys = sep[..., 2]

    return staff, symbols, stems_rests, notehead, clefs_keys

def extract(img_path: str, output_path: str, dodewarp: bool,save_npy: bool) -> str:
    img_path = Path(img_path)
    f_name = os.path.splitext(img_path.name)[0]
    pkl_path = img_path.parent / f"{f_name}.pkl"
    if pkl_path.exists():
        # Load from cache
        pred = pickle.load(open(pkl_path, "rb"))
        notehead = pred["note"]
        symbols = pred["symbols"]
        staff = pred["staff"]
        clefs_keys = pred["clefs_keys"]
        stems_rests = pred["stems_rests"]
    else:
        staff, symbols, stems_rests, notehead, clefs_keys = generate_pred(str(img_path), use_tf=False)

    # Load the original image, resize to the same size as prediction.
    image_pil = Image.open(str(img_path))
    if "GIF" != image_pil.format:
        image = cv2.imread(str(img_path))
    else:
        gif_image = image_pil.convert('RGB')
        gif_img_arr = np.array(gif_image)
        image = gif_img_arr[:, :, ::-1].copy()

    image = cv2.resize(image, (staff.shape[1], staff.shape[0]))

    if dodewarp:
        logger.info("Dewarping")
        coords_x, coords_y = estimate_coords(staff)
        staff = dewarp(staff, coords_x, coords_y)
        symbols = dewarp(symbols, coords_x, coords_y)
        stems_rests = dewarp(stems_rests, coords_x, coords_y)
        clefs_keys = dewarp(clefs_keys, coords_x, coords_y)
        notehead = dewarp(notehead, coords_x, coords_y)
        for i in range(image.shape[2]):
            image[..., i] = dewarp(image[..., i], coords_x, coords_y)

    # Register predictions
    symbols = symbols + clefs_keys + stems_rests
    symbols[symbols>1] = 1
    dataDict = dict()
    dataDict['stem_rest'] = stems_rests
    dataDict['clefs_keys'] = clefs_keys
    dataDict['staff'] = staff
    dataDict['notehead'] = notehead
    dataDict['symbols'] = symbols
    dataDict['image'] = image

    if save_npy:
        np.save(output_path+f_name+'.npy', dataDict)
    return dataDict

def download_file(title: str, url: str, save_path: str) -> None:
    resp = urllib.request.urlopen(url)
    length = int(resp.getheader("Content-Length", -1))

    chunk_size = 2**9
    total = 0
    with open(save_path, "wb") as out:
        while True:
            print(f"{title}: {total*100/length:.1f}% {total}/{length}", end="\r")
            data = resp.read(chunk_size)
            if not data:
                break
            total += out.write(data)
        print(f"{title}: 100% {length}/{length}"+" "*20)


def runModel1(img_path,outputPath = "./images", dodewarp=False, save_npy=True) -> dict:
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"The given image path doesn't exists: {img_path}")

    # Check there are checkpoints
    chk_path = os.path.join(MODULE_PATH, "checkpoints/unet_big/model.onnx")
    if not os.path.exists(chk_path):
        logger.warn("No checkpoint found in %s", chk_path)
        for idx, (title, url) in enumerate(CHECKPOINTS_URL.items()):
            logger.info(f"Downloading checkpoints ({idx+1}/{len(CHECKPOINTS_URL)})")
            save_dir = "unet_big" if title.startswith("1st") else "seg_net"
            save_dir = os.path.join(MODULE_PATH, "checkpoints", save_dir)
            save_path = os.path.join(save_dir, title.split("_")[1])
            download_file(title, url, save_path)

    clear_data()
    dataDict = extract(img_path,outputPath, dodewarp,save_npy)
    return dataDict

