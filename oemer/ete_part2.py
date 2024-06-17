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
    symbol_thresholds = [0.5, 0.4, 0.4]
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


def polish_symbols(rgb_black_th=300):
    img = layers.get_layer('original_image')
    sym_pred = layers.get_layer('symbols_pred')

    img = Image.fromarray(img).resize((sym_pred.shape[1], sym_pred.shape[0]))
    arr = np.sum(np.array(img), axis=-1)
    arr = np.where(arr < rgb_black_th, 1, 0)  # Filter background
    ker = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    arr = cv2.dilate(cv2.erode(arr.astype(np.uint8), ker), ker)  # Filter staff lines
    mix = np.where(sym_pred+arr>1, 1, 0)
    return mix


def register_notehead_bbox(bboxes):
    symbols = layers.get_layer('symbols_pred')
    layer = layers.get_layer('bboxes')
    for (x1, y1, x2, y2) in bboxes:
        yi, xi = np.where(symbols[y1:y2, x1:x2]>0)
        yi += y1
        xi += x1
        layer[yi, xi] = np.array([x1, y1, x2, y2])
    return layer


def register_note_id() -> None:
    symbols = layers.get_layer('symbols_pred')
    layer = layers.get_layer('note_id')
    notes = layers.get_layer('notes')
    for idx, note in enumerate(notes):
        x1, y1, x2, y2 = note.bbox
        yi, xi = np.where(symbols[y1:y2, x1:x2]>0)
        yi += y1
        xi += x1
        layer[yi, xi] = idx
        notes[idx].id = idx


def extract(img_path:str,model1Data:dict,outputPath:str, dodewarp:bool) -> str:
    if len(outputPath)>0 and outputPath[-1] != '/':
        outputPath = outputPath + '/'
    img_path = Path(img_path)
    f_name = os.path.splitext(img_path.name)[0]
    stems_rests = model1Data['stems_rests']
    clefs_keys = model1Data['clefs_keys']
    notehead = model1Data['notehead']
    symbols = model1Data['symbols']
    staff = model1Data['staff']
    image = model1Data['image']

    layers.register_layer("stems_rests_pred", stems_rests)
    layers.register_layer("clefs_keys_pred", clefs_keys)
    layers.register_layer("notehead_pred", notehead)
    layers.register_layer("symbols_pred", symbols)
    layers.register_layer("staff_pred", staff)
    layers.register_layer("original_image", image)

    # ---- Extract staff lines and group informations ---- #
    logger.info("Extracting stafflines")
    staffs, zones = staff_extract()
    layers.register_layer("staffs", staffs)  # Array of 'Staff' instances
    layers.register_layer("zones", zones)  # Range of each zones, array of 'range' object.

    # ---- Extract noteheads ---- #
    logger.info("Extracting noteheads")
    notes = note_extract()

    # Array of 'NoteHead' instances.
    layers.register_layer('notes', np.array(notes))

    # Add a new layer (w * h), indicating note id of each pixel.
    layers.register_layer('note_id', np.zeros(symbols.shape, dtype=np.int64)-1)
    register_note_id()

    # ---- Extract groups of note ---- #
    logger.info("Grouping noteheads")
    groups, group_map = group_extract()
    layers.register_layer('note_groups', np.array(groups))
    layers.register_layer('group_map', group_map)

    # ---- Extract symbols ---- #
    logger.info("Extracting symbols")
    barlines, clefs, sfns, rests = symbol_extract()
    layers.register_layer('barlines', np.array(barlines))
    layers.register_layer('clefs', np.array(clefs))
    layers.register_layer('sfns', np.array(sfns))
    layers.register_layer('rests', np.array(rests))

    # ---- Parse rhythm ---- #
    logger.info("Extracting rhythm types")
    rhythm_extract()

    mainDir = outputPath+'mergeImg/'+f_name
    clefsfnrestImg = createImg('clefsfnrest',image,clefs,sfns,rests)
    notesfnImg = createImg('notesfn',image,notes,'','')
    notesImg = createImg('notes',image,notes,'','')
    staffbarlineImg = createImg('staffbarline',image,staffs,barlines,'')
    
    if clefsfnrestImg.shape == notesfnImg.shape == notesImg.shape == staffbarlineImg.shape:
        x = np.hstack((clefsfnrestImg,notesfnImg,notesImg,staffbarlineImg))
        cv2.imwrite(mainDir+'.jpg',x)
        print(f'image {mainDir}.jpg created')
    else:
        print('4 images have different size, cannot merge')

    # ---- Build MusicXML ---- #
    logger.info("Building MusicXML document")
    basename = os.path.basename(img_path).replace(".jpg", "").replace(".png", "")
    builder = MusicXMLBuilder(title=basename.capitalize())
    builder.build()
    xml = builder.to_musicxml()

    # ---- Write out the MusicXML ---- #
    out_path = outputPath
    if not out_path.endswith(".musicxml"):
        # Take the output path as the folder.
        out_path = os.path.join(out_path, basename+".musicxml")

    with open(out_path, "wb") as ff:
        ff.write(xml)

    return out_path


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


def runModel2(imgpath, model1Data,outputPath = "./images/", dodewarp=False) -> None:
    mxl_path = extract(imgpath, model1Data,outputPath, dodewarp)
    # img = teaser()
    # img.save(mxl_path.replace(".musicxml", "_teaser.png"))