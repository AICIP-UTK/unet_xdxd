# i*- coding: utf-8 -*-
"""
v13 model

* Input: v12_im

Author: Kohei <i@ho.lc>
"""
from logging import getLogger, Formatter, StreamHandler, INFO, FileHandler
from pathlib import Path
import subprocess
import glob
import math
import sys
import json
import re
import warnings
import os

import scipy
import tqdm
import click
import tables as tb
import pandas as pd
import numpy as np
from keras.models import Model
from keras.engine.topology import merge as merge_l
from keras.layers import (
    Input, Convolution2D, MaxPooling2D, UpSampling2D,
    Reshape, core, Dropout,
    Activation, BatchNormalization)
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping, History
from keras import backend as K
import skimage.draw
import rasterio
import rasterio.features
import shapely.wkt
import shapely.ops
import shapely.geometry
import skimage.transform as ski_transform

#Added: Jan 30th: mask2linestrings.
import argparse
import cv2
import queue
from shapely.geometry import LineString, Point
import sys
from skimage import morphology

# Constants
width = 1300
height = 1300
white = 255
black = 0
spacing = 21

# Generates the lists of coordinate alterations needed to search around a
# candidate pixel up to a certain spacing away from the candidate pixel
def grid(spacing):
    search = []

    for r in range(-spacing, spacing+1, 1):
        for c in range(-spacing, spacing+1, 1):
            if r == -spacing or r == spacing or c == -spacing or c == spacing:
                search.append((r, c))
    return search

searches = []
for i in range(spacing):
    searches.append(grid(i+1))


MODEL_NAME = 'v13'
ORIGINAL_SIZE = 1300
INPUT_SIZE = 256
STRIDE_SZ = 197

BASE_DIR = "/data/train"
BASE_TEST_DIR = "/data/test"
WORKING_DIR = "/data/working"
IMAGE_DIR = "/data/working/images/{}".format('v5')
V5_IMAGE_DIR = "/data/working/images/{}".format('v5')

# ---------------------------------------------------------
# Parameters
MIN_POLYGON_AREA = 30  # 30

# ---------------------------------------------------------
# Input files
FMT_TRAIN_SUMMARY_PATH = str(
    Path(BASE_DIR) /
    Path("{prefix:s}_Train/") /
    Path("summaryData/{prefix:s}_Train_Building_Solutions.csv"))
FMT_TRAIN_RGB_IMAGE_PATH = str(
    Path(BASE_DIR) /
    Path("{prefix:s}_Train/") /
    Path("RGB-PanSharpen/RGB-PanSharpen_{image_id:s}.tif"))
FMT_TEST_RGB_IMAGE_PATH = str(
    Path(BASE_TEST_DIR) /
    Path("{prefix:s}_Test/") /
    Path("RGB-PanSharpen/RGB-PanSharpen_{image_id:s}.tif"))
FMT_TRAIN_MSPEC_IMAGE_PATH = str(
    Path(BASE_DIR) /
    Path("{prefix:s}_Train/") /
    Path("MUL-PanSharpen/MUL-PanSharpen_{image_id:s}.tif"))
FMT_TEST_MSPEC_IMAGE_PATH = str(
    Path(BASE_TEST_DIR) /
    Path("{prefix:s}_Test/") /
    Path("MUL-PanSharpen/MUL-PanSharpen_{image_id:s}.tif"))

# ---------------------------------------------------------
# Preprocessing result
FMT_RGB_BANDCUT_TH_PATH = IMAGE_DIR + "/rgb_bandcut.csv"
FMT_MUL_BANDCUT_TH_PATH = IMAGE_DIR + "/mul_bandcut.csv"

# ---------------------------------------------------------
# Image list, Image container and mask container
FMT_VALTRAIN_IMAGELIST_PATH = V5_IMAGE_DIR + "/{prefix:s}_valtrain_ImageId.csv"
FMT_VALTEST_IMAGELIST_PATH = V5_IMAGE_DIR + "/{prefix:s}_valtest_ImageId.csv"
FMT_VALTRAIN_IM_STORE = IMAGE_DIR + "/valtrain_{}_im.h5"
FMT_VALTEST_IM_STORE = IMAGE_DIR + "/valtest_{}_im.h5"
FMT_VALTRAIN_MASK_STORE = IMAGE_DIR + "/valtrain_{}_mask.h5"
FMT_VALTEST_MASK_STORE = IMAGE_DIR + "/valtest_{}_mask.h5"
FMT_VALTRAIN_MUL_STORE = IMAGE_DIR + "/valtrain_{}_mul.h5"
FMT_VALTEST_MUL_STORE = IMAGE_DIR + "/valtest_{}_mul.h5"

FMT_TRAIN_IMAGELIST_PATH = V5_IMAGE_DIR + "/{prefix:s}_train_ImageId.csv"
FMT_TEST_IMAGELIST_PATH = V5_IMAGE_DIR + "/{prefix:s}_test_ImageId.csv"
FMT_TRAIN_IM_STORE = IMAGE_DIR + "/train_{}_im.h5"
FMT_TEST_IM_STORE = IMAGE_DIR + "/test_{}_im.h5"
FMT_TRAIN_MASK_STORE = IMAGE_DIR + "/train_{}_mask.h5"
FMT_TRAIN_MUL_STORE = IMAGE_DIR + "/train_{}_mul.h5"
FMT_TEST_MUL_STORE = IMAGE_DIR + "/test_{}_mul.h5"
FMT_MULMEAN = IMAGE_DIR + "/{}_mulmean.h5"

# ---------------------------------------------------------
# Model files
MODEL_DIR = "/data/working/models/{}".format(MODEL_NAME)
FMT_VALMODEL_PATH = MODEL_DIR + "/{}_val_weights.h5"
FMT_FULLMODEL_PATH = MODEL_DIR + "/{}_full_weights.h5"
FMT_VALMODEL_HIST = MODEL_DIR + "/{}_val_hist.csv"
FMT_VALMODEL_EVALHIST = MODEL_DIR + "/{}_val_evalhist.csv"
FMT_VALMODEL_EVALTHHIST = MODEL_DIR + "/{}_val_evalhist_th.csv"

# ---------------------------------------------------------
# Prediction & polygon result
FMT_TESTPRED_PATH = MODEL_DIR + "/{}_pred.h5"
FMT_VALTESTPRED_PATH = MODEL_DIR + "/{}_eval_pred.h5"
FMT_VALTESTPOLY_PATH = MODEL_DIR + "/{}_eval_poly.csv"
FMT_VALTESTLINE_PATH = MODEL_DIR + "/{}_eval_line.csv"
FMT_VALTESTTRUTH_PATH = MODEL_DIR + "/{}_eval_poly_truth.csv"
FMT_VALTESTTRUTHLINE_PATH = MODEL_DIR + "/{}_eval_line_truth.csv"
FMT_VALTESTPOLY_OVALL_PATH = MODEL_DIR + "/eval_poly.csv"
FMT_VALTESTLINE_OVALL_PATH = MODEL_DIR + "/eval_line.csv"
FMT_VALTESTTRUTH_OVALL_PATH = MODEL_DIR + "/eval_poly_truth.csv"
FMT_TESTPOLY_PATH = MODEL_DIR + "/{}_poly.csv"
FMT_TESTLINE_PATH = MODEL_DIR + "/{}_line.csv"
FN_SOLUTION_CSV = "data/output/{}.csv".format(MODEL_NAME)

# ---------------------------------------------------------
# Model related files (others)
FMT_VALMODEL_LAST_PATH = MODEL_DIR + "/{}_val_weights_last.h5"
FMT_FULLMODEL_LAST_PATH = MODEL_DIR + "/{}_full_weights_last.h5"

# ---------------------------------------------------------
# warnins and logging
warnings.simplefilter("ignore", UserWarning)
handler = StreamHandler()
handler.setLevel(INFO)
handler.setFormatter(Formatter('%(asctime)s %(levelname)s %(message)s'))
fh_handler = FileHandler(".{}.log".format(MODEL_NAME))
fh_handler.setFormatter(Formatter('%(asctime)s %(levelname)s %(message)s'))
logger = getLogger(__name__)
logger.setLevel(INFO)


if __name__ == '__main__':
    logger.addHandler(handler)
    logger.addHandler(fh_handler)

# Fix seed for reproducibility
np.random.seed(1145141919)


def directory_name_to_area_id(datapath):
    """
    Directory name to AOI number

    Usage:

        >>> directory_name_to_area_id("/data/test/AOI_2_Vegas")
        2
    """
    dir_name = Path(datapath).name
    if dir_name.startswith('AOI_2_Vegas'):
        return 2
    elif dir_name.startswith('AOI_3_Paris'):
        return 3
    elif dir_name.startswith('AOI_4_Shanghai'):
        return 4
    elif dir_name.startswith('AOI_5_Khartoum'):
        return 5
    else:
        raise RuntimeError("Unsupported city id is given.")


# def _remove_interiors(line):
#     if "), (" in line:
#         line_prefix = line.split('), (')[0]
#         line_terminate = line.split('))",')[-1]
#         line = (
#             line_prefix +
#             '))",' +
#             line_terminate
#         )
#     return line


# def _calc_fscore_per_aoi(area_id):
#     prefix = area_id_to_prefix(area_id)
#     truth_file = FMT_VALTESTTRUTH_PATH.format(prefix)
#     poly_file = FMT_VALTESTPOLY_PATH.format(prefix)
#
#     cmd = [
#         'java',
#         '-jar',
#         '/root/visualizer-2.0/visualizer.jar',
#         '-truth',
#         truth_file,
#         '-solution',
#         poly_file,
#         '-no-gui',
#         '-band-triplets',
#         '/root/visualizer-2.0/data/band-triplets.txt',
#         '-image-dir',
#         'pass',
#     ]
#     proc = subprocess.Popen(
#         cmd,
#         stdout=subprocess.PIPE,
#         stderr=subprocess.PIPE,
#     )
#     stdout_data, stderr_data = proc.communicate()
#     lines = [line for line in stdout_data.decode('utf8').split('\n')[-10:]]
#
#     """
# Overall F-score : 0.85029
#
# AOI_2_Vegas:
#   TP       : 27827
#   FP       : 4999
#   FN       : 4800
#   Precision: 0.847712
#   Recall   : 0.852883
#   F-score  : 0.85029
#     """
#
#     if stdout_data.decode('utf8').strip().endswith("Overall F-score : 0"):
#         overall_fscore = 0
#         tp = 0
#         fp = 0
#         fn = 0
#         precision = 0
#         recall = 0
#         fscore = 0
#
#     elif len(lines) > 0 and lines[0].startswith("Overall F-score : "):
#         assert lines[0].startswith("Overall F-score : ")
#         assert lines[2].startswith("AOI_")
#         assert lines[3].strip().startswith("TP")
#         assert lines[4].strip().startswith("FP")
#         assert lines[5].strip().startswith("FN")
#         assert lines[6].strip().startswith("Precision")
#         assert lines[7].strip().startswith("Recall")
#         assert lines[8].strip().startswith("F-score")
#
#         overall_fscore = float(re.findall("([\d\.]+)", lines[0])[0])
#         tp = int(re.findall("(\d+)", lines[3])[0])
#         fp = int(re.findall("(\d+)", lines[4])[0])
#         fn = int(re.findall("(\d+)", lines[5])[0])
#         precision = float(re.findall("([\d\.]+)", lines[6])[0])
#         recall = float(re.findall("([\d\.]+)", lines[7])[0])
#         fscore = float(re.findall("([\d\.]+)", lines[8])[0])
#     else:
#         logger.warn("Unexpected data >>> " + stdout_data.decode('utf8'))
#         raise RuntimeError("Unsupported format")
#
#     return {
#         'overall_fscore': overall_fscore,
#         'tp': tp,
#         'fp': fp,
#         'fn': fn,
#         'precision': precision,
#         'recall': recall,
#         'fscore': fscore,
#     }


def prefix_to_area_id(prefix):
    area_dict = {
        'AOI_1_Rio': 1,
        'AOI_2_Vegas': 2,
        'AOI_3_Paris': 3,
        'AOI_4_Shanghai': 4,
        'AOI_5_Khartoum': 5,
    }
    return area_dict[area_id]


def area_id_to_prefix(area_id):
    """
    area_id から prefix を返す
    """
    area_dict = {
        1: 'AOI_1_Rio',
        2: 'AOI_2_Vegas',
        3: 'AOI_3_Paris',
        4: 'AOI_4_Shanghai',
        5: 'AOI_5_Khartoum',
    }
    return area_dict[area_id]

# ---------------------------------------------------------
# mask2linestring functions:

# Generates the lists of coordinate alterations needed to search around a candidate pixel
# up to a certain spacing away from the candidate pixel
def grid(spacing):
    search = []

    for r in range(-spacing, spacing+1, 1):
        for c in range(-spacing, spacing+1, 1):
            if r == -spacing or r == spacing or c == -spacing or c == spacing:
                search.append((r, c))
    return search


# From the current pixel, follow along the white pixels and add those pixels to a line
# If multiple paths appear, create a new job
def follow(jobs, image, old_location):

    # Create "linestring"
    linestring = []
    old_direction = -1

    # Follow the line while there are still pixels
    done = False
    while not done:
        found = 0
        row = old_location[0]
        column = old_location[1]
        # Add the current pixel to the linestring
        linestring.append(old_location)
        # Search 1, 2, etc. pixels away until a pixel is found or the limit is reached
        for reach, search in enumerate(searches):
            # Look at the 8 pixels around the center
            for direction, pixel in enumerate(search):
                r1 = pixel[0]
                c1 = pixel[1]
                # If the search goes outside the bounds
                if row+r1 < 0 or row+r1 > len(image)-1 or column+c1 < 0 or column+c1 > len(image[0])-1:
                    continue
                # If the pixel looked at is white
                if image[row+r1][column+c1] == white:
                    found += 1
                    # If the search is looking 1 pixel away, check direction
                    if reach == 0:
                        new_direction = direction
                    # Otherwise, assume a direction change
                    else:
                        new_direction = -1
                    # If this is the first pixel found
                    if found == 1:
                        # If the direction of travel has not changed
                        if old_direction == new_direction:
                            # Replace the last pixel in the linestring
                            linestring.pop()
                        old_direction = new_direction
                        # Select new pixel to investigate
                        new_location = (row+r1, column+c1)
                        # Black out the current pixel
                        row = old_location[0]
                        column = old_location[1]
                        image[row][column] = black
                    # If another pixel is found
                    else:
                        # Add another job to search for that linestring
                        job = (image, old_location)
                        jobs.put(job)

            # If a pixel is found at this search level, stop searching
            if found > 0:
                break
        # If no surrounding pixels were white
        if found == 0:
            done = True
            # Black out the current pixel
            row = old_location[0]
            column = old_location[1]
            image[row][column] = black
        # Update the pixel to investigate if there is one
        if not done:
            old_location = new_location

    return linestring


# Given a skeletonized image, finds the linestrings that define it
# If the linestrings have gaps in them less than or equal to spacing, they still count
def skeleton2linestrings(image, spacing):

    linestrings = []

    # Iterate over all pixels
    for row in range(len(image)):
        for column in range(len(image[0])):
            # If the pixel is white
            if image[row][column] == white:
                # Create empty queue of jobs
                jobs = queue.Queue()

                # Add this pixel to the jobs queue
                old_location = (row, column)
                job = (image, old_location)
                jobs.put(job)

                while not jobs.empty():
                    job = jobs.get()
                    linestring = follow(jobs, job[0], job[1])

                    # Remove all the linestrings that are just a single point
                    if len(linestring) > 1:
                        linestrings.append(linestring)

    return linestrings

from skimage.morphology import dilation, closing, erosion
from skimage.morphology import disk

def mask2linestrings(image, spacing, image_id):
    
    disk_size = 20
    select = disk(disk_size)

    # Dilation
    image = dilation(image, select)
    
    # Skeletonize the image
    skeletonized = morphology.medial_axis(image)
    skeletonized = skeletonized.astype(np.uint8)
    skeletonized *= 255

    # Find the linestrings from the image
    linestrings = skeleton2linestrings(skeletonized, spacing)
    return linestrings


def write_csv_predict(images, image_ids, spacing, csv_filename):
    with open(csv_filename, 'w') as csv_predict:
        csv_predict.write("ImageId,WKT_Pix\n")

        for image, image_id in zip(images, image_ids):
            binary_image = image
            binary_image = np.swapaxes(binary_image, 0, 2)
            binary_image = np.swapaxes(binary_image, 0, 1)
            binary_image = ski_transform.resize(binary_image, (1300,1300))
            binary_image = (binary_image > 0.5).astype(np.uint8)
            binary_image = np.swapaxes(binary_image, 0, 2)
            binary_image = np.swapaxes(binary_image, 1, 2)
            binary_image = np.squeeze(binary_image)
            cv2.imwrite("/data/output/outputImages/withDilation/{}.tif".format(image_id),binary_image)
            linestrings = mask2linestrings(binary_image, spacing, image_id)

            if len(linestrings) == 0:
                lines = ["{},LINESTRING EMPTY".format(image_id)]
            else:
                lines = []

                for linestring in linestrings:
                    line = "{},\"LINESTRING (".format(image_id)
                    for i, coordinate in enumerate(linestring):
                        #line += "{} {}".format((coordinate[1]/256.0)*1300, (coordinate[0]/256.0)*1300)
                        line += "{} {}".format(coordinate[1], coordinate[0])
                        if i != (len(linestring)-1):
                            line += ", "
                        else:
                            line += ")\""
                    lines.append(line)
            for line in lines:
                csv_predict.write(line+"\n")


# ---------------------------------------------------------
# main


# def _get_model_parameter(area_id):
#     prefix = area_id_to_prefix(area_id)
#     fn_hist = FMT_VALMODEL_EVALTHHIST.format(prefix)
#     best_row = pd.read_csv(fn_hist).sort_values(
#         by='fscore',
#         ascending=False,
#     ).iloc[0]
#
#     param = dict(
#         fn_epoch=int(best_row['zero_base_epoch']),
#         min_poly_area=int(best_row['min_area_th']),
#     )
#     return param


# def _internal_test_predict_best_param(area_id,
#                                       save_pred=True):
#     prefix = area_id_to_prefix(area_id)
#     param = _get_model_parameter(area_id)
#     epoch = param['fn_epoch']
#     min_th = param['min_poly_area']
#
#     # Prediction phase
#     logger.info("Prediction phase: {}".format(prefix))
#
#     X_mean = get_mul_mean_image(area_id)
#
#     # Load model weights
#     # Predict and Save prediction result
#     fn = FMT_TESTPRED_PATH.format(prefix)
#     fn_model = FMT_VALMODEL_PATH.format(prefix + '_{epoch:02d}')
#     fn_model = fn_model.format(epoch=epoch)
#     model = get_unet()
#     model.load_weights(fn_model)
#
#     fn_test = FMT_TEST_IMAGELIST_PATH.format(prefix=prefix)
#     df_test = pd.read_csv(fn_test, index_col='ImageId')
#
#     y_pred = model.predict_generator(
#         generate_test_batch(
#             area_id,
#             batch_size=64,
#             immean=X_mean,
#             enable_tqdm=True,
#         ),
#         val_samples=len(df_test) * 9,
#     )
#     del model
#
#     # Save prediction result
#     if save_pred:
#         with tb.open_file(fn, 'w') as f:
#             atom = tb.Atom.from_dtype(y_pred.dtype)
#             filters = tb.Filters(complib='blosc', complevel=9)
#             ds = f.create_carray(f.root, 'pred', atom, y_pred.shape,
#                                  filters=filters)
#             ds[:] = y_pred
#
#     return y_pred


def _internal_test_2(area_id):
    prefix = area_id_to_prefix(area_id)
    save_pred=True
    epoch = 9

    # Prediction phase
    logger.info("Prediction phase: {}".format(prefix))

    X_mean = get_mul_mean_image(area_id)

    # Predict and Save prediction result
    fn = FMT_TESTPRED_PATH.format(prefix)
    #fn_model = FMT_VALMODEL_PATH.format(prefix + '_{epoch:02d}')
    #fn_model = fn_model.format(epoch=epoch)

    if os.path.isfile(FMT_VALMODEL_LAST_PATH.format(prefix)):
        fn_model = FMT_VALMODEL_LAST_PATH.format(prefix)
    elif os.path.isfile(FMT_VALMODEL_PATH.format(prefix + '_{epoch:02d}')):
        fn_model = FMT_VALMODEL_PATH.format(prefix + '_{epoch:02d}')
        fn_model = fn_model.format(epoch=epoch)
    else:
        print("ERROR: Trained model not found.")
        exit(1)

    model = get_unet()
    model.load_weights(fn_model)

    fn_test = FMT_TEST_IMAGELIST_PATH.format(prefix=prefix)
    df_test = pd.read_csv(fn_test, index_col='ImageId')

    y_pred = model.predict_generator(
        generate_test_batch(
            area_id,
            batch_size=64,
            immean=X_mean,
            enable_tqdm=True,
        ),
        val_samples=len(df_test)
    )
    del model

    # Save prediction result in a dataframe
    if save_pred:
        with tb.open_file(fn, 'w') as f:
            atom = tb.Atom.from_dtype(y_pred.dtype)
            filters = tb.Filters(complib='blosc', complevel=9)
            ds = f.create_carray(f.root, 'pred', atom, y_pred.shape,
                                 filters=filters)
            ds[:] = y_pred

    image_ids = df_test.index.tolist()
    spacing = 1

    #time_now = time.strftime("%Y%m%d-%H%M%S")
    #fn_out = FMT_TESTLINE_PATH.format(prefix +'_'+ time_now +'_')
    fn_out = FMT_TESTLINE_PATH.format(prefix)
    write_csv_predict(y_pred, image_ids, spacing, fn_out)


# def _internal_test(area_id):
#     prefix = area_id_to_prefix(area_id)
#     y_pred = _internal_test_predict_best_param(area_id, save_pred=False)
#
#     fn_test = FMT_TEST_IMAGELIST_PATH.format(prefix=prefix)
#     df_test = pd.read_csv(fn_test, index_col='ImageId')
#     image_ids = df_test.index.tolist()
#     spacing = 1
#
#     fn_out = FMT_TESTPOLY_PATH.format(prefix)
#
#     write_csv_predict(y_pred, image_ids, spacing, fn_out)
#
#     """
#     # Postprocessing phase
#     logger.info("Postprocessing phase")
#     # if not Path(FMT_VALTESTPOLY_PATH.format(prefix)).exists():
#     fn_test = FMT_TEST_IMAGELIST_PATH.format(prefix=prefix)
#     df_test = pd.read_csv(fn_test, index_col='ImageId')
#     fn = FMT_TESTPRED_PATH.format(prefix)
#     with tb.open_file(fn, 'r') as f:
#         y_pred = np.array(f.get_node('/pred'))
#
#     fn_out = FMT_TESTPOLY_PATH.format(prefix)
#     with open(fn_out, 'w') as f:
#         f.write("ImageId,WKT_Pix\n")
#         for idx, image_id in enumerate(df_test.index.tolist()):
#             pred_values = np.zeros((1300, 1300))
#             pred_count = np.zeros((1300, 1300))
#             for slice_pos in range(9):
#                 slice_idx = idx * 9 + slice_pos
#
#                 pos_j = int(math.floor(slice_pos / 3.0))
#                 pos_i = int(slice_pos % 3)
#                 x0 = STRIDE_SZ * pos_i
#                 y0 = STRIDE_SZ * pos_j
#                 pred_values[x0:x0+INPUT_SIZE, y0:y0+INPUT_SIZE] += (
#                     y_pred[slice_idx][0]
#                 )
#                 pred_count[x0:x0+INPUT_SIZE, y0:y0+INPUT_SIZE] += 1
#             pred_values = pred_values / pred_count
#
#             linstrings = mask2linestrings(pred_values, spacing)
#             if len(linstrings) > 0:
#                 for i, row in linstrings.iterrows():
#                     line = "{},{}\n".format(
#                         image_id,
#                         row.wkt)
#                     f.write(line)
#             else:
#                 f.write("{},{},{},0\n".format(
#                     image_id,
#                     -1,
#                     "EMPTY"))
#     """


# def _internal_validate_predict_best_param(area_id,
#                                           enable_tqdm=False):
#     """
#     best param  で valtest  の prediction proba を return する
#     y_pred は保存しない
#
#     (used from ensemble model)
#     """
#     param = _get_model_parameter(area_id)
#     epoch = param['fn_epoch']
#     y_pred = _internal_validate_predict(
#         area_id,
#         epoch=epoch,
#         save_pred=False,
#         enable_tqdm=enable_tqdm)
#
#     return y_pred


# def _internal_validate_predict(area_id,
#                                epoch=3,
#                                save_pred=True,
#                                enable_tqdm=False):
#     prefix = area_id_to_prefix(area_id)
#     X_mean = get_mul_mean_image(area_id)
#
#     # Load model weights
#     # Predict and Save prediction result
#     fn_model = FMT_VALMODEL_PATH.format(prefix + '_{epoch:02d}')
#     fn_model = fn_model.format(epoch=epoch)
#     model = get_unet()
#     model.load_weights(fn_model)
#
#     fn_test = FMT_VALTEST_IMAGELIST_PATH.format(prefix=prefix)
#     df_test = pd.read_csv(fn_test, index_col='ImageId')
#
#     y_pred = model.predict_generator(
#         generate_valtest_batch(
#             area_id,
#             batch_size=64,
#             immean=X_mean,
#             enable_tqdm=enable_tqdm,
#         ),
#         val_samples=len(df_test) * 9,
#     )
#     del model
#
#     # Save prediction result
#     if save_pred:
#         fn = FMT_VALTESTPRED_PATH.format(prefix)
#         with tb.open_file(fn, 'w') as f:
#             atom = tb.Atom.from_dtype(y_pred.dtype)
#             filters = tb.Filters(complib='blosc', complevel=9)
#             ds = f.create_carray(f.root,
#                                  'pred',
#                                  atom,
#                                  y_pred.shape,
#                                  filters=filters)
#             ds[:] = y_pred
#     return y_pred


# def _internal_validate_fscore_wo_pred_file(area_id,
#                                            epoch=3,
#                                            min_th=MIN_POLYGON_AREA,
#                                            enable_tqdm=False):
#     prefix = area_id_to_prefix(area_id)
#
#     # Prediction phase
#     logger.info("Prediction phase")
#     y_pred = _internal_validate_predict(
#         area_id,
#         save_pred=False,
#         epoch=epoch,
#         enable_tqdm=enable_tqdm)
#
#     fn_test = FMT_VALTEST_IMAGELIST_PATH.format(prefix=prefix)
#     df_test = pd.read_csv(fn_test, index_col='ImageId')
#     image_ids = df_test.index.tolist()
#     spacing = 1
#
#     fn_out = FMT_VALTESTPOLY_PATH.format(prefix)
#
#     write_csv_predict(y_pred, image_ids, spacing, fn_out)
#
#     """
#     # Postprocessing phase
#     logger.info("Postprocessing phase")
#     fn_test = FMT_VALTEST_IMAGELIST_PATH.format(prefix=prefix)
#     df_test = pd.read_csv(fn_test, index_col='ImageId')
#
#     fn_out = FMT_VALTESTPOLY_PATH.format(prefix)
#     with open(fn_out, 'w') as f:
#         f.write("ImageId,WKT_Pix\n")
#         test_list = df_test.index.tolist()
#         iterator = enumerate(test_list)
#
#         for idx, image_id in tqdm.tqdm(iterator, total=len(test_list)):
#             pred_values = np.zeros((1300, 1300))
#             pred_count = np.zeros((1300, 1300))
#             for slice_pos in range(9):
#                 slice_idx = idx * 9 + slice_pos
#
#                 pos_j = int(math.floor(slice_pos / 3.0))
#                 pos_i = int(slice_pos % 3)
#                 x0 = STRIDE_SZ * pos_i
#                 y0 = STRIDE_SZ * pos_j
#                 pred_values[x0:x0+INPUT_SIZE, y0:y0+INPUT_SIZE] += (
#                     y_pred[slice_idx][0]
#                 )
#                 pred_count[x0:x0+INPUT_SIZE, y0:y0+INPUT_SIZE] += 1
#             pred_values = pred_values / pred_count
#
#             linstrings = mask2linestrings(pred_values, spacing)
#             if len(linstrings) > 0:
#                 for i, row in linstrings.iterrows():
#                     line = "{},{}\n".format(
#                         image_id,
#                         row.wkt)
#                     f.write(line)
#             else:
#                 f.write("{},{},{},0\n".format(
#                     image_id,
#                     -1,
#                     "EMPTY"))
#     """
#
#
#     fn_test = FMT_VALTEST_IMAGELIST_PATH.format(prefix=prefix)
#     df_test = pd.read_csv(fn_test)
#     image_ids = df_test.ImageId.unique()
#     spacing = 1
#
#     fn_out = FMT_VALTESTTRUTH_PATH.format(prefix)
#
#     write_csv_predict(y_pred, image_ids, spacing, fn_out)
#
#     """
#     # ------------------------
#     # Validation solution file
#     logger.info("Validation solution file")
#     fn_true = FMT_TRAIN_SUMMARY_PATH.format(prefix=prefix)
#     df_true = pd.read_csv(fn_true)
#     # # Remove prefix "PAN_"
#     # df_true.loc[:, 'ImageId'] = df_true.ImageId.str[4:]
#
#     fn_test = FMT_VALTEST_IMAGELIST_PATH.format(prefix=prefix)
#     df_test = pd.read_csv(fn_test)
#     df_test_image_ids = df_test.ImageId.unique()
#
#     fn_out = FMT_VALTESTTRUTH_PATH.format(prefix)
#     with open(fn_out, 'w') as f:
#         f.write("ImageId,WKT_Pix\n")
#         df_true = df_true[df_true.ImageId.isin(df_test_image_ids)]
#         for idx, r in df_true.iterrows():
#             f.write("{},{},\"{}\",{:.6f}\n".format(
#                 r.ImageId,
#                 r.BuildingId,
#                 r.WKT_Pix,
#                 1.0))
#      """


# def _internal_validate_fscore(area_id,
#                               epoch=3,
#                               predict=True,
#                               min_th=MIN_POLYGON_AREA,
#                               enable_tqdm=False):
#     prefix = area_id_to_prefix(area_id)
#
#     # Prediction phase
#     logger.info("Prediction phase")
#     if predict:
#         _internal_validate_predict(
#             area_id,
#             epoch=epoch,
#             enable_tqdm=enable_tqdm)
#
#
#     fn_test = FMT_VALTEST_IMAGELIST_PATH.format(prefix=prefix)
#     df_test = pd.read_csv(fn_test, index_col='ImageId')
#     image_ids = df_test.index.tolist()
#     spacing = 1
#
#     fn_out = FMT_VALTESTPOLY_PATH.format(prefix)
#
#     write_csv_predict(y_pred, image_ids, spacing, fn_out)
#
#     """
#     # Postprocessing phase
#     logger.info("Postprocessing phase")
#     fn_test = FMT_VALTEST_IMAGELIST_PATH.format(prefix=prefix)
#     df_test = pd.read_csv(fn_test, index_col='ImageId')
#     fn = FMT_VALTESTPRED_PATH.format(prefix)
#     with tb.open_file(fn, 'r') as f:
#         y_pred = np.array(f.get_node('/pred'))
#
#     fn_out = FMT_VALTESTPOLY_PATH.format(prefix)
#     with open(fn_out, 'w') as f:
#         f.write("ImageId,WKT_Pix\n")
#         test_list = df_test.index.tolist()
#         iterator = enumerate(test_list)
#
#         for idx, image_id in tqdm.tqdm(iterator, total=len(test_list)):
#             pred_values = np.zeros((1300, 1300))
#             pred_count = np.zeros((1300, 1300))
#             for slice_pos in range(9):
#                 slice_idx = idx * 9 + slice_pos
#
#                 pos_j = int(math.floor(slice_pos / 3.0))
#                 pos_i = int(slice_pos % 3)
#                 x0 = STRIDE_SZ * pos_i
#                 y0 = STRIDE_SZ * pos_j
#                 pred_values[x0:x0+INPUT_SIZE, y0:y0+INPUT_SIZE] += (
#                     y_pred[slice_idx][0]
#                 )
#                 pred_count[x0:x0+INPUT_SIZE, y0:y0+INPUT_SIZE] += 1
#             pred_values = pred_values / pred_count
#
#             linstrings = mask2linestrings(pred_values, spacing)
#             if len(linstrings) > 0:
#                 for i, row in linstrings.iterrows():
#                     line = "{},{}\n".format(
#                         image_id,
#                         row.wkt)
#                     f.write(line)
#             else:
#                 f.write("{},{},{},0\n".format(
#                     image_id,
#                     -1,
#                     "EMPTY"))
#
#     """
#
#
#     fn_test = FMT_VALTEST_IMAGELIST_PATH.format(prefix=prefix)
#     df_test = pd.read_csv(fn_test)
#     image_ids = df_test.ImageId.unique()
#     spacing = 1
#
#     fn_out = FMT_VALTESTTRUTH_PATH.format(prefix)
#
#     write_csv_predict(y_pred, image_ids, spacing, fn_out)
#
#     """
#     # ------------------------
#     # Validation solution file
#     logger.info("Validation solution file")
#     # if not Path(FMT_VALTESTTRUTH_PATH.format(prefix)).exists():
#     if True:
#         fn_true = FMT_TRAIN_SUMMARY_PATH.format(prefix=prefix)
#         df_true = pd.read_csv(fn_true)
#         # # Remove prefix "PAN_"
#         # df_true.loc[:, 'ImageId'] = df_true.ImageId.str[4:]
#
#         fn_test = FMT_VALTEST_IMAGELIST_PATH.format(prefix=prefix)
#         df_test = pd.read_csv(fn_test)
#         df_test_image_ids = df_test.ImageId.unique()
#
#         fn_out = FMT_VALTESTTRUTH_PATH.format(prefix)
#         with open(fn_out, 'w') as f:
#             f.write("ImageId,WKT_Pix\n")
#             df_true = df_true[df_true.ImageId.isin(df_test_image_ids)]
#             for idx, r in df_true.iterrows():
#                 f.write("{},\"{}\",{:.6f}\n".format(
#                     r.ImageId,
#                     r.WKT_Pix,
#                     1.0))
#     """


# def mask_to_poly(mask, min_polygon_area_th=MIN_POLYGON_AREA):
#     mask = (mask > 0.5).astype(np.uint8)
#     shapes = rasterio.features.shapes(mask.astype(np.int16), mask > 0)
#     poly_list = []
#     mp = shapely.ops.cascaded_union(
#         shapely.geometry.MultiPolygon([
#             shapely.geometry.shape(shape)
#             for shape, value in shapes
#         ]))
#
#     if isinstance(mp, shapely.geometry.Polygon):
#         df = pd.DataFrame({
#             'area_size': [mp.area],
#             'poly': [mp],
#         })
#     else:
#         df = pd.DataFrame({
#             'area_size': [p.area for p in mp],
#             'poly': [p for p in mp],
#         })
#
#     df = df[df.area_size > min_polygon_area_th].sort_values(
#         by='area_size', ascending=False)
#     df.loc[:, 'wkt'] = df.poly.apply(lambda x: shapely.wkt.dumps(
#         x, rounding_precision=0))
#     df.loc[:, 'bid'] = list(range(1, len(df) + 1))
#     df.loc[:, 'area_ratio'] = df.area_size / df.area_size.max()
#     return df


def jaccard_coef(y_true, y_pred):
    ## Used in the unet code
    smooth = 1e-12
    intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return K.mean(jac)


def jaccard_coef_int(y_true, y_pred):
    ## Used in the unet code
    smooth = 1e-12
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    intersection = K.sum(y_true * y_pred_pos, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred_pos, axis=[0, -1, -2])
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return K.mean(jac)


def generate_test_batch(area_id,
                        batch_size=64,
                        immean=None,
                        enable_tqdm=False):
    ## Used in the test function
    prefix = area_id_to_prefix(area_id)
    df_test = pd.read_csv(FMT_TEST_IMAGELIST_PATH.format(prefix=prefix))
    fn_im = FMT_TEST_MUL_STORE.format(prefix)

    slice_id_list = []
    for idx, row in df_test.iterrows():
        slice_id = row.ImageId
        slice_id_list.append(slice_id)

    if enable_tqdm:
        pbar = tqdm.tqdm(total=len(slice_id_list))

    while 1:
        total_sz = len(slice_id_list)
        n_batch = int(math.floor(total_sz / batch_size) + 1)
        with tb.open_file(fn_im, 'r') as f_im:
            for i_batch in range(n_batch):
                target_slice_ids = slice_id_list[
                    i_batch*batch_size:(i_batch+1)*batch_size
                ]
                if len(target_slice_ids) == 0:
                    continue

                X_test = []
                y_test = []
                for slice_id in target_slice_ids:
                    im = np.array(f_im.get_node('/' + slice_id))
                    im = np.swapaxes(im, 0, 2)
                    im = np.swapaxes(im, 1, 2)
                    X_test.append(im)
                    mask = np.zeros((INPUT_SIZE, INPUT_SIZE)).astype(np.uint8)
                    y_test.append(mask)
                X_test = np.array(X_test)
                y_test = np.array(y_test)
                y_test = y_test.reshape((-1, 1, INPUT_SIZE, INPUT_SIZE))

                if immean is not None:
                    X_test = X_test - immean

                if enable_tqdm:
                    pbar.update(y_test.shape[0])

                yield (X_test, y_test)

        if enable_tqdm:
            pbar.close()


# def generate_valtest_batch(area_id,
#                            batch_size=8,
#                            immean=None,
#                            enable_tqdm=False):
#     prefix = area_id_to_prefix(area_id)
#     df_train = pd.read_csv(FMT_VALTEST_IMAGELIST_PATH.format(prefix=prefix))
#     fn_im = FMT_VALTEST_MUL_STORE.format(prefix)
#     fn_mask = FMT_VALTEST_MASK_STORE.format(prefix)
#
#     slice_id_list = []
#     for idx, row in df_train.iterrows():
#         for slice_pos in range(9):
#             slice_id = row.ImageId + '_' + str(slice_pos)
#             slice_id_list.append(slice_id)
#
#     if enable_tqdm:
#         pbar = tqdm.tqdm(total=len(slice_id_list))
#
#     while 1:
#         total_sz = len(slice_id_list)
#         n_batch = int(math.floor(total_sz / batch_size) + 1)
#         with tb.open_file(fn_im, 'r') as f_im,\
#                 tb.open_file(fn_mask, 'r') as f_mask:
#             for i_batch in range(n_batch):
#                 target_slice_ids = slice_id_list[
#                     i_batch*batch_size:(i_batch+1)*batch_size
#                 ]
#                 if len(target_slice_ids) == 0:
#                     continue
#
#                 X_train = []
#                 y_train = []
#                 for slice_id in target_slice_ids:
#                     im = np.array(f_im.get_node('/' + slice_id))
#                     im = np.swapaxes(im, 0, 2)
#                     im = np.swapaxes(im, 1, 2)
#
#                     X_train.append(im)
#                     mask = np.array(f_mask.get_node('/' + slice_id))
#                     mask = (mask > 0).astype(np.uint8)
#                     y_train.append(mask)
#                 X_train = np.array(X_train)
#                 y_train = np.array(y_train)
#                 y_train = y_train.reshape((-1, 1, INPUT_SIZE, INPUT_SIZE))
#
#                 if immean is not None:
#                     X_train = X_train - immean
#
#                 if enable_tqdm:
#                     pbar.update(y_train.shape[0])
#
#                 yield (X_train, y_train)
#
#         if enable_tqdm:
#             pbar.close()


def generate_valtrain_batch(area_id, batch_size=8, immean=None):
    """
    Generate the training batches for the fit_generator
    """
    prefix = area_id_to_prefix(area_id)
    df_train = pd.read_csv(FMT_VALTRAIN_IMAGELIST_PATH.format(prefix=prefix))
    fn_im = FMT_VALTRAIN_MUL_STORE.format(prefix)
    fn_mask = FMT_VALTRAIN_MASK_STORE.format(prefix)

    """
    slice_id_list = []
    for idx, row in df_train.iterrows():
        for slice_pos in range(9):
            slice_id = row.ImageId + '_' + str(slice_pos)
            slice_id_list.append(slice_id)
    np.random.shuffle(slice_id_list)
    """

    image_id_list = []
    for idx, row in df_train.iterrows():
        image_id = row.ImageId
        image_id_list.append(image_id)
    np.random.shuffle(image_id_list)

    while 1:
        total_sz = len(image_id_list)
        n_batch = int(math.floor(total_sz / batch_size) + 1)
        with tb.open_file(fn_im, 'r') as f_im,\
                tb.open_file(fn_mask, 'r') as f_mask:
            for i_batch in range(n_batch):
                target_image_ids = image_id_list[
                    i_batch*batch_size:(i_batch+1)*batch_size
                ]
                if len(target_image_ids) == 0:
                    continue

                X_train = []
                y_train = []
                for image_id in target_image_ids:
                    im = np.array(f_im.get_node('/' + image_id))
                    im = np.swapaxes(im, 0, 2)
                    im = np.swapaxes(im, 1, 2)
                    X_train.append(im)
                    mask = np.array(f_mask.get_node('/' + image_id))
                    mask = (mask > 0).astype(np.uint8)
                    y_train.append(mask)
                X_train = np.array(X_train)
                y_train = np.array(y_train)
                y_train = y_train.reshape((-1, 1, INPUT_SIZE, INPUT_SIZE))

                if immean is not None:
                    X_train = X_train - immean

                yield (X_train, y_train)



def get_unet():
    conv_params = dict(activation='relu', border_mode='same')
    merge_params = dict(mode='concat', concat_axis=1)
    inputs = Input((8, 256, 256))
    conv1 = Convolution2D(32, 3, 3, **conv_params)(inputs)
    conv1 = Convolution2D(32, 3, 3, **conv_params)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(64, 3, 3, **conv_params)(pool1)
    conv2 = Convolution2D(64, 3, 3, **conv_params)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(128, 3, 3, **conv_params)(pool2)
    conv3 = Convolution2D(128, 3, 3, **conv_params)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(256, 3, 3, **conv_params)(pool3)
    conv4 = Convolution2D(256, 3, 3, **conv_params)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(512, 3, 3, **conv_params)(pool4)
    conv5 = Convolution2D(512, 3, 3, **conv_params)(conv5)

    up6 = merge_l([UpSampling2D(size=(2, 2))(conv5), conv4], **merge_params)
    conv6 = Convolution2D(256, 3, 3, **conv_params)(up6)
    conv6 = Convolution2D(256, 3, 3, **conv_params)(conv6)

    up7 = merge_l([UpSampling2D(size=(2, 2))(conv6), conv3], **merge_params)
    conv7 = Convolution2D(128, 3, 3, **conv_params)(up7)
    conv7 = Convolution2D(128, 3, 3, **conv_params)(conv7)

    up8 = merge_l([UpSampling2D(size=(2, 2))(conv7), conv2], **merge_params)
    conv8 = Convolution2D(64, 3, 3, **conv_params)(up8)
    conv8 = Convolution2D(64, 3, 3, **conv_params)(conv8)

    up9 = merge_l([UpSampling2D(size=(2, 2))(conv8), conv1], **merge_params)
    conv9 = Convolution2D(32, 3, 3, **conv_params)(up9)
    conv9 = Convolution2D(32, 3, 3, **conv_params)(conv9)

    conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(conv9)

    optimizer = SGD(lr=0.002, momentum=0.9, nesterov=True)
    model = Model(input=inputs, output=conv10)
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy', jaccard_coef, jaccard_coef_int])
    return model


# def get_mean_image(area_id):
#     # For the RGB images
#     prefix = area_id_to_prefix(area_id)
#
#     with tb.open_file(FMT_IMMEAN.format(prefix), 'r') as f:
#         im_mean = np.array(f.get_node('/immean'))
#     return im_mean


def get_mul_mean_image(area_id):
    prefix = area_id_to_prefix(area_id)

    with tb.open_file(FMT_MULMEAN.format(prefix), 'r') as f:
        im_mean = np.array(f.get_node('/mulmean'))
    return im_mean


# def get_train_data(area_id):
#     prefix = area_id_to_prefix(area_id)
#     fn_train = FMT_TRAIN_IMAGELIST_PATH.format(prefix=prefix)
#     df_train = pd.read_csv(fn_train)
#
#     X_train = []
#     fn_im = FMT_TRAIN_MUL_STORE.format(prefix)
#     with tb.open_file(fn_im, 'r') as f:
#         for idx, image_id in enumerate(df_train.ImageId.tolist()):
#             for slice_pos in range(9):
#                 slice_id = image_id + '_' + str(slice_pos)
#
#                 im = np.array(f.get_node('/' + slice_id))
#                 im = np.swapaxes(im, 0, 2)
#                 im = np.swapaxes(im, 1, 2)
#                 X_train.append(im)
#     X_train = np.array(X_train)
#
#     y_train = []
#     fn_mask = FMT_TRAIN_MASK_STORE.format(prefix)
#     with tb.open_file(fn_mask, 'r') as f:
#         for idx, image_id in enumerate(df_train.ImageId.tolist()):
#             for slice_pos in range(9):
#                 slice_id = image_id + '_' + str(slice_pos)
#
#                 mask = np.array(f.get_node('/' + slice_id))
#                 mask = (mask > 0.5).astype(np.uint8)
#                 y_train.append(mask)
#     y_train = np.array(y_train)
#     y_train = y_train.reshape((-1, 1, INPUT_SIZE, INPUT_SIZE))
#
#     return X_train, y_train


# def get_test_data(area_id):
#     prefix = area_id_to_prefix(area_id)
#     fn_test = FMT_TEST_IMAGELIST_PATH.format(prefix=prefix)
#     df_test = pd.read_csv(fn_test)
#
#     X_test = []
#     fn_im = FMT_TEST_MUL_STORE.format(prefix)
#     with tb.open_file(fn_im, 'r') as f:
#         for idx, image_id in enumerate(df_test.ImageId.tolist()):
#             for slice_pos in range(9):
#                 slice_id = image_id + '_' + str(slice_pos)
#
#                 im = np.array(f.get_node('/' + slice_id))
#                 im = np.swapaxes(im, 0, 2)
#                 im = np.swapaxes(im, 1, 2)
#                 X_test.append(im)
#     X_test = np.array(X_test)
#
#     return X_test


def get_valtest_data(area_id):
    ## Used in training
    prefix = area_id_to_prefix(area_id)
    fn_test = FMT_VALTEST_IMAGELIST_PATH.format(prefix=prefix)
    df_test = pd.read_csv(fn_test)

    X_val = []
    fn_im = FMT_VALTEST_MUL_STORE.format(prefix)
    with tb.open_file(fn_im, 'r') as f:
        for idx, image_id in enumerate(df_test.ImageId.tolist()):
            #for slice_pos in range(9):
                #slice_id = image_id + '_' + str(slice_pos)
            im = np.array(f.get_node('/' + image_id))
            im = np.swapaxes(im, 0, 2)
            im = np.swapaxes(im, 1, 2)
            X_val.append(im)
    X_val = np.array(X_val)

    y_val = []
    fn_mask = FMT_VALTEST_MASK_STORE.format(prefix)
    with tb.open_file(fn_mask, 'r') as f:
        for idx, image_id in enumerate(df_test.ImageId.tolist()):
            #for slice_pos in range(9):
                #slice_id = image_id + '_' + str(slice_pos)
             mask = np.array(f.get_node('/' + image_id))
             mask = (mask > 0.5).astype(np.uint8)
             y_val.append(mask)
    y_val = np.array(y_val)
    y_val = y_val.reshape((-1, 1, INPUT_SIZE, INPUT_SIZE))

    return X_val, y_val


# def _get_valtrain_data_head(area_id):
#     prefix = area_id_to_prefix(area_id)
#     fn_train = FMT_VALTRAIN_IMAGELIST_PATH.format(prefix=prefix)
#     df_train = pd.read_csv(fn_train)
#
#     X_val = []
#     fn_im = FMT_VALTRAIN_MUL_STORE.format(prefix)
#     with tb.open_file(fn_im, 'r') as f:
#         for idx, image_id in enumerate(df_train.ImageId.tolist()):
#             slice_pos = 5
#             slice_id = image_id + '_' + str(slice_pos)
#             im = np.array(f.get_node('/' + slice_id))
#             im = np.swapaxes(im, 0, 2)
#             im = np.swapaxes(im, 1, 2)
#             X_val.append(im)
#     X_val = np.array(X_val)
#
#     y_val = []
#     fn_mask = FMT_VALTRAIN_MASK_STORE.format(prefix)
#     with tb.open_file(fn_mask, 'r') as f:
#         for idx, image_id in enumerate(df_train.ImageId.tolist()):
#             slice_pos = 5
#             slice_id = image_id + '_' + str(slice_pos)
#             mask = np.array(f.get_node('/' + slice_id))
#             mask = (mask > 0.5).astype(np.uint8)
#             y_val.append(mask)
#     y_val = np.array(y_val)
#     y_val = y_val.reshape((-1, 1, INPUT_SIZE, INPUT_SIZE))
#
#     return X_val, y_val


# def get_valtrain_data(area_id):
#     prefix = area_id_to_prefix(area_id)
#     fn_train = FMT_VALTRAIN_IMAGELIST_PATH.format(prefix=prefix)
#     df_train = pd.read_csv(fn_train)
#
#     X_val = []
#     fn_im = FMT_VALTRAIN_MUL_STORE.format(prefix)
#     with tb.open_file(fn_im, 'r') as f:
#         for idx, image_id in enumerate(df_train.ImageId.tolist()):
#             for slice_pos in range(9):
#                 slice_id = image_id + '_' + str(slice_pos)
#                 im = np.array(f.get_node('/' + slice_id))
#                 im = np.swapaxes(im, 0, 2)
#                 im = np.swapaxes(im, 1, 2)
#                 X_val.append(im)
#     X_val = np.array(X_val)
#
#     y_val = []
#     fn_mask = FMT_VALTRAIN_MASK_STORE.format(prefix)
#     with tb.open_file(fn_mask, 'r') as f:
#         for idx, image_id in enumerate(df_train.ImageId.tolist()):
#             for slice_pos in range(9):
#                 slice_id = image_id + '_' + str(slice_pos)
#                 mask = np.array(f.get_node('/' + slice_id))
#                 mask = (mask > 0.5).astype(np.uint8)
#                 y_val.append(mask)
#     y_val = np.array(y_val)
#     y_val = y_val.reshape((-1, 1, INPUT_SIZE, INPUT_SIZE))
#
#     return X_val, y_val


# def __load_band_cut_th(band_fn, bandsz=3):
#     df = pd.read_csv(band_fn, index_col='area_id')
#     all_band_cut_th = {area_id: {} for area_id in range(2, 6)}
#     for area_id, row in df.iterrows():
#         for chan_i in range(bandsz):
#             all_band_cut_th[area_id][chan_i] = dict(
#                 min=row['chan{}_min'.format(chan_i)],
#                 max=row['chan{}_max'.format(chan_i)],
#             )
#     return all_band_cut_th


# def get_slice_3chan_test_im(image_id, band_cut_th):
#     fn = test_image_id_to_path(image_id)
#     with rasterio.open(fn, 'r') as f:
#         values = f.read().astype(np.float32)
#         for chan_i in range(3):
#             min_val = band_cut_th[chan_i]['min']
#             max_val = band_cut_th[chan_i]['max']
#             values[chan_i] = np.clip(values[chan_i], min_val, max_val)
#             values[chan_i] = (values[chan_i] - min_val) / (max_val - min_val)
#     values = np.swapaxes(values, 0, 2)
#     values = np.swapaxes(values, 0, 1)
#     assert values.shape == (650, 650, 3)
#
#     for slice_pos in range(9):
#         pos_j = int(math.floor(slice_pos / 3.0))
#         pos_i = int(slice_pos % 3)
#         x0 = STRIDE_SZ * pos_i
#         y0 = STRIDE_SZ * pos_j
#         im = values[x0:x0+INPUT_SIZE, y0:y0+INPUT_SIZE]
#         assert im.shape == (256, 256, 3)
#         yield slice_pos, im


# def get_slice_3chan_im(image_id, band_cut_th):
#     fn = train_image_id_to_path(image_id)
#     with rasterio.open(fn, 'r') as f:
#         values = f.read().astype(np.float32)
#         for chan_i in range(3):
#             min_val = band_cut_th[chan_i]['min']
#             max_val = band_cut_th[chan_i]['max']
#             values[chan_i] = np.clip(values[chan_i], min_val, max_val)
#             values[chan_i] = (values[chan_i] - min_val) / (max_val - min_val)
#     values = np.swapaxes(values, 0, 2)
#     values = np.swapaxes(values, 0, 1)
#     assert values.shape == (650, 650, 3)
#
#     for slice_pos in range(9):
#         pos_j = int(math.floor(slice_pos / 3.0))
#         pos_i = int(slice_pos % 3)
#         x0 = STRIDE_SZ * pos_i
#         y0 = STRIDE_SZ * pos_j
#         im = values[x0:x0+INPUT_SIZE, y0:y0+INPUT_SIZE]
#         assert im.shape == (256, 256, 3)
#         yield slice_pos, im


# def get_slice_8chan_test_im(image_id, band_cut_th):
#     fn = test_image_id_to_mspec_path(image_id)
#     with rasterio.open(fn, 'r') as f:
#         values = f.read().astype(np.float32)
#         for chan_i in range(8):
#             min_val = band_cut_th[chan_i]['min']
#             max_val = band_cut_th[chan_i]['max']
#             values[chan_i] = np.clip(values[chan_i], min_val, max_val)
#             values[chan_i] = (values[chan_i] - min_val) / (max_val - min_val)
#     values = np.swapaxes(values, 0, 2)
#     values = np.swapaxes(values, 0, 1)
#     assert values.shape == (650, 650, 8)
#
#     for slice_pos in range(9):
#         pos_j = int(math.floor(slice_pos / 3.0))
#         pos_i = int(slice_pos % 3)
#         x0 = STRIDE_SZ * pos_i
#         y0 = STRIDE_SZ * pos_j
#         im = values[x0:x0+INPUT_SIZE, y0:y0+INPUT_SIZE]
#         assert im.shape == (256, 256, 8)
#         yield slice_pos, im


# def get_slice_8chan_im(image_id, band_cut_th):
#     fn = train_image_id_to_mspec_path(image_id)
#     with rasterio.open(fn, 'r') as f:
#         values = f.read().astype(np.float32)
#         for chan_i in range(8):
#             min_val = band_cut_th[chan_i]['min']
#             max_val = band_cut_th[chan_i]['max']
#             values[chan_i] = np.clip(values[chan_i], min_val, max_val)
#             values[chan_i] = (values[chan_i] - min_val) / (max_val - min_val)
#     values = np.swapaxes(values, 0, 2)
#     values = np.swapaxes(values, 0, 1)
#     assert values.shape == (650, 650, 8)
#
#     for slice_pos in range(9):
#         pos_j = int(math.floor(slice_pos / 3.0))
#         pos_i = int(slice_pos % 3)
#         x0 = STRIDE_SZ * pos_i
#         y0 = STRIDE_SZ * pos_j
#         im = values[x0:x0+INPUT_SIZE, y0:y0+INPUT_SIZE]
#         assert im.shape == (256, 256, 8)
#         yield slice_pos, im


# def get_mask_im(df, image_id):
#     im_mask = np.zeros((650, 650))
#     for idx, row in df[df.ImageId == image_id].iterrows():
#         shape_obj = shapely.wkt.loads(row.PolygonWKT_Pix)
#         if shape_obj.exterior is not None:
#             coords = list(shape_obj.exterior.coords)
#             x = [round(float(pp[0])) for pp in coords]
#             y = [round(float(pp[1])) for pp in coords]
#             yy, xx = skimage.draw.polygon(y, x, (650, 650))
#             im_mask[yy, xx] = 1
#
#             interiors = shape_obj.interiors
#             for interior in interiors:
#                 coords = list(interior.coords)
#                 x = [round(float(pp[0])) for pp in coords]
#                 y = [round(float(pp[1])) for pp in coords]
#                 yy, xx = skimage.draw.polygon(y, x, (650, 650))
#                 im_mask[yy, xx] = 0
#     im_mask = (im_mask > 0.5).astype(np.uint8)
#     return im_mask


# def get_slice_mask_im(df, image_id):
#     im_mask = np.zeros((650, 650))
#     for idx, row in df[df.ImageId == image_id].iterrows():
#         shape_obj = shapely.wkt.loads(row.WKT_Pix)
#         if shape_obj.exterior is not None:
#             coords = list(shape_obj.exterior.coords)
#             x = [round(float(pp[0])) for pp in coords]
#             y = [round(float(pp[1])) for pp in coords]
#             yy, xx = skimage.draw.polygon(y, x, (650, 650))
#             im_mask[yy, xx] = 1
#
#             interiors = shape_obj.interiors
#             for interior in interiors:
#                 coords = list(interior.coords)
#                 x = [round(float(pp[0])) for pp in coords]
#                 y = [round(float(pp[1])) for pp in coords]
#                 yy, xx = skimage.draw.polygon(y, x, (650, 650))
#                 im_mask[yy, xx] = 0
#     im_mask = (im_mask > 0.5).astype(np.uint8)
#
#     # Generate slices (3x3 tiles) from the mask image
#     for slice_pos in range(9):
#         pos_j = int(math.floor(slice_pos / 3.0))
#         pos_i = int(slice_pos % 3)
#         x0 = STRIDE_SZ * pos_i
#         y0 = STRIDE_SZ * pos_j
#         im_mask_part = im_mask[x0:x0+INPUT_SIZE, y0:y0+INPUT_SIZE]
#         assert im_mask_part.shape == (256, 256)
#         yield slice_pos, im_mask_part


# def prep_valtrain_test_slice_image(area_id):
#     prefix = area_id_to_prefix(area_id)
#     logger.info("prep_valtrain_test_slice_image for {}".format(prefix))
#
#     df_train = pd.read_csv(
#         FMT_VALTRAIN_IMAGELIST_PATH.format(prefix=prefix),
#         index_col='ImageId')
#     df_test = pd.read_csv(
#         FMT_VALTEST_IMAGELIST_PATH.format(prefix=prefix),
#         index_col='ImageId')
#     df_summary = load_train_summary_data(area_id)
#
#     # MUL
#     band_cut_th = __load_band_cut_th(
#         FMT_MUL_BANDCUT_TH_PATH, bandsz=8)[area_id]
#
#     fn = FMT_VALTRAIN_MUL_STORE.format(prefix)
#     logger.info("Prepare image container: {}".format(fn))
#     if not Path(fn).exists():
#         with tb.open_file(fn, 'w') as f:
#             for image_id in tqdm.tqdm(df_train.index, total=len(df_train)):
#                 for slice_pos, im in get_slice_8chan_im(image_id, band_cut_th):
#                     slice_id = image_id + "_{}".format(slice_pos)
#                     atom = tb.Atom.from_dtype(im.dtype)
#                     filters = tb.Filters(complib='blosc', complevel=9)
#                     ds = f.create_carray(f.root, slice_id, atom, im.shape,
#                                          filters=filters)
#                     ds[:] = im
#
#     fn = FMT_VALTEST_MUL_STORE.format(prefix)
#     logger.info("Prepare image container: {}".format(fn))
#     if not Path(fn).exists():
#         with tb.open_file(fn, 'w') as f:
#             for image_id in tqdm.tqdm(df_test.index, total=len(df_test)):
#                 for slice_pos, im in get_slice_8chan_im(image_id, band_cut_th):
#                     slice_id = image_id + "_{}".format(slice_pos)
#                     atom = tb.Atom.from_dtype(im.dtype)
#                     filters = tb.Filters(complib='blosc', complevel=9)
#                     ds = f.create_carray(f.root, slice_id, atom, im.shape,
#                                          filters=filters)
#                     ds[:] = im
#
#     # RGB
#     band_cut_th = __load_band_cut_th(FMT_RGB_BANDCUT_TH_PATH)[area_id]
#
#     fn = FMT_VALTRAIN_IM_STORE.format(prefix)
#     logger.info("Prepare image container: {}".format(fn))
#     if not Path(fn).exists():
#         with tb.open_file(fn, 'w') as f:
#             for image_id in tqdm.tqdm(df_train.index, total=len(df_train)):
#                 for slice_pos, im in get_slice_3chan_im(image_id, band_cut_th):
#                     slice_id = image_id + "_{}".format(slice_pos)
#                     atom = tb.Atom.from_dtype(im.dtype)
#                     filters = tb.Filters(complib='blosc', complevel=9)
#                     ds = f.create_carray(f.root, slice_id, atom, im.shape,
#                                          filters=filters)
#                     ds[:] = im
#
#     fn = FMT_VALTEST_IM_STORE.format(prefix)
#     logger.info("Prepare image container: {}".format(fn))
#     if not Path(fn).exists():
#         with tb.open_file(fn, 'w') as f:
#             for image_id in tqdm.tqdm(df_test.index, total=len(df_test)):
#                 for slice_pos, im in get_slice_3chan_im(image_id, band_cut_th):
#                     slice_id = image_id + "_{}".format(slice_pos)
#                     atom = tb.Atom.from_dtype(im.dtype)
#                     filters = tb.Filters(complib='blosc', complevel=9)
#                     ds = f.create_carray(f.root, slice_id, atom, im.shape,
#                                          filters=filters)
#                     ds[:] = im
#
#     fn = FMT_VALTRAIN_MASK_STORE.format(prefix)
#     logger.info("Prepare image container: {}".format(fn))
#     if not Path(fn).exists():
#         with tb.open_file(fn, 'w') as f:
#             for image_id in tqdm.tqdm(df_train.index, total=len(df_train)):
#                 for pos, im_mask in get_slice_mask_im(df_summary, image_id):
#                     atom = tb.Atom.from_dtype(im_mask.dtype)
#                     slice_id = image_id + "_" + str(pos)
#                     filters = tb.Filters(complib='blosc', complevel=9)
#                     ds = f.create_carray(f.root, slice_id, atom, im_mask.shape,
#                                          filters=filters)
#                     ds[:] = im_mask
#
#     fn = FMT_VALTEST_MASK_STORE.format(prefix)
#     logger.info("Prepare image container: {}".format(fn))
#     if not Path(fn).exists():
#         with tb.open_file(fn, 'w') as f:
#             for image_id in tqdm.tqdm(df_test.index, total=len(df_test)):
#                 for pos, im_mask in get_slice_mask_im(df_summary, image_id):
#                     atom = tb.Atom.from_dtype(im_mask.dtype)
#                     slice_id = image_id + "_" + str(pos)
#                     filters = tb.Filters(complib='blosc', complevel=9)
#                     ds = f.create_carray(f.root, slice_id, atom, im_mask.shape,
#                                          filters=filters)
#                     ds[:] = im_mask


# def prep_train_test_slice_image(area_id):
#     prefix = area_id_to_prefix(area_id)
#     logger.info("prep_train_test_slice_images for {}".format(prefix))
#
#     df_train = pd.read_csv(
#         FMT_TRAIN_IMAGELIST_PATH.format(prefix=prefix),
#         index_col='ImageId')
#     df_test = pd.read_csv(
#         FMT_TEST_IMAGELIST_PATH.format(prefix=prefix),
#         index_col='ImageId')
#     df_summary = load_train_summary_data(area_id)
#
#     # MUL
#     band_cut_th = __load_band_cut_th(
#         FMT_MUL_BANDCUT_TH_PATH, bandsz=8)[area_id]
#
#     fn = FMT_TRAIN_MUL_STORE.format(prefix)
#     logger.info("Prepare image container: {}".format(fn))
#     if not Path(fn).exists():
#         with tb.open_file(fn, 'w') as f:
#             for image_id in tqdm.tqdm(df_train.index, total=len(df_train)):
#                 for slice_pos, im in get_slice_8chan_im(image_id, band_cut_th):
#                     slice_id = image_id + "_{}".format(slice_pos)
#                     atom = tb.Atom.from_dtype(im.dtype)
#                     filters = tb.Filters(complib='blosc', complevel=9)
#                     ds = f.create_carray(f.root, slice_id, atom, im.shape,
#                                          filters=filters)
#                     ds[:] = im
#
#     fn = FMT_TEST_MUL_STORE.format(prefix)
#     logger.info("Prepare image container: {}".format(fn))
#     if not Path(fn).exists():
#         with tb.open_file(fn, 'w') as f:
#             for image_id in tqdm.tqdm(df_test.index, total=len(df_test)):
#                 for slice_pos, im in get_slice_8chan_test_im(
#                         image_id,
#                         band_cut_th):
#                     slice_id = image_id + "_{}".format(slice_pos)
#                     atom = tb.Atom.from_dtype(im.dtype)
#                     filters = tb.Filters(complib='blosc', complevel=9)
#                     ds = f.create_carray(f.root, slice_id, atom, im.shape,
#                                          filters=filters)
#                     ds[:] = im
#
#     # RGB
#     band_cut_th = __load_band_cut_th(FMT_RGB_BANDCUT_TH_PATH)[area_id]
#
#     fn = FMT_TRAIN_IM_STORE.format(prefix)
#     logger.info("Prepare image container: {}".format(fn))
#     if not Path(fn).exists():
#         with tb.open_file(fn, 'w') as f:
#             for image_id in tqdm.tqdm(df_train.index, total=len(df_train)):
#                 for slice_pos, im in get_slice_3chan_im(image_id, band_cut_th):
#                     slice_id = image_id + "_{}".format(slice_pos)
#                     atom = tb.Atom.from_dtype(im.dtype)
#                     filters = tb.Filters(complib='blosc', complevel=9)
#                     ds = f.create_carray(f.root, slice_id, atom, im.shape,
#                                          filters=filters)
#                     ds[:] = im
#
#     fn = FMT_TEST_IM_STORE.format(prefix)
#     logger.info("Prepare image container: {}".format(fn))
#     if not Path(fn).exists():
#         with tb.open_file(fn, 'w') as f:
#             for image_id in tqdm.tqdm(df_test.index, total=len(df_test)):
#                 for slice_pos, im in get_slice_3chan_test_im(image_id,
#                                                              band_cut_th):
#                     slice_id = image_id + "_{}".format(slice_pos)
#                     atom = tb.Atom.from_dtype(im.dtype)
#                     filters = tb.Filters(complib='blosc', complevel=9)
#                     ds = f.create_carray(f.root, slice_id, atom, im.shape,
#                                          filters=filters)
#                     ds[:] = im
#
#     fn = FMT_TRAIN_MASK_STORE.format(prefix)
#     logger.info("Prepare image container: {}".format(fn))
#     if not Path(fn).exists():
#         with tb.open_file(fn, 'w') as f:
#             for image_id in tqdm.tqdm(df_train.index, total=len(df_train)):
#                 for pos, im_mask in get_slice_mask_im(df_summary, image_id):
#                     atom = tb.Atom.from_dtype(im_mask.dtype)
#                     slice_id = image_id + "_" + str(pos)
#                     filters = tb.Filters(complib='blosc', complevel=9)
#                     ds = f.create_carray(f.root, slice_id, atom, im_mask.shape,
#                                          filters=filters)
#                     ds[:] = im_mask


# def calc_bandvalues_cut_threshold():
#     ## Already done int the v5 code
#     rows = []
#     for area_id in range(2, 6):
#         band_cut_th = __calc_mul_multiband_cut_threshold(area_id)
#         prefix = area_id_to_prefix(area_id)
#         row = dict(prefix=area_id_to_prefix(area_id))
#         row['area_id'] = area_id
#         for chan_i in band_cut_th.keys():
#             row['chan{}_max'.format(chan_i)] = band_cut_th[chan_i]['max']
#             row['chan{}_min'.format(chan_i)] = band_cut_th[chan_i]['min']
#         rows.append(row)
#     pd.DataFrame(rows).to_csv(FMT_MUL_BANDCUT_TH_PATH, index=False)
#
#     rows = []
#     for area_id in range(2, 6):
#         band_cut_th = __calc_rgb_multiband_cut_threshold(area_id)
#         prefix = area_id_to_prefix(area_id)
#         row = dict(prefix=area_id_to_prefix(area_id))
#         row['area_id'] = area_id
#         for chan_i in band_cut_th.keys():
#             row['chan{}_max'.format(chan_i)] = band_cut_th[chan_i]['max']
#             row['chan{}_min'.format(chan_i)] = band_cut_th[chan_i]['min']
#         rows.append(row)
#     pd.DataFrame(rows).to_csv(FMT_RGB_BANDCUT_TH_PATH, index=False)


# def __calc_rgb_multiband_cut_threshold(area_id):
#     ## Already done int the v5 code
#     prefix = area_id_to_prefix(area_id)
#     band_values = {k: [] for k in range(3)}
#     band_cut_th = {k: dict(max=0, min=0) for k in range(3)}
#
#     image_id_list = pd.read_csv(FMT_VALTRAIN_IMAGELIST_PATH.format(
#         prefix=prefix)).ImageId.tolist()
#     for image_id in tqdm.tqdm(image_id_list[:500]):
#         image_fn = train_image_id_to_path(image_id)
#         with rasterio.open(image_fn, 'r') as f:
#             values = f.read().astype(np.float32)
#             for i_chan in range(3):
#                 values_ = values[i_chan].ravel().tolist()
#                 values_ = np.array(
#                     [v for v in values_ if v != 0]
#                 )  # Remove sensored mask
#                 band_values[i_chan].append(values_)
#
#     image_id_list = pd.read_csv(FMT_VALTEST_IMAGELIST_PATH.format(
#         prefix=prefix)).ImageId.tolist()
#     for image_id in tqdm.tqdm(image_id_list[:500]):
#         image_fn = train_image_id_to_path(image_id)
#         with rasterio.open(image_fn, 'r') as f:
#             values = f.read().astype(np.float32)
#             for i_chan in range(3):
#                 values_ = values[i_chan].ravel().tolist()
#                 values_ = np.array(
#                     [v for v in values_ if v != 0]
#                 )  # Remove sensored mask
#                 band_values[i_chan].append(values_)
#
#     for i_chan in range(3):
#         band_values[i_chan] = np.concatenate(
#             band_values[i_chan]).ravel()
#         band_cut_th[i_chan]['max'] = scipy.percentile(
#             band_values[i_chan], 98)
#         band_cut_th[i_chan]['min'] = scipy.percentile(
#             band_values[i_chan], 2)
#     return band_cut_th


# def __calc_mul_multiband_cut_threshold(area_id):
#     ## Already done int the v5 code
#     prefix = area_id_to_prefix(area_id)
#     band_values = {k: [] for k in range(8)}
#     band_cut_th = {k: dict(max=0, min=0) for k in range(8)}
#
#     image_id_list = pd.read_csv(FMT_VALTRAIN_IMAGELIST_PATH.format(
#         prefix=prefix)).ImageId.tolist()
#     for image_id in tqdm.tqdm(image_id_list[:500]):
#         image_fn = train_image_id_to_mspec_path(image_id)
#         with rasterio.open(image_fn, 'r') as f:
#             values = f.read().astype(np.float32)
#             for i_chan in range(8):
#                 values_ = values[i_chan].ravel().tolist()
#                 values_ = np.array(
#                     [v for v in values_ if v != 0]
#                 )  # Remove censored mask
#                 band_values[i_chan].append(values_)
#
#     image_id_list = pd.read_csv(FMT_VALTEST_IMAGELIST_PATH.format(
#         prefix=prefix)).ImageId.tolist()
#     for image_id in tqdm.tqdm(image_id_list[:500]):
#         image_fn = train_image_id_to_mspec_path(image_id)
#         with rasterio.open(image_fn, 'r') as f:
#             values = f.read().astype(np.float32)
#             for i_chan in range(8):
#                 values_ = values[i_chan].ravel().tolist()
#                 values_ = np.array(
#                     [v for v in values_ if v != 0]
#                 )  # Remove censored mask
#                 band_values[i_chan].append(values_)
#
#     for i_chan in range(8):
#         band_values[i_chan] = np.concatenate(
#             band_values[i_chan]).ravel()
#         band_cut_th[i_chan]['max'] = scipy.percentile(
#             band_values[i_chan], 98)
#         band_cut_th[i_chan]['min'] = scipy.percentile(
#             band_values[i_chan], 2)
#     return band_cut_th


def train_image_id_to_mspec_path(image_id):
    """
    For the Multi Spectral training data
    """
    prefix = image_id_to_prefix(image_id)
    fn = FMT_TRAIN_MSPEC_IMAGE_PATH.format(
        prefix=prefix,
        image_id=image_id)
    return fn


def test_image_id_to_mspec_path(image_id):
    """
    For the Multi Spectral test data
    """
    prefix = image_id_to_prefix(image_id)
    fn = FMT_TEST_MSPEC_IMAGE_PATH.format(
        prefix=prefix,
        image_id=image_id)
    return fn


def train_image_id_to_path(image_id):
    """
    For the RGB training data
    """
    prefix = image_id_to_prefix(image_id)
    fn = FMT_TRAIN_RGB_IMAGE_PATH.format(
        prefix=prefix,
        image_id=image_id)
    return fn


def test_image_id_to_path(image_id):
    """
    For the RGB test data
    """
    prefix = image_id_to_prefix(image_id)
    fn = FMT_TEST_RGB_IMAGE_PATH.format(
        prefix=prefix,
        image_id=image_id)
    return fn


def image_id_to_prefix(image_id):
    prefix = image_id.split('img')[0][:-1]
    return prefix


def load_train_summary_data(area_id):
    # Loads the training set groundtruth into a dataframe
    prefix = area_id_to_prefix(area_id)
    fn = FMT_TRAIN_SUMMARY_PATH.format(prefix=prefix)
    df = pd.read_csv(fn)
    # df.loc[:, 'ImageId'] = df.ImageId.str[4:]
    return df


# def split_val_train_test(area_id):
#     prefix = area_id_to_prefix(area_id)
#
#     df = load_train_summary_data(area_id)
#     df_agg = df.groupby('ImageId').agg('first')
#     image_id_list = df_agg.index.tolist()
#     np.random.shuffle(image_id_list)
#     sz_valtrain = int(len(image_id_list) * 0.7)
#     sz_valtest = len(image_id_list) - sz_valtrain
#
#     # Parent directory
#     parent_dir = Path(FMT_VALTEST_IMAGELIST_PATH.format(prefix=prefix)).parent
#     if not parent_dir.exists():
#         parent_dir.mkdir(parents=True)
#
#     pd.DataFrame({'ImageId': image_id_list[:sz_valtrain]}).to_csv(
#         FMT_VALTRAIN_IMAGELIST_PATH.format(prefix=prefix),
#         index=False)
#     pd.DataFrame({'ImageId': image_id_list[sz_valtrain:]}).to_csv(
#         FMT_VALTEST_IMAGELIST_PATH.format(prefix=prefix),
#         index=False)


# def get_image_mask_from_dataframe(df, image_id):
#     im_mask = np.zeros((650, 650))
#     for idx, row in df[df.ImageId == image_id].iterrows():
#         shape_obj = shapely.wkt.loads(row.PolygonWKT_Pix)
#         if shape_obj.exterior is not None:
#             coords = list(shape_obj.exterior.coords)
#             x = [round(float(pp[0])) for pp in coords]
#             y = [round(float(pp[1])) for pp in coords]
#             yy, xx = skimage.draw.polygon(y, x, (650, 650))
#             im_mask[yy, xx] = 1
#
#             interiors = shape_obj.interiors
#             for interior in interiors:
#                 coords = list(interior.coords)
#                 x = [round(float(pp[0])) for pp in coords]
#                 y = [round(float(pp[1])) for pp in coords]
#                 yy, xx = skimage.draw.polygon(y, x, (650, 650))
#                 im_mask[yy, xx] = 0
#     im_mask = (im_mask > 0.5).astype(np.uint8)
#     return im_mask


@click.group()
def cli():
    pass


# @cli.command()
# def testmerge():
#     # file check
#     for area_id in range(2, 6):
#         prefix = area_id_to_prefix(area_id)
#         fn_out = FMT_TESTPOLY_PATH.format(prefix)
#         if not Path(fn_out).exists():
#             logger.info("Required file not found: {}".format(fn_out))
#             sys.exit(1)
#
#     # file check
#     for area_id in range(2, 6):
#         prefix = area_id_to_prefix(area_id)
#         fn_out = FMT_VALTESTPOLY_PATH.format(prefix)
#         if not Path(fn_out).exists():
#             logger.info("Required file not found: {}".format(fn_out))
#             sys.exit(1)
#
#     # merge files
#     rows = []
#     for area_id in range(2, 6):
#         prefix = area_id_to_prefix(area_id)
#         fn_out = FMT_VALTESTPOLY_PATH.format(prefix)
#         with open(fn_out, 'r') as f:
#             line = f.readline()
#             if area_id == 2:
#                 rows.append(line)
#             for line in f:
#                 # remove interiors
#                 line = _remove_interiors(line)
#                 rows.append(line)
#
#     fn_out = FMT_VALTESTPOLY_OVALL_PATH
#     with open(fn_out, 'w') as f:
#         for line in rows:
#             f.write(line)
#
#     # merge files
#     rows = []
#     for area_id in range(2, 6):
#         prefix = area_id_to_prefix(area_id)
#         fn_out = FMT_VALTESTTRUTH_PATH.format(prefix)
#         with open(fn_out, 'r') as f:
#             line = f.readline()
#             if area_id == 2:
#                 rows.append(line)
#             for line in f:
#                 rows.append(line)
#     fn_out = FMT_VALTESTTRUTH_OVALL_PATH
#     with open(fn_out, 'w') as f:
#         for line in rows:
#             f.write(line)
#
#     # merge files
#     rows = []
#     for area_id in range(2, 6):
#         prefix = area_id_to_prefix(area_id)
#         fn_out = FMT_TESTPOLY_PATH.format(prefix)
#         with open(fn_out, 'r') as f:
#             line = f.readline()
#             if area_id == 2:
#                 rows.append(line)
#             for line in f:
#                 # remove interiors
#                 line = _remove_interiors(line)
#                 rows.append(line)
#
#     with open(FN_SOLUTION_CSV, 'w') as f:
#         for line in rows:
#             f.write(line)


# @cli.command()
# @click.argument('data_path', type=str)
# def testproc(data_path):
#     area_id = directory_name_to_area_id(data_path)
#     prefix = area_id_to_prefix(area_id)
#     logger.info(">>>> Test proc for {}".format(prefix))
#
#     _internal_test(area_id)
#     logger.info(">>>> Test proc for {} ... done".format(prefix))


# @cli.command()
# @click.argument('area_id', type=int)
# @click.option('--epoch', type=int, default=0)
# @click.option('--th', type=int, default=MIN_POLYGON_AREA)
# @click.option('--predict/--no-predict', default=False)
# def validate_city_fscore(area_id, epoch, th, predict):
#     _internal_validate_fscore(
#         area_id,
#         epoch=epoch,
#         enable_tqdm=True,
#         min_th=th,
#         predict=predict)
#     evaluate_record = _calc_fscore_per_aoi(area_id)
#     evaluate_record['epoch'] = epoch
#     evaluate_record['min_area_th'] = th
#     evaluate_record['area_id'] = area_id
#     logger.info("\n" + json.dumps(evaluate_record, indent=4))

# @cli.command()
# @click.argument('datapath', type=str)
# def evalfscore(datapath):
#     area_id = directory_name_to_area_id(datapath)
#     prefix = area_id_to_prefix(area_id)
#     logger.info("Evaluate fscore on validation set: {}".format(prefix))
#
#     # for each epoch
#     # if not Path(FMT_VALMODEL_EVALHIST.format(prefix)).exists():
#     if True:
#         df_hist = pd.read_csv(FMT_VALMODEL_HIST.format(prefix))
#         df_hist.loc[:, 'epoch'] = list(range(1, len(df_hist) + 1))
#
#         rows = []
#         for zero_base_epoch in range(0, len(df_hist)):
#             logger.info(">>> Epoch: {}".format(zero_base_epoch))
#
#             _internal_validate_fscore_wo_pred_file(
#                 area_id,
#                 epoch=zero_base_epoch,
#                 enable_tqdm=True,
#                 min_th=MIN_POLYGON_AREA)
#             evaluate_record = _calc_fscore_per_aoi(area_id)
#             evaluate_record['zero_base_epoch'] = zero_base_epoch
#             evaluate_record['min_area_th'] = MIN_POLYGON_AREA
#             evaluate_record['area_id'] = area_id
#             logger.info("\n" + json.dumps(evaluate_record, indent=4))
#             rows.append(evaluate_record)
#
#         pd.DataFrame(rows).to_csv(
#             FMT_VALMODEL_EVALHIST.format(prefix),
#             index=False)
#
#     # find best min-poly-threshold
#     df_evalhist = pd.read_csv(FMT_VALMODEL_EVALHIST.format(prefix))
#     best_row = df_evalhist.sort_values(by='fscore', ascending=False).iloc[0]
#     best_epoch = int(best_row.zero_base_epoch)
#     best_fscore = best_row.fscore
#
#     # optimize min area th
#     rows = []
#     for th in [30, 60, 90, 120, 150, 180, 210, 240]:
#         logger.info(">>> TH: {}".format(th))
#         predict_flag = False
#         if th == 30:
#             predict_flag = True
#
#         _internal_validate_fscore(
#             area_id,
#             epoch=best_epoch,
#             enable_tqdm=True,
#             min_th=th,
#             predict=predict_flag)
#         evaluate_record = _calc_fscore_per_aoi(area_id)
#         evaluate_record['zero_base_epoch'] = best_epoch
#         evaluate_record['min_area_th'] = th
#         evaluate_record['area_id'] = area_id
#         logger.info("\n" + json.dumps(evaluate_record, indent=4))
#         rows.append(evaluate_record)
#
#     pd.DataFrame(rows).to_csv(
#         FMT_VALMODEL_EVALTHHIST.format(prefix),
#         index=False)
#
#     logger.info("Evaluate fscore on validation set: {} .. done".format(prefix))


@cli.command()
@click.argument('datapath', type=str)
def validate(datapath): # This is the training process
    area_id = directory_name_to_area_id(datapath)
    prefix = area_id_to_prefix(area_id)
    logger.info(">> validate sub-command: {}".format(prefix))

    prefix = area_id_to_prefix(area_id)
    logger.info("Loading valtest and mulmean ...")
    X_mean = get_mul_mean_image(area_id)
    X_val, y_val = get_valtest_data(area_id)
    X_val = X_val - X_mean

    if not Path(MODEL_DIR).exists():
        Path(MODEL_DIR).mkdir(parents=True)

    init_ep = 0
    name_tail = "_{:02d}".format(init_ep)
    while os.path.isfile(FMT_VALMODEL_PATH.format(prefix + name_tail)):
        last_model_path = FMT_VALMODEL_PATH.format(prefix + name_tail)
        print(last_model_path)
        init_ep += 1
        name_tail = "_{:02d}".format(init_ep)

    logger.info("Instantiate U-Net model")
    model = get_unet()
    model_checkpoint = ModelCheckpoint(
        FMT_VALMODEL_LAST_PATH.format(prefix),
        monitor='val_jaccard_coef_int',
        save_best_only=True)

    model_earlystop = EarlyStopping(
        monitor='val_jaccard_coef_int',
        patience=10,
        verbose=0,
        mode='max')
    model_history = History()

    if os.path.isfile(FMT_VALMODEL_LAST_PATH.format(prefix)):
        model.load_weights(FMT_VALMODEL_LAST_PATH.format(prefix))
    elif init_ep >= 1:
        model.load_weights(last_model_path)
    else:
        print("No weight/model file found. Training from scratch...")

    df_train = pd.read_csv(FMT_VALTRAIN_IMAGELIST_PATH.format(
        prefix=prefix))
    logger.info("Fit")
    model.fit_generator(
        generate_valtrain_batch(area_id, batch_size=3, immean=X_mean),
        samples_per_epoch=len(df_train)*3,
        nb_epoch=20,
        verbose=1,
        validation_data=(X_val, y_val),
        callbacks=[model_checkpoint, model_earlystop, model_history])

    # model.save_weights(FMT_VALMODEL_LAST_PATH.format(prefix))

    # Save evaluation history
    pd.DataFrame(model_history.history).to_csv(
        FMT_VALMODEL_HIST.format(prefix), index=False)
    logger.info(">> validate sub-command: {} ... Done".format(prefix))


@cli.command()
@click.argument('datapath', type=str)
def test(datapath):
    area_id = directory_name_to_area_id(datapath)
    prefix = area_id_to_prefix(area_id)
    logger.info(">>>> Test proc for {}".format(prefix))

    _internal_test_2(area_id)
    logger.info(">>>> Test proc for {} ... done".format(prefix))


if __name__ == '__main__':
    cli()
