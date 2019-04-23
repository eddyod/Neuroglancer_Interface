# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 11:11:37 2019

@author: thinh
"""
import os
import sys

import pathlib
import numpy as np
import json
import re
from PIL import Image
import math

import boto3
from skimage import io
from neuroglancer_scripts.scripts import (generate_scales_info,
                                          slices_to_precomputed,
                                          compute_scales)


# ====================== working with S3 bucket ======================

def get_bucket(s3_creds_file, bucket_name):
    s3_creds_file = pathlib.Path(s3_creds_file)
    with open(s3_creds_file) as f:
        creds = json.load(f)
        aws_access_key_id = creds['access_key']
        aws_secret_access_key = creds['secret_key']
        region_name = creds['region']

    session = boto3.Session(
        aws_access_key_id = aws_access_key_id,
        aws_secret_access_key = aws_secret_access_key,
        region_name = region_name)
    s3 = session.resource('s3')
    return s3.Bucket(bucket_name)


# write every recursive folder(s), file(s) inside `dir_to_write_from` to `s3_dir_to_write_to`
def upload_directory_to_s3(s3_creds_file, bucket_name, dir_to_write_from, s3_dir_to_write_to, overwrite=True):
    bucket = get_bucket(s3_creds_file, bucket_name)
    dir_to_write_from = '{0}'.format(pathlib.Path(dir_to_write_from))
    # write the whole directory onto S3 bucket, upload files
    for subdir, dirs, files in os.walk(dir_to_write_from):
        print('Uploading: {0}'.format(subdir))
        for file in files:
            full_path = os.path.join(subdir, file)
            fp_s3 = re.sub(r'\\', '/', full_path)
            f_key = ''.join([s3_dir_to_write_to, re.sub(re.sub(r'\\', '/', dir_to_write_from), '', fp_s3)])

            if not overwrite:
                obj_keys = [obj.key for obj in bucket.objects.filter(Prefix = f_key)]
                if len(obj_keys) == 0:
                    bucket.upload_file(full_path, f_key)
            else:
                bucket.upload_file(full_path, f_key)


def delete_dir_from_s3(s3_creds_file, bucket_name, s3_dir_to_delete):
    bucket = get_bucket(s3_creds_file, bucket_name)
    obj_keys = [{'Key': obj.key} for obj in bucket.objects.filter(Prefix = s3_dir_to_delete)]
    # delete limit of 1000 objects per call
    while len(obj_keys) > 0:
        sub_obj_keys = [obj_keys.pop(idx) for idx in sorted(range(np.minimum(999, len(obj_keys))), reverse = True)]
        bucket.delete_objects(Delete = {'Objects': sub_obj_keys})


def list_files_from_s3(s3_creds_file, bucket_name, prefix, parts):
    bucket = get_bucket(s3_creds_file, bucket_name)
    return [obj.key for obj in bucket.objects.filter(Prefix = prefix)
            if re.search(parts, obj.key)]


def resize_canvas(old_image_path, new_image_path,
                  canvas_width=500, canvas_height=500):
    """
    Resize the canvas of old_image_path.
    Store the new image in new_image_path. Center the image on the new canvas.
    Parameters
    ----------
    old_image_path : str
    new_image_path : str
    canvas_width : int
    canvas_height : int
    """
    im = Image.open(old_image_path)
    old_width, old_height = im.size
    # Center the image
    x1 = int(math.floor((canvas_width - old_width) / 2))
    y1 = int(math.floor((canvas_height - old_height) / 2))
    mode = im.mode
    if len(mode) == 1:  # L, 1
        new_background = (255)
    if len(mode) == 3:  # RGB
        new_background = (255, 255, 255)
    if len(mode) == 4:  # RGBA, CMYK
        new_background = (255, 255, 255, 255)
    newImage = Image.new(mode, (canvas_width, canvas_height), new_background)
    newImage.paste(im, (x1, y1, x1 + old_width, y1 + old_height))
    newImage.save(new_image_path)


def download_ordered_files_from_s3(s3_creds_file, bucket_name,
                                   sorted_filename, folder_to_write_to,
                                   ext='.tif', s3_prefix='', s3_parts=''):
    bucket = get_bucket(s3_creds_file, bucket_name)
    folder_to_write_to = pathlib.Path(folder_to_write_to)
    f_infos = []
    # ---------------- download images on S3 ----------------
    if isinstance(sorted_filename, list):
        f_infos = [(re.search('(?<=\s{1})(\d+)', l).group(), re.search('(.+)(?=\s\d)', l).group()) for l in sorted_filename]
    elif isinstance(sorted_filename, str) and os.path.exists(sorted_filename):
        with open(sorted_filename) as f:
            f_infos = [(re.search('(?<=\s{1})(\d+)', l).group(), re.search('(.+)(?=\s\d)', l).group()) for l in f]

    print(s3_prefix)
    # -- search S3 for all the keys matching sorted_filenames
    placeholder_idx = []
    s3_keys = []
    print('Searching:     ', end='')
    for idx, key in f_infos:
        print('\b\b\b\b', end='')
        print('{0:04d}'.format(int(idx)), end='')
        k = list_files_from_s3(s3_creds_file, bucket_name, s3_prefix, key)
        if re.search('Placeholder', key) or len(k) == 0:
            placeholder_idx.append(idx)
            continue
        elif len(k) > 1:
            raise NameError('Found more than one file with the given prefix and filename: {key}')
        else:
            obj_key = ''.join([s3_prefix, key, s3_parts, ext])
            fname_to_write = '_'.join(['{0:04d}'.format(int(idx)), key, ext])
            s3_keys.append((obj_key, os.path.join(folder_to_write_to, fname_to_write)))
    print('\n')

    if len(s3_keys) == 0:
        raise FileNotFoundError('No file(s) found for download')
    print('Found {0} files to download'.format(len(s3_keys)))
    if not os.path.exists(folder_to_write_to):
        os.makedirs(folder_to_write_to)
    for s3_k, dest in s3_keys:
        bucket.download_file(s3_k, dest)


def convert_to_precomputed(folder_to_convert_from, folder_to_convert_to, voxel_resolution, voxel_offset=[0, 0, 0]):
    # ---------------- Conversion to precomputed format ----------------
    folder_to_convert_from = '{0}'.format(pathlib.Path(folder_to_convert_from))
    folder_to_convert_to = '{0}'.format(pathlib.Path(folder_to_convert_to))

    info_fullres_template = {
        "type": "image",
        "num_channels": None,
        "scales": [{
            "chunk_sizes": [],
            "encoding": "raw",
            "key": "full",
            "resolution": [None, None, None],
            "size": [None, None, None],
            "voxel_offset": voxel_offset}],
        "data_type": None}

    # make a folder under the "precomputed" dir and execute conversion routine
    if not os.path.isdir(folder_to_convert_from):
        raise NotADirectoryError
    # make a corresponding folder in the "precomputed_dir"
    if not os.path.exists(folder_to_convert_to):
        os.makedirs(folder_to_convert_to)
    # read 1 image to get the shape
    imgs = os.listdir(folder_to_convert_from)
    img = io.imread(os.path.join(folder_to_convert_from, imgs[0]))
    # write info_fullres.json
    info_fullres = info_fullres_template.copy()
    info_fullres['scales'][0]['size'] = [img.shape[1], img.shape[0], len(imgs)]
    info_fullres['scales'][0]['resolution'] = voxel_resolution
    info_fullres['num_channels'] = img.shape[2] if len(img.shape) > 2 else 1
    info_fullres['data_type'] = str(img.dtype)
    with open(os.path.join(folder_to_convert_to, 'info_fullres.json'), 'w') as outfile:
        json.dump(info_fullres, outfile)

    # --- neuroglancer-scripts routine ---
    #  generate_scales_info - make info.json
    generate_scales_info.main(['', os.path.join(folder_to_convert_to, 'info_fullres.json'),
                               folder_to_convert_to])
    # slices_to_precomputed - build the precomputed for the fullress
    slices_to_precomputed.main(
        ['', folder_to_convert_from, folder_to_convert_to, '--flat', '--no-gzip'])
    # compute_scales - build the precomputed for other scales
    compute_scales.main(['', folder_to_convert_to, '--flat', '--no-gzip'])


def s3_convert_to_precomputed(s3_creds_file, s3_bucket_name_for_download, s3_bucket_name_for_upload,
                              sorted_filename, folder_to_write_to,
                              folder_to_convert_from, folder_to_convert_to, s3_dir_to_write_to, voxel_resolution,
                              ext='.tif', s3_prefix='', s3_parts='', voxel_offset=[0, 0, 0], overwrite=False):

    folder_to_write_to = '{0}'.format(pathlib.Path(folder_to_write_to))
    folder_to_convert_from = '{0}'.format(pathlib.Path(folder_to_convert_from))
    folder_to_convert_to = '{0}'.format(pathlib.Path(folder_to_convert_to))

    print('============ Step 1 - download from S3 ====================')
    #download_ordered_files_from_s3(s3_creds_file, s3_bucket_name_for_download,
    #                               sorted_filename, folder_to_write_to,
    #                               ext, s3_prefix, s3_parts)
    print('============ Step 2 - convert to precomputed ==============')
    convert_to_precomputed(folder_to_convert_from, folder_to_convert_to, voxel_resolution, voxel_offset)
    print('============ Step 3 - upload precomputed to S3 ============')
    #upload_directory_to_s3(s3_creds_file, s3_bucket_name_for_upload, folder_to_convert_to, s3_dir_to_write_to, overwrite)


def main(argv=sys.argv):
    Image.MAX_IMAGE_PIXELS = None
    """The script's entry point. User pass a json config file as a single argument"""
    with open(argv[1]) as f:
        conversion_config = json.load(f)
    s3_convert_to_precomputed(**conversion_config)


if __name__ == "__main__":
    sys.exit(main())
