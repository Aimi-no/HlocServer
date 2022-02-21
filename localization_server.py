# -*- encoding: utf-8 -*-
# pip install flask

import flask
from flask import request
import os
import time
import socket
import argparse
from PIL import Image
import base64
import io
import numpy as np
from pathlib import Path
import sys
import re

import torch
import h5py
import logging
from types import SimpleNamespace
import cv2
from tqdm import tqdm
import pprint
import collections.abc as collections

from hloc import extract_features, match_features
from hloc import pairs_from_covisibility, pairs_from_retrieval
from hloc import colmap_from_nvm, triangulation, localize_sfm

from hloc import extractors
from hloc.utils.base_model import dynamic_load
from hloc.utils.tools import map_tensor
from hloc.utils.parsers import parse_image_lists
from hloc.utils.io import read_image, list_h5_names

from hloc import matchers
from hloc.utils.parsers import names_to_pair, parse_retrieval

from hloc.utils.read_write_model import read_model, read_images_binary
from torch.multiprocessing import Process


app = flask.Flask(__name__)

folder = ''
img_count = 0
args = []


# Configure the flask server
app.config['JSON_SORT_KEYS'] = False
global MATLAB_PATH
global MATLAB_SCRIPT_FOLDER
global MATLAB_FUNCTION

global features
global global_descriptors
global loc_matches
global outputs
global loc_pairs
global dataset
global sfm_pairs
global feature_model
global retrieval_model

global model_db_images
global model_points3D


def load_hloc_data(args):
    global dataset
    dataset = args.dataset
    images = dataset / 'images/images_upright/'
    queries = args.query_path

    global outputs
    outputs = args.outputs  # where everything will be saved
    sift_sfm = outputs / 'sfm_sift'  # from which we extract the reference poses
    reference_sfm = outputs / 'sfm_superpoint+superglue'  # the SfM model we will build
    global sfm_pairs
    sfm_pairs = outputs / f'pairs-db-covis{args.num_covis}.txt'  # top-k most covisible in SIFT model
    global loc_pairs
    loc_pairs = outputs / f'pairs-query-netvlad{args.num_loc}.txt'  # top-k retrieved by NetVLAD
    results = outputs / f'Aachen_hloc_superpoint+superglue_netvlad{args.query_number}.txt'

    # pick one of the configurations for extraction and matching
    retrieval_conf = extract_features.confs['netvlad']
    feature_conf = extract_features.confs['superpoint_aachen']
    matcher_conf = match_features.confs['superglue']

    global features
    features = extract_features.main(feature_conf, images, outputs) 
    print('---------')
    print(features)
    print('---------')

    colmap_from_nvm.main(
        dataset / '3D-models/aachen_cvpr2018_db.nvm',
        dataset / '3D-models/database_intrinsics.txt',
        dataset / 'aachen.db',
        sift_sfm)
    pairs_from_covisibility.main(
        sift_sfm, sfm_pairs, num_matched=args.num_covis)
        
    sfm_matches = match_features.main(
        matcher_conf, sfm_pairs, feature_conf['output'], outputs)

    triangulation.main(
        reference_sfm,
        sift_sfm,
        images,
        sfm_pairs,
        features,
        sfm_matches,
        colmap_path='colmap')

    global global_descriptors
    global_descriptors = extract_features.main(retrieval_conf, images, outputs)

    print('---------')
    print(global_descriptors)
    print('---------')
    pairs_from_retrieval.main(
        global_descriptors, loc_pairs, args.num_loc,
        query_prefix='query', db_model=reference_sfm)

    global loc_matches
    loc_matches = match_features.main(
        matcher_conf, loc_pairs, feature_conf['output'], outputs) 


    global feature_model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Model = dynamic_load(extractors, feature_conf['model']['name'])
    feature_model = Model(feature_conf['model']).eval().to(device)

    global retrieval_model
    Model = dynamic_load(extractors, retrieval_conf['model']['name'])
    retrieval_model = Model(retrieval_conf['model']).eval().to(device)
 
    global matcher_model
    Model = dynamic_load(matchers, matcher_conf['model']['name'])
    matcher_model = Model(matcher_conf['model']).eval().to(device)

    global model_db_images
    global model_points3D
    logging.info('Reading 3D model...')
    _, model_db_images, model_points3D = read_model(str(reference_sfm))


    return ''


@app.route('/api/matlab_run_cmd', methods=['POST'])
def api_matlab_run_cmd():
    try:
        global MATLAB_PATH
        global MATLAB_SCRIPT_FOLDER
        global MATLAB_FUNCTION

        print('Parsing input arguments')
        a = int(request.form['a'])
        b = int(request.form['b'])
        print('Running ' + MATLAB_FUNCTION + '(' + str(a) + ',' + str(b) + ')')
        # For Windows
        answer = os.popen( MATLAB_PATH +  ' -sd ' + MATLAB_SCRIPT_FOLDER + ' -batch ' + MATLAB_FUNCTION + '(' + str(a) + ',' + str(b) + ')').read()
        # For Linux
        #answer = os.popen( MATLAB_PATH +  ' -nodisplay -nosplash -r "cd(\'' + MATLAB_SCRIPT_FOLDER + '\');' + MATLAB_FUNCTION + '(' + str(a) + ',' + str(b) + ')' + ';exit;"').read()
        print('Sending Matlab answer: ' + answer)
        response_list = [('matlab_answer', answer)]

        response_dict = dict(response_list)

        return flask.jsonify(response_dict)

    except Exception as err:
        print('ko:', err)

    return 'ok'

@torch.no_grad()
def extract_feature_one_image(feature_conf, img_path, img_name, filepath, model):
    loader = extract_features.ImageDataset(Path(img_path), feature_conf['preprocessing'], [img_name])
    loader = torch.utils.data.DataLoader(loader, num_workers=1)

    filepath.parent.mkdir(exist_ok=True, parents=True)
    skip_names = set(list_h5_names(filepath)
                    if filepath.exists() else ())
    if set(loader.dataset.names).issubset(set(skip_names)):
        logging.info('Skipping the extraction.')
        return filepath

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    '''Model = dynamic_load(extractors, feature_conf['model']['name'])
    model = Model(feature_conf['model']).eval().to(device)'''

    for data in tqdm(loader):
        name = data['name'][0]  # remove batch dimension
        #print(name) db images
        if name in skip_names:
            continue

        pred = model(map_tensor(data, lambda x: x.to(device)))
        pred = {k: v[0].cpu().numpy() for k, v in pred.items()}

        pred['image_size'] = original_size = data['original_size'][0].numpy()
        if 'keypoints' in pred:
            size = np.array(data['image'].shape[-2:][::-1])
            scales = (original_size / size).astype(np.float32)
            pred['keypoints'] = (pred['keypoints'] + .5) * scales[None] - .5

        with h5py.File(str(filepath), 'a') as fd:
            query_grp = fd.require_group('query')
            folder_grp = query_grp.require_group('test')
            img_grp = folder_grp.require_group(img_name)

            for k, v in pred.items():
                img_grp.create_dataset(k, data=v)

        del pred



@torch.no_grad()
def match_features_one_image(features, query_name, loc_pairs, model, outputs): 
    matcher_conf = match_features.confs['superglue']

    if isinstance(features, Path) or Path(features).exists():
        features_q = features
    else:
        features_q = Path(outputs, features+'.h5')

    features_ref = features_q
    if isinstance(features_ref, collections.Iterable):
        features_ref = list(features_ref)
    else:
        features_ref = [features_ref]

    match_path = Path(
        outputs, f'{features}_{matcher_conf["output"]}_{loc_pairs.stem}.h5')

    logging.info('Matching local features with configuration:'
                f'\n{pprint.pformat(matcher_conf)}')

    if not features_q.exists():
        raise FileNotFoundError(f'Query feature file {features_q}.')
    for path in features_ref:
        if not path.exists():
            raise FileNotFoundError(f'Reference feature file {path}.')
    name2ref = {n: i for i, p in enumerate(features_ref)
                for n in list_h5_names(p)}


    assert loc_pairs.exists(), loc_pairs
    pairs = parse_retrieval(loc_pairs)
    pairs = [(q, r) for q, rs in pairs.items() for r in rs if q == query_name]
    #print(pairs)
    #print(len(pairs))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    '''Model = dynamic_load(matchers, matcher_conf['model']['name'])
    model = Model(matcher_conf['model']).eval().to(device)'''

    match_path.parent.mkdir(exist_ok=True, parents=True)
    skip_pairs = set(list_h5_names(match_path) if match_path.exists() else ())

    for (name0, name1) in tqdm(pairs, smoothing=.1):
        pair = names_to_pair(name0, name1)
        # Avoid to recompute duplicates to save time
        if pair in skip_pairs or names_to_pair(name0, name1) in skip_pairs:
            continue

        data = {}
        with h5py.File(str(features_q), 'r') as fd:
            grp = fd[name0]
            for k, v in grp.items():
                data[k+'0'] = torch.from_numpy(v.__array__()).float().to(device)
            # some matchers might expect an image but only use its size
            data['image0'] = torch.empty((1,)+tuple(grp['image_size'])[::-1])
        with h5py.File(str(features_ref[name2ref[name1]]), 'r') as fd:
            grp = fd[name1]
            for k, v in grp.items():
                data[k+'1'] = torch.from_numpy(v.__array__()).float().to(device)
            data['image1'] = torch.empty((1,)+tuple(grp['image_size'])[::-1])
        data = {k: v[None] for k, v in data.items()}

        pred = model(data)
        with h5py.File(str(match_path), 'a') as fd:
            grp = fd.create_group(pair)
            matches = pred['matches0'][0].cpu().short().numpy()
            grp.create_dataset('matches0', data=matches)

            if 'matching_scores0' in pred:
                scores = pred['matching_scores0'][0].cpu().half().numpy()
                grp.create_dataset('matching_scores0', data=scores)

        skip_pairs.add(pair)
    return match_path


def thread_feature_processing(feature_conf, img_path, img_name, features, feature_model, sfm_pairs, matcher_model, outputs):
    extract_feature_one_image(feature_conf, img_path, img_name, features, feature_model)
    logging.info('Extracted features')
    sfm_matches = match_features_one_image(feature_conf['output'], 'query/test/' + img_name, sfm_pairs, matcher_model, outputs)
    logging.info('Features matched')

def thread_descriptor_processing(retrieval_conf, img_path, img_name, global_descriptors, retrieval_model, loc_pairs, args, reference_sfm, model_db_images, feature_conf, matcher_model, outputs):
    extract_feature_one_image(retrieval_conf, img_path, img_name, global_descriptors, retrieval_model)
    logging.info('Features extracted')
    pairs_from_retrieval.main(
            global_descriptors, loc_pairs, args.num_loc,
            query_prefix='query', db_model=reference_sfm, query_list = ['query/test/' + img_name], images = model_db_images, query_names = ['query/test/' + img_name])

    logging.info('Pairs from retrieval done')

    global loc_matches
    loc_matches = match_features_one_image(feature_conf['output'], 'query/test/' + img_name, loc_pairs, matcher_model, outputs)

    logging.info('Features matched')




@app.route('/api/localize', methods=['POST'])
def api_save_img():
    try:
        logging.info('Aquired request, request length: ' + str(request.content_length))
        data = request.stream.read()
        #print(request.headers)


        #with open('/local/homes/zderaann/testfile.txt', 'wb') as f:
        #    f.write(data)
        global img_count
        parsed = data.split(b'\r\n')
        logging.info('Split img info')

        width_index = parsed.index(b'Content-Disposition: form-data; name="width"')
        width = int(parsed[width_index + 2])
        height_index = parsed.index(b'Content-Disposition: form-data; name="height"')
        height = int(parsed[height_index + 2]) 
        imgname_index = parsed.index(b'Content-Disposition: form-data; name="imagename"')
        imgname = parsed[imgname_index + 2].decode('utf-8') 
        col_layout_index = parsed.index(b'Content-Disposition: form-data; name="colorLayout"')
        col_layout = int(parsed[col_layout_index + 2])
        cam_model_index = parsed.index(b'Content-Disposition: form-data; name="camModel"')
        cam_model = parsed[cam_model_index + 2].decode('utf-8') 
        params_index = parsed.index(b'Content-Disposition: form-data; name="camParams"')
        cam_params = parsed[params_index + 2].decode('utf-8') 
        
        intrisics = ' ' + cam_model + ' ' + str(width) + ' ' + str(height) + ' ' + cam_params

        # CAMERA_MODEL, WIDTH, HEIGHT, PARAMS[]

        path = os.path.dirname(os.path.realpath(__file__)) + '/'
        img_path = '/local/artwin/localization/Hierarchical-Localization/datasets/aachen/images/images_upright/query/test/'
        img_name = imgname + '_' + str(img_count) + '.jpg'
        filename = img_path +  img_name
        img_count = img_count + 1


        logging.info('Parsed img info')
        image_index = parsed.index(b'Content-Disposition: form-data; name="image"')
        imglines = parsed[image_index + 2:-1]
        indices = [i for i, s in enumerate(imglines) if b'--------------------------' in s]
        image_end = indices[0]
        img = b'\r\n'.join(imglines[0:image_end])
        logging.info('Parsed img data')
        logging.info('Saving image as: ' + filename)
        
        if col_layout == 0: #RGB
            image = Image.frombytes('RGB', (width, height), img, 'raw')
            image.save(filename)

        elif col_layout == 1: #GRB
            image = Image.frombytes('RGB', (width, height), img, 'raw')
            [G,R,B] = image.split()

            rgbimg = Image.merge('RGB', [R,G,B])
            rgbimg.save(filename)

        elif col_layout == 2: #BGR
            image = Image.frombytes('RGB', (width, height), img, 'raw')
            [B,G,R] = image.split()

            rgbimg = Image.merge('RGB', [R,G,B])
            rgbimg.save(filename)

        elif col_layout == 3: #GREY
            image = Image.frombytes('L', (width, height), img, 'raw')
            image.save(filename)

        elif col_layout == 4: #RGBA
            image = Image.frombytes('RGBA', (width, height), img, 'raw')
            image.save(filename)

        elif col_layout == 5: #RGBX, might not work, Pillow provides limited support for this mode
            image = Image.frombytes('RGBX', (width, height), img, 'raw')
            image.save(filename)
        else:
            raise ValueError('Not a supported layout of image colors!')

        logging.info('Image saved')


        #-----------POSE CALCULATION HERE----------
        # Intrisics

        #intrisics = ' SIMPLE_RADIAL 1600 1200 1469.2 800 600 -0.0353019'
        query_path = args.query_saving_path + '/' + imgname + '_' + str(img_count) + '_query.txt'
        f = open(query_path, 'w')
        f.write('query/test/' + img_name + intrisics)
        f.close()

        logging.info('Saved query file')


        dataset = args.dataset
        images = dataset / 'images/images_upright/'
        outputs = args.outputs  # where everything will be saved
        sift_sfm = outputs / 'sfm_sift'  # from which we extract the reference poses
        reference_sfm = outputs / 'sfm_superpoint+superglue'  # the SfM model we will build
        sfm_pairs = outputs / f'pairs-db-covis{args.num_covis}.txt'  # top-k most covisible in SIFT model
        loc_pairs = outputs / f'pairs-query-netvlad{args.num_loc}.txt'  # top-k retrieved by NetVLAD
        results = outputs / f'Aachen_hloc_superpoint+superglue_netvlad{args.query_number}.txt'

        # pick one of the configurations for extraction and matching
        retrieval_conf = extract_features.confs['netvlad']
        feature_conf = extract_features.confs['superpoint_aachen']
        matcher_conf = match_features.confs['superglue']

        logging.info('Configurations set')

        
        global features
        global feature_model
        global matcher_model


        # ------------------ Descriptor extraction ---------------------- #
        global global_descriptors
        retrieval_conf = extract_features.confs['netvlad']
        global retrieval_model
        global model_db_images

        # ------------------ Feature extraction ---------------------- #
        torch.multiprocessing.set_start_method('spawn', force=True)

        descriptor_thread = Process(target=thread_descriptor_processing, args=(retrieval_conf, img_path, img_name, global_descriptors, retrieval_model, loc_pairs, args, reference_sfm, model_db_images, feature_conf, matcher_model, outputs))
        descriptor_thread.start()

        feature_thread = Process(target=thread_feature_processing, args=(feature_conf, img_path, img_name, features, feature_model, sfm_pairs, matcher_model, outputs))
        feature_thread.start()

        #extract_feature_one_image(feature_conf, img_path, img_name, features, feature_model)
        #logging.info('Extracted features')
        # ------------------ Feature matching ---------------------- #
        
        #sfm_matches = match_features_one_image(feature_conf['output'], 'query/test/' + img_name, sfm_pairs, matcher_model)
        

        #logging.info('Features matched')


        
        
        #extract_feature_one_image(retrieval_conf, img_path, img_name, global_descriptors, retrieval_model)
        #logging.info('Features extracted')

        # ------------------ Pairs from retireval ---------------------- #
        
        #pairs_from_retrieval.main(
        #    global_descriptors, loc_pairs, args.num_loc,
        #    query_prefix='query', db_model=reference_sfm, query_list = ['query/test/' + img_name], images = model_db_images, query_names = ['query/test/' + img_name])

        #logging.info('Pairs from retrieval done')

        #loc_matches = match_features.main(
        #    matcher_conf, loc_pairs, feature_conf['output'], outputs) # Need to do for the query image as well!

        # ------------------ Matching descriptors ---------------------- #
        #loc_matches = match_features_one_image(feature_conf['output'], 'query/test/' + img_name, loc_pairs, matcher_model)

        #logging.info('Features matched')

        # ------------------ Localization ---------------------- #
        global loc_matches
        global model_points3D
        query = Path(query_path)
        result_path = '/local/artwin/localization/Hierarchical-Localization/outputs/aachen/Aachen_hloc_superpoint+superglue_netvlad' + str(img_count - 1) + '.txt'
        results = Path(result_path)

        feature_thread.join()
        descriptor_thread.join()

        localize_sfm.main(
            reference_sfm,
            query,
            loc_pairs,
            features,
            loc_matches,
            results,
            covisibility_clustering=False,
            db_images = model_db_images,
            points3D = model_points3D)
        
        f = open(result_path, 'r')
        data = f.read()
        f.close()
        
        if data == '':
            print('Localization failed')
            response_list = [('file', filename), ('translation', [0,0,0]), ('rotation', [1,0,0,0])]

            response_dict = dict(response_list)
            
            return flask.jsonify(response_dict)

        logging.info('Localized')
        parsed = data.split(' ')

        translation = [float(parsed[5]), float(parsed[6]), float(parsed[7])]
        rotation = [float(parsed[1]), float(parsed[2]), float(parsed[3]), float(parsed[4])]
        response_list = [('imgname', imgname), ('translation', translation), ('rotation', rotation)] 

        response_dict = dict(response_list)
        
        return flask.jsonify(response_dict)
            


    except Exception as err:
        print('ko:', err)

    return 'ok'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Server for running Matlab script')
    parser.add_argument('--matlab_path',
                        required = False,
                        help = 'Path to the matlab.exe')
    parser.add_argument('--matlab_script_folder',
                        required = False,
                        help = 'Path to the folder containing the .m file')
    parser.add_argument('--matlab_function',
                        required = False,
                        help = 'Name of the Matlab function to run')
    parser.add_argument('--dataset', type=Path, default='/local/artwin/localization/Hierarchical-Localization/datasets/aachen',
                    help='Path to the dataset, default: %(default)s')
    parser.add_argument('--outputs', type=Path, default='/local/artwin/localization/Hierarchical-Localization/outputs/aachen',
                        help='Path to the output directory, default: %(default)s')
    parser.add_argument('--num_covis', type=int, default=20,
                        help='Number of image pairs for SfM, default: %(default)s')
    parser.add_argument('--num_loc', type=int, default=50,
                        help='Number of image pairs for loc, default: %(default)s')
    parser.add_argument('--query_path', type=Path, default = '/local/artwin/localization/Hierarchical-Localization/datasets/aachen/queries/one_query.txt',
                        help='Path to query txt file, default: %(default)s')
    parser.add_argument('--query_number', type=int, default=1,
                        help='Number of query, default: %(default)s')
    parser.add_argument('--query_saving_path', type=str, default = '/local/homes/zderaann',
                        help='Path to a folder where query files will be saved, default: %(default)s')


    args = parser.parse_args()

    global MATLAB_PATH
    MATLAB_PATH = args.matlab_path

    global MATLAB_SCRIPT
    MATLAB_SCRIPT_FOLDER = args.matlab_script_folder

    global MATLAB_FUNCTION
    MATLAB_FUNCTION = args.matlab_function

    load_hloc_data(args)

    IP = socket.gethostbyname(socket.gethostname())
    app.run(port = 443, host = '10.35.161.7')


