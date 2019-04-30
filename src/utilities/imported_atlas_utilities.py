import os
import sys

import json
import yaml


# from utilities2015 import *
# from registration_utilities import *
# from annotation_utilities import *
# from metadata import *
# from data_manager import *


# Load all structures
paired_structures = ['5N', '6N', '7N', '7n', 'Amb', 'LC', 'LRt', 'Pn', 'Tz', 'VLL', 'RMC', \
                     'SNC', 'SNR', '3N', '4N', 'Sp5I', 'Sp5O', 'Sp5C', 'PBG', '10N', 'VCA', 'VCP', 'DC']
# singular_structures = ['AP', '12N', 'RtTg', 'sp5', 'outerContour', 'SC', 'IC']
singular_structures = ['AP', '12N', 'RtTg', 'SC', 'IC']

# Make a list of all structures INCLUDING left and right variants
all_structures_total = list( singular_structures )
rh_structures = []
lh_structures = []
for structure in paired_structures:
    all_structures_total.append( structure+'_R' )
    all_structures_total.append( structure+'_L' )
    rh_structures.append( structure+'_R' )
    lh_structures.append( structure+'_L' )
#print all_structures_total


def create_alignment_specs( stack, detector_id ):
    fn_global = stack+'_visualization_global_alignment_spec.json'
    data = {}

    data["stack_m"] ={
            "name":"atlasV7",
            "vol_type": "score",
            "resolution":"10.0um"
            }
    data["stack_f"] ={
        "name":stack, 
        "vol_type": "score", 
        "resolution":"10.0um",
        "detector_id":detector_id
        }
    data["warp_setting"] = 0

    with open(fn_global, 'w') as outfile:
        json.dump(data, outfile)
        
    
    data = {}
    json_structure_list = []
    for structure in all_structures_total:

        data[structure] ={
            "stack_m": 
                {
                "name":"atlasV7", 
                "vol_type": "score", 
                "structure": [structure],
                "resolution":"10.0um"
                },
            "stack_f":
                {
                        "name":stack,
                        "vol_type": "score",
                        "structure":[structure],
                        "resolution":"10.0um",
                        "detector_id":detector_id
                        },
            "warp_setting": 7
            }

    fn_structure = stack+'_visualization_per_structure_alignment_spec.json'

    with open(fn_structure, 'w') as outfile:
        json.dump(data, outfile)
        
    return fn_global, fn_structure

# Load volumes, convert to proper coordinates, export as contours
def get_structure_contours_from_structure_volumes_v3(volumes, stack, sections, 
                                                     resolution, level, sample_every=1,
                                                    use_unsided_name_as_key=False):
    """
    Re-section atlas volumes and obtain structure contours on each section.
    Resolution of output contours are in volume resolution.
    v3 supports multiple levels.

    Args:
        volumes (dict of (3D array, 3-tuple)): {structure: (volume, origin_wrt_wholebrain)}. volume is a 3d array of probability values.
        sections (list of int):
        resolution (int): resolution of input volumes.
        level (float or dict or dict of list): the cut-off probability at which surfaces are generated from probabilistic volumes. Default is 0.5.
        sample_every (int): how sparse to sample contour vertices.

    Returns:
        Dict {section: {name_s: contour vertices}}.
    """

    from collections import defaultdict
    
    structure_contours_wrt_alignedBrainstemCrop_rawResol = defaultdict(lambda: defaultdict(dict))

    converter = CoordinatesConverter(stack=stack, section_list=metadata_cache['sections_to_filenames'][stack].keys())

    converter.register_new_resolution('structure_volume', resol_um=convert_resolution_string_to_um(resolution=resolution, stack=stack))
    converter.register_new_resolution('image', resol_um=convert_resolution_string_to_um(resolution='raw', stack=stack))
    
    for name_s, (structure_volume_volResol, origin_wrt_wholebrain_volResol) in volumes.iteritems():

        converter.derive_three_view_frames(name_s, 
        origin_wrt_wholebrain_um=convert_resolution_string_to_um(resolution=resolution, stack=stack) * origin_wrt_wholebrain_volResol,
        zdim_um=convert_resolution_string_to_um(resolution=resolution, stack=stack) * structure_volume_volResol.shape[2])

        positions_of_all_sections_wrt_structureVolume = converter.convert_frame_and_resolution(
        p=np.array(sections)[:,None],
        in_wrt=('wholebrain', 'sagittal'), in_resolution='section',
        out_wrt=(name_s, 'sagittal'), out_resolution='structure_volume')[..., 2].flatten()
            
        structure_ddim = structure_volume_volResol.shape[2]
        
        valid_mask = (positions_of_all_sections_wrt_structureVolume >= 0) & (positions_of_all_sections_wrt_structureVolume < structure_ddim)
        if np.count_nonzero(valid_mask) == 0:
#             sys.stderr.write("%s, valid_mask is empty.\n" % name_s)
            continue

        positions_of_all_sections_wrt_structureVolume = positions_of_all_sections_wrt_structureVolume[valid_mask]
        positions_of_all_sections_wrt_structureVolume = np.round(positions_of_all_sections_wrt_structureVolume).astype(np.int)
        
        if isinstance(level, dict):
            level_this_structure = level[name_s]
        else:
            level_this_structure = level

        if isinstance(level_this_structure, float):
            level_this_structure = [level_this_structure]
                        
        for one_level in level_this_structure:

            contour_2d_wrt_structureVolume_sectionPositions_volResol = \
            find_contour_points_3d(structure_volume_volResol >= one_level,
                                    along_direction='sagittal',
                                    sample_every=sample_every,
                                    positions=positions_of_all_sections_wrt_structureVolume)

            for d_wrt_structureVolume, cnt_uv_wrt_structureVolume in contour_2d_wrt_structureVolume_sectionPositions_volResol.iteritems():

                contour_3d_wrt_structureVolume_volResol = np.column_stack([cnt_uv_wrt_structureVolume, np.ones((len(cnt_uv_wrt_structureVolume),)) * d_wrt_structureVolume])

    #             contour_3d_wrt_wholebrain_uv_rawResol_section = converter.convert_frame_and_resolution(
    #                 p=contour_3d_wrt_structureVolume_volResol,
    #                 in_wrt=(name_s, 'sagittal'), in_resolution='structure_volume',
    #                 out_wrt=('wholebrain', 'sagittal'), out_resolution='image_image_section')

                contour_3d_wrt_alignedBrainstemCrop_uv_rawResol_section = converter.convert_frame_and_resolution(
                        p=contour_3d_wrt_structureVolume_volResol,
                        in_wrt=(name_s, 'sagittal'), in_resolution='structure_volume',
                        out_wrt=('wholebrainXYcropped', 'sagittal'), out_resolution='image_image_section')

                assert len(np.unique(contour_3d_wrt_alignedBrainstemCrop_uv_rawResol_section[:,2])) == 1
                sec = int(contour_3d_wrt_alignedBrainstemCrop_uv_rawResol_section[0,2])

                if use_unsided_name_as_key:
                    name = convert_to_unsided_label(name_s)
                else:
                    name = name_s

                structure_contours_wrt_alignedBrainstemCrop_rawResol[sec][name][one_level] = contour_3d_wrt_alignedBrainstemCrop_uv_rawResol_section[..., :2]
        
    return structure_contours_wrt_alignedBrainstemCrop_rawResol


def load_hdf_v2(fn, key='data'):
    import pandas
    return pandas.read_hdf(fn, key)

def load_sorted_filenames(stack):
    sorted_filenames_path = os.path.join( os.environ['ATLAS_DATA_ROOT_DIR'], 'CSHL_data_processed', stack, stack+'_sorted_filenames.txt' )
    
    with open(sorted_filenames_path, 'r') as sf:
        sorted_filenames_string = sf.read()
    sorted_filenames_list = sorted_filenames_string.split('\n')[0:len(sorted_filenames_string.split('\n'))-1]

    sorted_filenames = [{},{}] # filename_to_section, section_to_filename

    for sorted_fn_line in sorted_filenames_list:
        filename, slice_num = sorted_fn_line.split(' ')
        slice_num = int(slice_num)
        if filename == 'Placeholder':
            continue

        sorted_filenames[0][filename] = slice_num
        sorted_filenames[1][slice_num] = filename
        
    return sorted_filenames

def get_image_filepath_v2(stack, prep_id, resol, version, fn):
    image_filepath_root = os.path.join( os.environ['ATLAS_DATA_ROOT_DIR'], 'CSHL_data_processed', stack, \
                                       stack+'_prep'+str(prep_id)+'_'+resol+'_'+version )
    
    return os.path.join( image_filepath_root, fn+'_prep'+str(prep_id)+'_'+resol+'_'+version+'.tif'  )

def load_json(fp):
    with open(fp, 'r') as json_file:
        return json.load(json_file)

# The following functions are exclusively for loading volumes

# def volume_type_to_str(t):
#     if t == 'score':
#         return 'scoreVolume'
#     elif t == 'annotation':
#         return 'annotationVolume'
#     elif t == 'annotationAsScore':
#         return 'annotationAsScoreVolume'
#     elif t == 'annotationSmoothedAsScore':
#         return 'annotationSmoothedAsScoreVolume'
#     elif t == 'outer_contour':
#         return 'outerContourVolume'
#     elif t == 'intensity':
#         return 'intensityVolume'
#     elif t == 'intensity_metaimage':
#         return 'intensityMetaImageVolume'
#     else:
#         raise Exception('Volume type %s is not recognized.' % t)

# def get_original_volume_basename_v2(stack_spec):
#         """
#         Args:
#             stack_spec (dict):
#                 - prep_id
#                 - detector_id
#                 - vol_type
#                 - structure (str or list)
#                 - name
#                 - resolution
#         """

#         if 'prep_id' in stack_spec:
#             prep_id = stack_spec['prep_id']
#         else:
#             prep_id = None

#         if 'detector_id' in stack_spec:
#             detector_id = stack_spec['detector_id']
#         else:
#             detector_id = None

#         if 'vol_type' in stack_spec:
#             volume_type = stack_spec['vol_type']
#         else:
#             volume_type = None

#         if 'structure' in stack_spec:
#             structure = stack_spec['structure']
#         else:
#             structure = None

#         assert 'name' in stack_spec, stack_spec
#         stack = stack_spec['name']

#         if 'resolution' in stack_spec:
#             resolution = stack_spec['resolution']
#         else:
#             resolution = None

#         components = []
#         if prep_id is not None:
#             if isinstance(prep_id, str):
#                 components.append(prep_id)
#             elif isinstance(prep_id, int):
#                 components.append('prep%(prep)d' % {'prep':prep_id})
#         if detector_id is not None:
#             components.append('detector%(detector_id)d' % {'detector_id':detector_id})
#         if resolution is not None:
#             components.append(resolution)

#         tmp_str = '_'.join(components)
#         basename = '%(stack)s_%(tmp_str)s%(volstr)s' % \
#             {'stack':stack, 'tmp_str': (tmp_str+'_') if tmp_str != '' else '', 'volstr':volume_type_to_str(volume_type)}
#         if structure is not None:
#             if isinstance(structure, str):
#                 basename += '_' + structure
#             elif isinstance(structure, list):
#                 basename += '_' + '_'.join(sorted(structure))
#             else:
#                 raise

#         return basename
    
# def get_warped_volume_basename_v2(alignment_spec, trial_idx=None):
#         """
#         Args:
#             alignment_spec (dict): must have these keys warp_setting, stack_m and stack_f
#         """

#         warp_setting = alignment_spec['warp_setting']
#         basename_m = get_original_volume_basename_v2(alignment_spec['stack_m'])
#         basename_f = get_original_volume_basename_v2(alignment_spec['stack_f'])
#         vol_name = basename_m + '_warp%(warp)d_' % {'warp':warp_setting} + basename_f

#         if trial_idx is not None:
#             vol_name += '_trial_%d' % trial_idx

#         return vol_name

# def get_transformed_volume_filepath_v2(alignment_spec, resolution=None, trial_idx=None, structure=None):

#     if resolution is None:
#         if 'resolution' in alignment_spec['stack_m']:
#             resolution = alignment_spec['stack_m']['resolution']

#     if structure is None:
#         if 'structure' in alignment_spec['stack_m']:
#             structure = alignment_spec['stack_m']['structure']

#     warp_basename = get_warped_volume_basename_v2(alignment_spec=alignment_spec,
#                                                              trial_idx=trial_idx)
#     vol_basename = warp_basename + '_' + resolution
#     vol_basename_with_structure_suffix = vol_basename + ('_' + structure) if structure is not None else ''

#     return os.path.join(os.environ['ATLAS_DATA_ROOT_DIR'], alignment_spec['stack_m']['name'],\
#         vol_basename, 'score_volumes', vol_basename_with_structure_suffix + '.bp')

# def convert_volume_forms(volume, out_form):
#     """
#     Convert a (volume, origin) tuple into a bounding box.
#     """

#     if isinstance(volume, np.ndarray):
#         vol = volume
#         ori = np.zeros((3,))
#     elif isinstance(volume, tuple):
#         assert len(volume) == 2
#         vol = volume[0]
#         if len(volume[1]) == 3:
#             ori = volume[1]
#         elif len(volume[1]) == 6:
#             ori = volume[1][[0,2,4]]
#         else:
#             raise

#     bbox = np.array([ori[0], ori[0] + vol.shape[1]-1, ori[1], ori[1] + vol.shape[0]-1, ori[2], ori[2] + vol.shape[2]-1])

#     if out_form == ("volume", 'origin'):
#         return (vol, ori)
#     elif out_form == ("volume", 'bbox'):
#         return (vol, bbox)
#     elif out_form == "volume":
#         return vol
#     else:
#         raise Exception("out_form %s is not recognized.")
        
# def get_transformed_volume_origin_filepath(alignment_spec, structure=None, wrt='wholebrain', resolution=None):
#         """
#         Args:
#             alignment_spec (dict): specifies the multi-map.
#             wrt (str): specify which domain is the bounding box relative to.
#             resolution (str): specifies the resolution of the multi-map.
#             structure (str): specifies one map of the multi-map.
#         """

#         if resolution is None:
#             if 'resolution' in alignment_spec['stack_m']:
#                 resolution = alignment_spec['stack_m']['resolution']

#         if structure is None:
#             if 'structure' in alignment_spec['stack_m']:
#                 structure = alignment_spec['stack_m']['structure']

#         warp_basename = get_warped_volume_basename_v2(alignment_spec=alignment_spec, trial_idx=None)
#         vol_basename = warp_basename + '_' + resolution
#         vol_basename_with_structure_suffix = vol_basename + ('_' + structure if structure is not None else '')

#         return os.path.join(os.environ['ATLAS_DATA_ROOT_DIR'], alignment_spec['stack_m']['name'],
#                     vol_basename, 'score_volumes', vol_basename_with_structure_suffix + '_origin_wrt_' + wrt + '.txt')
        
# def load_data(filepath, filetype=None):
#         if not os.path.exists(filepath):
#             sys.stderr.write('File does not exist: %s\n' % filepath)

#         if filetype == 'bp':
#             return bp.unpack_ndarray_file(filepath)
#         elif filetype == 'image':
#             return imread(filepath)
#         elif filetype == 'hdf':
#             try:
#                 return load_hdf(filepath)
#             except:
#                 return load_hdf_v2(filepath)
#         elif filetype == 'bbox':
#             return np.loadtxt(filepath).astype(np.int)
#         elif filetype == 'annotation_hdf':
#             contour_df = read_hdf(filepath, 'contours')
#             return contour_df
#         elif filetype == 'pickle':
#             import cPickle as pickle
#             return pickle.load(open(filepath, 'r'))
#         elif filetype == 'file_section_map':
#             with open(filepath, 'r') as f:
#                 fn_idx_tuples = [line.strip().split() for line in f.readlines()]
#                 filename_to_section = {fn: int(idx) for fn, idx in fn_idx_tuples}
#                 section_to_filename = {int(idx): fn for fn, idx in fn_idx_tuples}
#             return filename_to_section, section_to_filename
#         elif filetype == 'label_name_map':
#             label_to_name = {}
#             name_to_label = {}
#             with open(filepath, 'r') as f:
#                 for line in f.readlines():
#                     name_s, label = line.split()
#                     label_to_name[int(label)] = name_s
#                     name_to_label[name_s] = int(label)
#             return label_to_name, name_to_label
#         elif filetype == 'anchor':
#             with open(filepath, 'r') as f:
#                 anchor_fn = f.readline().strip()
#             return anchor_fn
#         elif filetype == 'transform_params':
#             with open(filepath, 'r') as f:
#                 lines = f.readlines()

#                 global_params = one_liner_to_arr(lines[0], float)
#                 centroid_m = one_liner_to_arr(lines[1], float)
#                 xdim_m, ydim_m, zdim_m  = one_liner_to_arr(lines[2], int)
#                 centroid_f = one_liner_to_arr(lines[3], float)
#                 xdim_f, ydim_f, zdim_f  = one_liner_to_arr(lines[4], int)

#             return global_params, centroid_m, centroid_f, xdim_m, ydim_m, zdim_m, xdim_f, ydim_f, zdim_f
#         elif filepath.endswith('ini'):
#             return load_ini(fp)
#         else:
#             print('File type %s not recognized.\n' % filetype)
    
# def load_transformed_volume_v2( alignment_spec, resolution=None, structure=None, trial_idx=None,
#                                 return_origin_instead_of_bbox=False, legacy=False):
#     vol = load_data(get_transformed_volume_filepath_v2(alignment_spec=alignment_spec,
#                                                        resolution=resolution,
#                                                        structure=structure))

#     origin = load_data(get_transformed_volume_origin_filepath(wrt='fixedWholebrain',
#                                                               alignment_spec=alignment_spec,
#                                                               resolution=resolution,
#                                                               structure=structure))
#     if return_origin_instead_of_bbox:
#         return (vol, origin)
#     else:
#         return convert_volume_forms((vol, origin), out_form=('volume','bbox'))