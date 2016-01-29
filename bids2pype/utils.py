from __future__ import print_function, division
import os 
import os.path as osp
import json
import fnmatch
#import glob as gb
#from six import string_types
import numpy as np
import pandas as pd


VERB  = {'none':10, 'info':3, 'warn':2, 'bug':1, 'critical':0}
LEVELS = {'Run', 'Session', 'Subject', 'Group'}
LEVELSJSON = {  'Run':'run_model', 'Session':'ses_model', 
                'Subject':'sub_model', 'Group':'group_model'}

def get_prefix_suffix(level):
    """
    """
    assert level in LEVELS, "{} not a model level".format(level)
    if level == 'Run':
        nii_Y_prefix = 'sub-*'
        nii_Y_suffix = '_run-*.nii.gz'

    return nii_Y_prefix, nii_Y_suffix


def _get_json_dict_from_file(json_file):
    """
    """
    assert osp.isfile(json_file), "{} is not a file".format(json_file)
    try: 
        with open(json_file) as fjson:
            json_dic = json.load(fjson)
    except: ValueError, " {} cannot be loaded by json module".format(json_file)

    return json_dic


def _get_json_dict_from_tsv_file(tsv_file):
    """
    """
    assert osp.isfile(tsv_file), "{} is not a file ".format(tsv_file)
    
    df = pd.read_csv(tsv_file, index_col=False, sep='\t')
    tsv_dict = df.to_dict(orient='list')

    return tsv_dict

def get_json_dict(json_model, level):
    """
    get the dict corresponding to this level
    """

    try: 
        with open(json_model) as fjson:
            model_dic = json.load(fjson)
    except: ValueError, " {} cannot be loaded by json module".format(json_model)

    json_level = LEVELSJSON[level]

    # in this json file, get the level dictionary 
    if json_level in model_dic:
        assert model_dic[json_level]['Level'] == level, \
                    "{} not in {}".format(level, model_dic[json_level])
        dict_level = model_dic[json_level]
    else:
        assert model_dic['Level'] == level, \
                    "{} not in {}".format(level, model_dic)
        dict_level = model_dic

    return dict_level

def get_json_model_Ydata(json_model, level='Run', verbose=VERB['none']):
    """
    Reads a json model, then search in the base_dir to return the data
    or set of data to which the model should be applied

    """
    
    # json file like .../models/something.json, 
    basedir_to_search = osp.dirname(osp.dirname(json_model))
    if verbose <= VERB['info']: 
        print('base dir', basedir_to_search)
        print('json_model', json_model)

    dict_level = get_json_dict(json_model, level)

    if level == 'Run':
        returned_list = get_runs_data(basedir_to_search, dict_level)

#    if level == 'Session':
#        returned_list = get_sessions_data(basedir_to_search, dict_level)
        
    return returned_list 


def get_runs_data(basedir, model_dic, verbose=VERB['none']):
    """
    search for the runs specified in model_dic in this base directory 
    """
    data_key = 'DependentVariable'
    assert 'DependentVariable' in model_dic, "{} not in {}".format(data_key, model_dic)

    nii_Y_prefix, nii_Y_suffix = get_prefix_suffix('Run')
    nii_to_search = nii_Y_prefix + model_dic['DependentVariable'] + nii_Y_suffix
    if verbose <= VERB['warn']: print(nii_to_search)

    return glob_recursive(basedir, nii_to_search)


def glob_recursive(source_dir, pattern):
    """
    recursive glob in the source_dir
    """
    assert osp.isdir(source_dir), \
            '{} does not seem to be a directory'.format(source_dir)
    matches = []
    for root, dirnames, filenames in os.walk(source_dir):
        for filename in fnmatch.filter(filenames, pattern):
            matches.append(os.path.join(root, filename))
    return matches


def get_funcs_models(base_dir, model_pattern, level='Run', verbose=VERB['none']):
    """
    This function creates the link between a given nii or nii.gz
    file and the model that should be apply to it at the run level
    

    parameters:
    -----------
    base_dir: string
        the base directory of the bids data
    model_pattern: 
        glob pattern to identify model files

    output
    ------
    dictionary
        Contains the link between nii files to be processed (keys)
        and their model
    """
    def sort_by_dir_len(x):
        return(len(osp.dirname(x)))
    
    all_jsons = glob_recursive(base_dir, model_pattern)
    # return the list of directory by order of most root to most leaf
    all_jsons_sorted = sorted(all_jsons, key=sort_by_dir_len)

    current_dict = {}
    for json_model in all_jsons_sorted:
        list_of_data = get_json_model_Ydata(json_model, level=level, verbose=verbose)
        dict_level = get_json_dict(json_model, level)
        for data in list_of_data:
            current_dict[data] = dict_level
    
    return current_dict

def data_for_regressor(tsv_dict, datatype, trial):
    """
    get the onsets or the duration (called datatype) for this trial
    """
    assert datatype in tsv_dict, \
                "{} is not in dict {}".format(datatype, tsv_dict)
    assert trial in  tsv_dict['trial_type'], \
                "There is no {} in {}".format(trial, tsv_dict['trial_type'])

    trial_data = [ tsv_dict[datatype][i] \
            for i,_trial in enumerate(tsv_dict['trial_type']) if _trial == trial]

    return trial_data

def _check_keys_in(keys, somewhere):
    """
    """
    for elt in keys:
        assert elt in somewhere, "{} not in {}".format(elt, somewhere)


def _get_tsv_lines(tsv_dict, column_name, trial):
    """
    This function takes a tsv dictionary, a column name, a trial type,
    and returns the lines of the tsv for corresponding to trial for that column
    """
    assert column_name in tsv_dict,  \
                "There is no {} in {}".format(column_name, tsv_dict.keys())

    column_data = np.asarray(tsv_dict[column_name])
    if trial == 'n/a':
        col_bool =  np.ones(column_data.shape, dtype=bool)
    else: 
        col_bool = column_data == trial 

    # for the moment, fails if no trial of that type 
    assert np.any(col_bool), \
            "{} column has no {}".format(column_name, trial)

    return col_bool

def _get_tsv_values(tsv_dict, column_name, col_bool):
    """
    """
    assert column_name in tsv_dict,  \
                "There is no {} in {}".format(column_name, tsv_dict.keys())
    col_array = np.asarray(tsv_dict[column_name])
    col_values = col_array[col_bool]
    assert len(col_values) > 0, \
            "no values for {}".format(column_name, tsv_dict.keys()) 
    return list(col_values) 

def get_run_conditions(datafile, model_dict, verbose=VERB['none']):
    """
    datafile: should be a .nii or .nii.gz data
    model_dict: the run level part of the model

    returns conditions_names, and a list of 
    dictionaries (one per condition/trial) containing
    onsets, duration, HRF, ... for this model
    """
   
    # for runs, the datafile is just a filename
    # check data exist ?
    assert osp.isfile(datafile), "{} is not a file".format(datafile)

    # get tsv filename:
    tsv_file = datafile.split('_bold.nii')[0] + '_events.tsv'
    tsv_dict = _get_json_dict_from_tsv_file(tsv_file)

    # get condition names:
    # should trial_type be there if only one type of trial_type ?
    _check_keys_in({'onset', 'duration', 'trial_type'}, tsv_dict)
    _check_keys_in({'Columns'}, model_dict)
    #
    regressors = model_dict['Columns']
    dict_regressors = {} 
    for kreg in regressors:
        dict_regressors[kreg] = {}
        regressor = regressors[kreg]
        _check_keys_in({'Variable', 'HRFmodelling','Level'}, regressor)

        if verbose <= VERB['info']: 
            print('\nregressor[Variable]: ', regressor['Variable'])

        dict_cond = {}
        dict_cond['HRF'] = regressor['HRFmodelling']

        # First, get the columns through 'Variable' and 'Level':
        trial_level = regressor['Level']
        explanatory = regressor['Variable']
        col_bool = _get_tsv_lines(tsv_dict, explanatory, trial_level)

        # Second, get the values for these lines
        _check_keys_in({'onset', 'duration'}, tsv_dict)
        dict_cond['onset'] = _get_tsv_values(tsv_dict, 'onset', col_bool) 
        dict_cond['duration'] = _get_tsv_values(tsv_dict, 'duration', col_bool) 

        # Any parametric modulation ?
        dict_cond['prm_modulation'] = None
        if 'prm_modulation' in regressor:
            dict_cond['prm_modulation'] = \
                        _get_tsv_values(tsv_dict, regressor['prm_modulation'], col_bool) 

        # Any temporal modulation ?
        dict_cond['tmp_modulation'] = None
        if 'tmp_modulation' in regressor:
            dict_cond['tmp_modulation'] = regressor['tmp_modulation']
        
        dict_regressors[kreg] = dict_cond
        if verbose <= VERB['info']: 
            print(kreg, dict_cond.keys())
            print('\ndict for this variable: ', dict_cond)

    condition_names = regressors.keys()

    return condition_names, dict_regressors 


def get_nipype_run_info(datafile, model_dict, verbose=VERB['none'], **kwargs):
    """
    returns what's needed by nipype: conditions, onsets, durations
    """
    condition_names, dict_regressors = get_run_conditions(datafile, model_dict, verbose=verbose)

    nipype_run_info = {}
    nipype_run_info['condition_names'] = condition_names
    nipype_run_info['onsets'] = [dict_regressors[cond]['onset'] for cond in condition_names]
    nipype_run_info['durations'] = [dict_regressors[cond]['duration'] for cond in condition_names]
    nipype_run_info['prm_modulation'] = \
                [dict_regressors[cond]['prm_modulation'] for cond in condition_names]
    nipype_run_info['tmp_modulation'] = \
                [dict_regressors[cond]['tmp_modulation'] for cond in condition_names]
    nipype_run_info['HRF'] = [dict_regressors[cond]['HRF'] for cond in condition_names]

    return nipype_run_info    

