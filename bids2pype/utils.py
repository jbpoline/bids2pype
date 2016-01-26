from __future__ import print_function, division
import os 
import os.path as osp
import json
import fnmatch
#import glob as gb
#from six import string_types
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

def get_run_conditions(datafile, model_dict):
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
    assert osp.isfile(tsv_file), "{} is not a file ".format(tsv_file)
    
    df = pd.read_csv(tsv_file, index_col=False, sep='\t')
    tsv_dict = df.to_dict(orient='list')

    # get condition names:
    regressors = model_dict['Columns']
    should_be_in_tsv_dict = {'onset', 'duration', 'trial_type'}
    for elt in should_be_in_tsv_dict:
        assert elt in tsv_dict, "{} not in {}".format(elt, regressors)

    dic_regressors = {} 
    for kreg in regressors:
        dic_regressors[kreg] = {}
        regressor = regressors[kreg]
        assert 'Level' in regressor, "Level not in {}".format(regressor)
        trial_level = regressor['Level']
        dict_cond = {}
        #dict_cond['name'] = kreg 
        dict_cond['onset'] =  \
                data_for_regressor(tsv_dict, 'onset', trial_level)
        dict_cond['duration'] =  \
                data_for_regressor(tsv_dict, 'duration', trial_level)
        dict_cond['HRF'] = regressor['HRF']
        dic_regressors[kreg] = dict_cond

    condition_names = regressors.keys()

    return condition_names, dic_regressors 


def get_nipype_run_info(datafile, model_dict, **kwargs):
    """
    returns what's needed by nipype: conditions, onsets, durations
    """
    
    condition_names, dic_regressors = get_run_conditions(datafile, model_dict)

    nipype_run_info = {}
    nipype_run_info['condition_names'] = condition_names
    nipype_run_info['onsets'] = [dic_regressors[cond]['onset'] for cond in condition_names]
    nipype_run_info['durations'] = [dic_regressors[cond]['duration'] for cond in condition_names]
    nipype_run_info['HRF'] = [dic_regressors[cond]['HRF'] for cond in condition_names]

    return nipype_run_info    
    


