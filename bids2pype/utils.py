from __future__ import print_function, division
import os 
import os.path as osp
import shutil
import json
import fnmatch
#import glob as gb
#from six import string_types
import numpy as np
import pandas as pd
from nipype.interfaces.base import Bunch


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

    else:
        print("Level {} not yet implemented".format(level))
        raise NotImplementedError

#    if level == 'Session':
#        returned_list = get_sessions_data(basedir_to_search, dict_level)
        
    return returned_list 


def get_runs_data(basedir, model_dic, verbose=VERB['none']):
    """
    search for the runs specified in model_dic in this base directory 
    """
    data_key = 'DependentVariable'
    assert 'DependentVariable' in model_dic, \
                                "{} not in {}".format(data_key, model_dic)

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
    for dirpath, dirnames, filenames in os.walk(source_dir):
        for filename in fnmatch.filter(filenames, pattern):
            matches.append(osp.join(dirpath, filename))
    return matches


def _rglob_sorted_by_depth(base_dir, pattern):
    """
    recursively find files with pattern in base_dir
    return a list of files found, from root to leaf 
    """
    def sort_by_dir_len(x):
        return(len(osp.dirname(x)))
    
    filenames = glob_recursive(base_dir, pattern)
    # return the list of directory by order of most root to most leaf

    return sorted(filenames, key=sort_by_dir_len)


def get_funcs_models(base_dir, model_pattern, level='Run', verbose=VERB['none']):
    """
    This function creates the link between a given nii or nii.gz
    filename and the model that should be apply to it at the run level
    
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
    all_jsons_sorted = _rglob_sorted_by_depth(base_dir, model_pattern)
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
    message = ""
    assert column_name in tsv_dict,  \
                "There is no {} in {}".format(column_name, tsv_dict.keys())

    column_data = np.asarray(tsv_dict[column_name])
    if trial == 'n/a':
        col_bool =  np.ones(column_data.shape, dtype=bool)
    else: 
        col_bool = column_data == trial 

    # for the moment, fails if no trial of that type 
    all_fine =  np.any(col_bool) 
    if not all_fine:
        message =  "\n{} column has no {}".format(column_name, trial)

    return col_bool, message

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


def _get_event_file_for_run(datafile):
    """
    input: 
    ------
    datafile: string
        a *_bold.nii.gz file
    output: 
    -------
    string
        corresponding filename
    """
    tsv_file = datafile.split('_bold.nii')[0] + '_events.tsv'
    assert osp.isfile(tsv_file), "{} is not a file ".format(tsv_file)
    return tsv_file


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
    tsv_file = _get_event_file_for_run(datafile)
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

        # First, get the lines through 'Variable' and 'Level':
        trial_level = regressor['Level']
        explanatory = regressor['Variable']
        col_bool, msg = _get_tsv_lines(tsv_dict, explanatory, trial_level)
        if msg:
            print(msg)
            print('removing key {} for {}'.format(kreg, datafile))
            # remove this regressor
            dict_regressors.pop(kreg, None)
            break

        # Second, get the values for these lines
        _check_keys_in({'onset', 'duration'}, tsv_dict)
        dict_cond['onset'] = _get_tsv_values(tsv_dict, 'onset', col_bool) 
        dict_cond['duration'] = _get_tsv_values(tsv_dict, 'duration', col_bool) 

        # Any parametric modulation ?
        dict_cond['prm_modulation'] = False 
        if 'ModulationVar' in regressor:

            dict_cond['prm_modulation'] = \
                        _get_tsv_values(tsv_dict, regressor['ModulationVar'], col_bool) 
            dict_cond['name_modulation'] = regressor['ModulationVar']
            dict_cond['order_modulation'] = 1
            if 'ModulationOrder' in regressor:
                dict_cond['order_modulation'] = regressor['ModulationOrder']

        # Any temporal modulation ?
        dict_cond['tmp_modulation'] = False
        if 'tmp_modulation' in regressor:
            dict_cond['tmp_modulation'] = regressor['ModulationTime']
        
        dict_regressors[kreg] = dict_cond
        if verbose <= VERB['info']: 
            print(kreg, dict_cond.keys())
            print('\ndict for this variable: ', dict_cond)

    #condition_names = regressors.keys()

    return dict_regressors 

def get_run_contrasts(model_dict):
    """
    """
    _check_keys_in({'Contrasts'}, model_dict)
    regressors = model_dict['Columns'] 
    contrast_dict = model_dict['Contrasts']
    dict_contrasts = {} 
    for con_name,val in contrast_dict.iteritems():
        dict_contrasts[con_name] = {}
        contrast =  dict_contrasts[con_name]       
        contrast['name'] = con_name
        contrast['conditions'] = val['Columns'] 
        assert set(contrast['conditions']).issubset(set(regressors.keys())), \
                "{} not subset of {}".format(contrast['conditions'], regressors.keys())
        contrast['Weights'] = val['Weights']
        contrast['Statistic'] = 'T' 

    return dict_contrasts


# def get_nipype_run_info(datafile, model_dict, verbose=VERB['none'], **kwargs):
#     """
#     returns what's needed by nipype: conditions, onsets, durations
#     """
#     dict_regressors = get_run_conditions(datafile, model_dict, verbose=verbose)
#     condition_names = dict_regressors.keys()
# 
#     nipype_run_info = {}
#     nipype_run_info['condition_names'] = condition_names
#     nipype_run_info['onsets'] = [dict_regressors[cond]['onset'] for cond in condition_names]
#     nipype_run_info['durations'] = [dict_regressors[cond]['duration'] for cond in condition_names]
#     nipype_run_info['prm_modulation'] = \
#                 [dict_regressors[cond]['prm_modulation'] for cond in condition_names]
#     nipype_run_info['tmp_modulation'] = \
#                 [dict_regressors[cond]['tmp_modulation'] for cond in condition_names]
#     nipype_run_info['HRF'] = [dict_regressors[cond]['HRF'] for cond in condition_names]
# 
#     return nipype_run_info


def make_nipype_bunch(dict_regressors, verbose=VERB['none']):
    """
    return a Bunch : the nipype input  
    """

    conditions = []
    onsets = []
    durations = []
    pmod = []

    condition_names = dict_regressors.keys()

    for cond, kdic in zip(condition_names, dict_regressors):
        dic = dict_regressors[kdic]
        assert type(dic) == dict, "{} not a dict".format(dic)
        if verbose <= VERB['info']:
            print(cond, dic)
        conditions.append(cond),
        onsets.append(dic['onset'])
        durations.append(dic['duration'])
        if dic['prm_modulation']:
            pmod_name = dic['name_modulation']
            pmod_poly = dic['order_modulation']
            pmod_param = dic['prm_modulation']
            pmod.append([Bunch(name=pmod_name, poly=pmod_poly, param=pmod_param), None])
        else:
            pmod.append([])
    
    return Bunch(conditions=conditions, 
                 onsets=onsets, 
                 durations=durations, 
                 pmod=pmod)


def _get_substr_between(thestring, after, before):
    """
    """
    # check that after and before are in thestring
    assert after in thestring, "{} not in {}".format(after, thestring)
    assert before in thestring, "{} not in {}".format(before , thestring)
    # get what's after
    whatisafter = thestring.split(after)[-1]
    # get what's before
    between = whatisafter.split(before)[0] 
    return between
   

def _get_task_json_dict(base_dir, datafile):
    """
    get the task-???_bold.json dictionary corresponding to the datafile  
    """
    # get the task-X _bold.json
    taskname = _get_substr_between(datafile, 'task-', '_')
    task_parameter_files = _rglob_sorted_by_depth(base_dir, '*'+taskname+'_bold.json')   
    assert len(task_parameter_files) == 1, \
            "found  {},  len > 1 not implemented".format(task_parameter_files)
    task_dict = _get_json_dict_from_file(task_parameter_files[0])
    return task_dict

def _get_nipype_contrasts(model_dict):
    """
    """
    contrasts_dict = get_run_contrasts(model_dict)
    # format for nipype 
    list_con = []
    for con,val in contrasts_dict.iteritems():
        this_con = (con, val['Statistic'], val['conditions'], val['Weights'])
        list_con.append(this_con)

    return list_con

def _get_nipype_specify_model_inputs(base_dir, model_pattern, 
                                               level='Run', verbose=VERB['none']): 
    """
    returns information ready for nipype specify_model:
    
    returns
    -------
    inputs_dict: dict 
        dict keys are:
        'time_repetition'
        'input_units'
        'high_pas_filter_cutoff'

    bunches: list
        list of Bunch object
        These objects contain the onsets, duration, etc 

    datafiles:
        the list of nii.gz files (as many as bunches)
    """

    assert level=='Run', "level {} not implemented".format(level)

    data_n_models = get_funcs_models(base_dir, model_pattern, level=level, verbose=verbose)
    datafiles = data_n_models.keys()

    # assumes for the moment high pass filter must be the same for all runs
    first_model = data_n_models[datafiles[0]]
    if  'HighPassFilterCutoff' in first_model:
        high_pass_filter_cutoff = first_model['HighPassFilterCutoff']
    else:
        high_pass_filter_cutoff = 120

    # assumes for the moment task info is the same for all runs
    task_dict = _get_task_json_dict(base_dir, datafiles[0])

    inputs_dict={}
    inputs_dict['time_repetition'] = task_dict['RepetitionTime']
    inputs_dict['input_units'] = 'secs'
    inputs_dict['high_pass_filter_cutoff'] = high_pass_filter_cutoff

    # assumes contrasts all the same for all runs

    
    # create a list of bunches, one per datafile
    bunches = []
    for datafile, model_dict in data_n_models.iteritems():
        #task_dict = _get_task_json_dict(base_dir, datafile)
        dict_regressors = get_run_conditions(datafile, model_dict, verbose=verbose)
        bunches.append(make_nipype_bunch(dict_regressors, verbose=verbose))
         
    return inputs_dict, bunches, datafiles

# specify_model = pe.Node(interface=model.SpecifyModel(), name="specify_model")
# specify_model.inputs.input_units             = 'secs'
# specify_model.inputs.time_repetition         = 3.
# specify_model.inputs.high_pass_filter_cutoff = 120
# specify_model.inputs.subject_info = 



def create_empty_bids(source_dir, dest_dir, list_pattern, verbose=VERB['none']):
    """
    recursive walk in the source_dir
    cp whatever is in pattern
    otherwise touch
    """
    def _mkdir(_dir):
        if not osp.isdir(_dir):
            try:
                os.makedirs(_dir)
            except:
                 print("cannot create directory {}".format(_dir))
                 raise

    def _touch(fname, times=None):
        try:
            with open(fname, 'a'):
                os.utime(fname, times)
        except:
            print(fname)
            raise

    assert osp.isdir(source_dir), \
            '{} does not seem to be a directory'.format(source_dir)

    for  dirpath, dirnames, filenames in os.walk(source_dir):
        newpath = dirpath.replace(source_dir, dest_dir)
        _mkdir(dirpath)
        for dirname in dirnames:
            _mkdir(osp.join(newpath, dirname))
        for _file in filenames:
            for pattern in list_pattern:
                # print(_file, pattern, fnmatch.filter([_file], pattern))
                if _file in fnmatch.filter([_file], pattern):
                    if verbose <= VERB['info']: print("copy ", _file)
                    shutil.copy(osp.join(dirpath, _file), osp.join(newpath, _file))
                    break
                else:
                    _touch(osp.join(newpath, _file))


