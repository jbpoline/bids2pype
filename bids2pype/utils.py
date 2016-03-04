from __future__ import print_function, division
import os 
import os.path as osp
import shutil
import json
import fnmatch
import re
#import glob as gb
#from six import string_types
import numpy as np
import pandas as pd
from nipype.interfaces.base import Bunch


VERB  = {'none':10, 'info':3, 'warn':2, 'bug':1, 'critical':0}

LEVELS = {'Run', 'Session', 'Subject', 'Group'}
LEVELSJSON = {  'Run':'run_model', 'Session':'ses_model', 
                'Subject':'sub_model', 'Group':'group_model'}

REGKEYS = ['Variable', 'Level', 'HRFModelling', 'ModulationOrder', 'ModulationVar', 'Demean']
RUNMODKEYS = ['Level', 'DependentVariable', 'Columns', 'Error', 
              'Contrasts', 'HighPassFilterCutoff']
CONKEYS = ['Column', 'Statistic',  'Weights']

DEFAULTS_PAR = {'order_modulation': 1, 'high_pass_filter_cutoff': 120}


def _get_json_dict_from_file(json_file):
    """
    """
    assert osp.isfile(json_file), "{} is not a file".format(json_file)
    try: 
        with open(json_file) as fjson:
            json_dic = json.load(fjson)
    except: ValueError, " {} cannot be loaded by json module".format(json_file)

    return json_dic


def _get_dict_from_tsv_file(tsv_file):
    """
    """
    assert osp.isfile(tsv_file), "{} is not a file ".format(tsv_file)
    
    df = pd.read_csv(tsv_file, index_col=False, sep='\t')
    tsv_dict = df.to_dict(orient='list')

    return tsv_dict

def get_json_dict(json_model, level):
    """
    get the dict corresponding to this level from the json model file
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


def get_prefix_suffix(level):
    """
    """
    assert level in LEVELS, "{} not a model level".format(level)
    if level == 'Run':
        nii_Y_prefix = 'sub-*'
        nii_Y_suffix = '_run-*.nii.gz'
    else:
        raise NotImplementedError, " Level {} not implemented".format(level)

    return nii_Y_prefix, nii_Y_suffix

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


def _possible_dirpath_for_Ydata(dirname):
    """
    return True if this path can contain data
    """
    return True


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


def associate_model_data(base_dir, model_pattern, level='Run', verbose=VERB['none']):
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
        # this at the moment may return unwanted data : too liberal
        dict_level = get_json_dict(json_model, level)
        for data in list_of_data:
            # only associate data and model for data which have an event file:
            if _get_event_filename_for_run(data): 
                current_dict[data] = dict_level
    
    return {'data_dict':current_dict, 'base_dir':base_dir}

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


def _run_has_event_file(datafile):
    pass

def _get_event_filename_for_run(datafile):
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
    if osp.isfile(tsv_file): #, "{} is not a file ".format(tsv_file):
        return tsv_file
    else:
        return ''

def _get_bids_variables(base_dir, datafile, check=False):
    """
    """
    relative_file = datafile.split(base_dir)[-1]
    run = _get_substr_between(relative_file, '_run-', '_bold', check=check)
    sub = _get_substr_between(relative_file, 'sub-', '_task', check=check)
    grp = _get_substr_between(relative_file, 'grp-', '_', check=check)
    ses = _get_substr_between(relative_file, 'ses-', '_', check=check)

    bvar_dict = {'run': run, 'ses': ses, 'sub': sub, 'grp': grp}

    return bvar_dict


def _get_file_from_pattern(base_dir, file_pattern, bvar_dict):
    """
    input: 
    ------
    base_dir: string 
    pattern: string
    bvar_dict: dict 
        contains 'run', 'ses', 'sub', 'grp' keys

    output: 
    -------
    string
        the filename
    """
    # replicate dict but with int
    ivar_dict = {}
    for k in bvar_dict:
        if bvar_dict[k]:
            ivar_dict[k] = int(bvar_dict[k])

    # print(ivar_dict)
    
    # find the {[???]*} indicating formatting 
    sre = "(\{0\[.{3}\].*?})" 
    compiled = re.compile(sre)
    find_sre = compiled.findall(file_pattern)
    for substr in find_sre:
        # substr should be '{[run]:01d}', or '{[run]}'
        if substr[-2] == 'd': 
            newstr = substr.format(ivar_dict)
        else:
            newstr = substr.format(bvar_dict)
        file_pattern = file_pattern.replace(substr, newstr, 1) 
        # print(substr, newstr, file_pattern)
                                # 1: replace only first occurence

    return osp.join(base_dir, file_pattern)


def _get_other_reg_file_name(base_dir, datafile, pattern):
    """
    """
    bvar_dict = _get_bids_variables(base_dir, datafile)
    file_name = _get_file_from_pattern(base_dir, pattern, bvar_dict)

    return file_name

def _get_other_regressors(file_name, regressor, kreg, verbose=VERB['none']):
    """
    from a file that contains columns of values, construct an "other regressor"

    parameters:
    -----------
    file_name: string
        file that contains the other regressors to be included in the model
    regressor: dict
        the dict described in the json model file
        should have keys:
            "FileSelector" : how to get the file
            "Regressors" : which columns are we taking from this file 
    kreg: string
        the regressor key (name), eg: motion-param

    returns: 
    --------
    dict_regressors: dict
        keys: names of the other regressors
        values: {'group', 'values'}
                'group': indicate the name of the original set of regressors
                         coming from the same file
                'values': the values of the regressor, as many as volume in the
                          data to be analyzed
    """
    #print(file_name)
    dict_regressors = {} 
    #- create file name from pattern and variables
    # pattern = regressor['FileSelector']['pattern']
    assert osp.isfile(file_name)

    #- check that the file exists
    if not file_name:
        print("{} not a file, regressor: {} ".format(file_name, regressor))
        raise
    assert osp.isfile(file_name)

    if verbose <= VERB['info']: 
        print("_get_other_regressors: file_name {}, regressor: {}".format(
                                                        file_name, regressor))
    # Read this file, account for one column file
    motpars=np.loadtxt(file_name)
    if motpars.ndim == 1:
        nb_lines, nb_col  = 1, motpars.shape[0]
    elif motpars.ndim == 2:
        nb_lines, nb_col = motpars.shape
    else:
        print(" array from {} does not seem to be well, motpars.shape".format(
                                                      file_name, motpars.shape))
        raise
    # debug:print("nb_col, nb_lines", nb_col, nb_lines)

    # what do we do with it:
    assert "Regressors" in regressor
    to_add = regressor["Regressors"] # should be ["all", "deriv1"]
  
    # get the columns indices
    if "all" in to_add:
        idx = range(nb_col)
    else: # to be polished: how do we specify columns
        assert 'columns' in to_add[0]
        idx = to_add[0]['columns']

    #print("idx :" , idx)
    for i in idx:
        name = kreg+"_{:02d}".format(i+1)
        dict_regressors[name] = {}
        dict_regressors[name]['values'] =  motpars[:,i]
        dict_regressors[name]['group'] = kreg 
        if "deriv1" in to_add:
            deriv_name = kreg+"_{:02d}_deriv1".format(i+1)
            dict_regressors[deriv_name] = {}
            td = np.zeros(motpars.shape[0])
            td[1:] = motpars[1:,i] - motpars[:-1,i]
            dict_regressors[deriv_name]['values'] = td 

    return dict_regressors

def get_run_conditions(base_dir, datafile, model_dict, verbose=VERB['none']):
    """
    base_dir: the data base directory, eg /somethin/data/ds005
    datafile: should be a .nii or .nii.gz data
    model_dict: the run level part of the model

    returns conditions_names, and a list of 
    dictionaries (one per condition/trial) containing
    onsets, duration, HRF, ... for this model

    returns
    -------
    dict_regressors: dict
        keys are the name of the regressors, 
        values are dict containing onsets, duration, HRFModelling, etc
    dict_other_regressors: dict
        keys are name of other regressors (not to be convolved)
        values are dict with 'values', and possibly other information 
    logging:

    """

    # proper logging for latter ...
    logging = {}
   
    # for runs, the datafile is just a filename
    # check data exist ?
    assert osp.isfile(datafile), "{} is not a file".format(datafile)

    # get tsv filename:
    tsv_file = _get_event_filename_for_run(datafile)
    if not tsv_file:
        print("no tsv_file for {}".format(datafile))
        raise
    tsv_dict = _get_dict_from_tsv_file(tsv_file)

    # get condition names:
    # should trial_type be there if only one type of trial_type ?
    _check_keys_in({'onset', 'duration', 'trial_type'}, tsv_dict)
    _check_keys_in({'Columns'}, model_dict)
    #
    regressors = model_dict['Columns']
    dict_regressors = {} 
    dict_other_regressors = {} 

    for kreg in regressors:
        logging[kreg] = {}
        logging[kreg]['is_well'] = True 
        logging[kreg]['msg'] = '' 
        dict_regressors[kreg] = {}
        regressor = regressors[kreg]

        # other regressors if we have a FileSelector
        if 'FileSelector' in regressor: 
            _check_keys_in({'pattern'}, regressor['FileSelector'])
            pattern = regressor['FileSelector']['pattern']
            file_name = _get_other_reg_file_name(base_dir, datafile, pattern)
            # print("the file_name: ", file_name)
            other_regressors = _get_other_regressors(file_name, regressor, kreg, 
                                                                verbose=verbose)
            dict_other_regressors.update(other_regressors)
            # remove this kreg from dict_regressors
            dict_regressors.pop(kreg, None)
           
        else: #- this is a standard onset type of regressor, will be HRF convolved
            _check_keys_in({'Variable', 'HRFModelling','Level'}, regressor)

            if verbose <= VERB['info']: 
                print('\nRegress :', kreg, 
                      'regressor[Variable]: ', regressor['Variable'])

            dict_cond = {}
            dict_cond['HRF'] = regressor['HRFModelling']

            # First, get the lines through 'Variable' and 'Level':
            trial_level = regressor['Level']
            explanatory = regressor['Variable']
            col_bool, nothing_there = _get_tsv_lines(tsv_dict, 
                                                     explanatory, trial_level)

            if nothing_there:
                msg =  nothing_there + ' ! \n' + 'Removing key {} for {}'.format(
                                                                    kreg, datafile)
                # remove this regressor in the returned dictionary
                dict_regressors.pop(kreg, None)
                if verbose <= VERB['info']: 
                    print(msg)
                logging[kreg]['msg'] = msg
                logging[kreg]['is_well'] = False
                continue # skip that kreg

            # Second, get the values for these lines
            _check_keys_in({'onset', 'duration'}, tsv_dict)
            dict_cond['onset'] = _get_tsv_values(tsv_dict, 'onset', col_bool) 

            # if there is a "duration" key in the model for this regressor,
            # take it and overide values in tsv file
            if "Duration" in regressor:
                the_duration = regressor['Duration']
                dict_cond['duration'] = \
                                list((np.ones(col_bool.shape)*the_duration)[col_bool])
            else:
                dict_cond['duration'] = \
                                _get_tsv_values(tsv_dict, 'duration', col_bool) 

            # Any parametric modulation ? 'prm_modulation' corresponds to the 'weight'
            if 'ModulationVar' in regressor:
                weights = _get_tsv_values(tsv_dict, regressor['ModulationVar'], col_bool)
                if 'Demean' in regressor:
                    weights = np.asarray(weights).astype(float)
                    weights -= weights.mean()

                dict_cond['prm_modulation'] = list(weights)
                dict_cond['name_modulation'] = regressor['ModulationVar']
                dict_cond['order_modulation'] = DEFAULTS_PAR['order_modulation']
                if 'ModulationOrder' in regressor:
                    dict_cond['order_modulation'] = regressor['ModulationOrder']

            #no parametric modulation
            else:
                dict_cond['prm_modulation'] =  list(np.ones(col_bool.shape)[col_bool])
                dict_cond['name_modulation'] = None
                dict_cond['order_modulation'] = None

            # Any temporal modulation ?
            dict_cond['tmp_modulation'] = False
            if 'tmp_modulation' in regressor:
                dict_cond['tmp_modulation'] = regressor['ModulationTime']
            
            dict_regressors[kreg] = dict_cond
            if verbose <= VERB['info']: 
                print( "\n keys for regressor ", kreg, " are: ", dict_cond.keys())
                print('\n dict for regressor: ', dict_cond)

    return dict_regressors, dict_other_regressors, logging


def get_run_contrasts(model_dict):
    """
    parameters
    ----------
    model_dict: dict
        see description in ...

    returns
    -------
    dict_contrasts: dict
        a dict containing all necessary information for the contrasts to be exported
    """

    _check_keys_in({'Contrasts'}, model_dict)
    regressors = model_dict['Columns'] 
    contrast_dict = model_dict['Contrasts']
    dict_contrasts = {} 

    for con_name,val in contrast_dict.iteritems():
        # fill contrast dict
        contrast =  {}
        contrast['name'] = con_name
        contrast['conditions'] = val['Columns'] 
        # check contrast conditions are in regressors
        assert set(contrast['conditions']).issubset(set(regressors.keys())), \
                "{} not subset of {}".format(contrast['conditions'], regressors.keys())
        contrast['Weights'] = val['Weights']
        contrast['Statistic'] = val['Statistic'] 

        #- add to dict_contrasts
        dict_contrasts[con_name] = contrast

    return dict_contrasts


#------------------------------------------------------------------------------#
#-----------------------  Export functions to nipype from here ----------------#
#------------------------------------------------------------------------------#




def make_nipype_bunch(dict_regressors, other_reg, 
                                       bunch_type='spm', verbose=VERB['none']):
    """
    return a Bunch : the nipype input for model specification  
    so far : the spm bunch with pmod and tmod
    """

    # does it make sense to create a bunch from empty regressors ?
    assert dict_regressors, "dict_regressors input is empty: {}".format(dict_regressors)

    conditions = []
    onsets = []
    durations = []
    amplitudes = []
    pmod = []

    # condition_names = dict_regressors.keys()
    # mend the order of things condition names 

    for cond, dic in dict_regressors.items(): #zip(condition_names, dict_regressors):
        # dic = dict_regressors[cond]
        assert type(dic) == dict, "{} not a dict".format(dic)
        if verbose <= VERB['info']:
            print("\nmake_nipype_bunch cond : ",  cond, "dic : ", dic)
        conditions.append(cond),
        onsets.append(dic['onset'])
        durations.append(dic['duration'])

        #----- spm type of bunches ------#
        if bunch_type == 'spm':
            if dic['name_modulation']:
                pmod_name = dic['name_modulation']
                pmod_poly = dic['order_modulation']
                pmod_param = dic['prm_modulation']
                pmod.append([Bunch(name=pmod_name, 
                                   poly=pmod_poly, 
                                   param=pmod_param), None])
            else:
                pmod.append([])
        #----- fsl type of bunches ------#
        elif bunch_type == 'fsl':
            # here the parametric modulation values encodes the 
            # 'weights' or 'amplitudes'
            amplitudes.append(dic['prm_modulation'])
        else:
            print("unknown bunch type {}".format(bunch_type))
            raise

    regressor_names = []
    regressors = []
    if other_reg:
        for key, val in other_reg.items():
            regressor_names.append(key)
            regressors.append(val['values'])

    if bunch_type == 'spm': # pmod - tmod ?
        return Bunch(conditions=conditions, 
                     onsets=onsets, 
                     durations=durations, 
                     pmod=pmod,
                     regressor_names=regressor_names, 
                     regressors=regressors
                     )
    elif bunch_type == 'fsl':
        return Bunch(conditions=conditions, 
                     onsets=onsets, 
                     durations=durations,
                     amplitudes=amplitudes,
                     regressor_names=regressor_names, 
                     regressors=regressors
                     )

def _get_substr_between(thestring, after, before, check=True):
    """
    find things that are after after and before before :)
    example: 

    >>> _get_substr_between('++aaa_the_good_stuff_bbb++', 'aaa', 'bbb')
    '_the_good_stuff_'

    """
    # check that after and before are in thestring
    if check:
        assert after in thestring, "{} not in {}".format(after, thestring)
        assert before in thestring, "{} not in {}".format(before , thestring)
    if not (after in thestring and before in thestring):
        return ''
    # get what's after
    whatisafter = thestring.split(after)[-1]
    # get what's before
    between = whatisafter.split(before)[0] 
    return between
   

def _get_task_json_dict(base_dir, datafile):
    """
    get the task-???_bold.json dictionary corresponding to the datafile  
    These contain repetition time and task name
    """
    # get the task-X _bold.json
    taskname = _get_substr_between(datafile, 'task-', '_')
    # task_parameter_files = _rglob_sorted_by_depth(base_dir, '*'+taskname+'_bold.json')   
    # in the future: we might have a task-something lower in the hierarchy that
    # should replace the top level one
    # here problematic when files ._* exist
    task_parameter_files = _rglob_sorted_by_depth(base_dir, 'task-'+taskname+'_bold.json')   
    
    # should be only one file:
    if len(task_parameter_files) != 1:
        raise NotImplementedError, \
            "found  {},  len != 1 not implemented, taskname {} basedir {} ".format(
                    task_parameter_files, taskname, base_dir)

    task_dict = _get_json_dict_from_file(task_parameter_files[0])

    return task_dict

def _get_nipype_contrasts(model_dict):
    """

    Parameters
    ----------
    model_dict: dict
        The dict corresponding to the json model file
    """
    contrasts_dict = get_run_contrasts(model_dict)
    # format for nipype 
    list_con = []
    for con,val in contrasts_dict.iteritems():
        this_con = (con, val['Statistic'], val['conditions'], val['Weights'])
        list_con.append(this_con)

    return list_con

def _get_nipype_specify_model_inputs(base_dir, model_pattern, bunch_type='spm', 
                                               level='Run', verbose=VERB['none']): 
    """
    returns information ready for nipype specify_model:
    For spm: returns pmod style of bunch 

    parameters:
    -----------
    base_dir: string
    model_pattern: string
        a glob would give you the models
    bunch_type: string
        one of {'spm', 'fsl'}
    
    returns
    -------
    inputs_dict: dict 
        dict keys are:
        'time_repetition'
        'input_units'
        'high_pas_filter_cutoff'
    bunches: list
        list of Bunch object
        These objects contain the onsets, duration, 
                                    fsl: amplitudes, 
                                    spm: pmod / tmod Bunches  
    datafiles:
        the list of nii.gz files (as many as bunches)

    """

    assert level=='Run', "level {} not implemented".format(level)

    association_dict = associate_model_data(base_dir, model_pattern, 
                                            level=level, verbose=verbose)
    data_n_models = association_dict['data_dict']
    datafiles = data_n_models.keys()

    #------ params supposed to be unique across models: take the first one ---#

    # assumes for the moment high pass filter must be the same for all runs
    first_model = data_n_models[datafiles[0]]
    if  'HighPassFilterCutoff' in first_model:
        high_pass_filter_cutoff = first_model['HighPassFilterCutoff']
    else:
        high_pass_filter_cutoff = DEFAULTS_PAR['high_pass_filter_cutoff']

    # assumes for the moment task info is the same for all runs
    task_dict = _get_task_json_dict(base_dir, datafiles[0])

    inputs_dict={}
    inputs_dict['time_repetition'] = task_dict['RepetitionTime']
    inputs_dict['input_units'] = 'secs'
    inputs_dict['high_pass_filter_cutoff'] = high_pass_filter_cutoff

    # create a list of bunches, one per datafile
    bunches = []
    for datafile, model_dict in data_n_models.iteritems():
        #task_dict = _get_task_json_dict(base_dir, datafile)
        dict_regressors, other_reg, _log = get_run_conditions(base_dir,
                                                        datafile, model_dict,
                                                        verbose=verbose)
        if verbose <= VERB['info']:
            cond_with_issues = [k for k in _log if not _log[k]['is_well']]
            print('issue with keys {}'.format(cond_with_issues))
        bunches.append(make_nipype_bunch(dict_regressors, other_reg,
                                         bunch_type=bunch_type, verbose=verbose))

    # bunches here is the "info" in Russ' ds005 notebook 
    return inputs_dict, bunches, datafiles

# specify_model = pe.Node(interface=model.SpecifyModel(), name="specify_model")
# specify_model.inputs.input_units             = 'secs'
# specify_model.inputs.time_repetition         = 3.
# specify_model.inputs.high_pass_filter_cutoff = 120
# specify_model.inputs.subject_info = 


def create_empty_bids(source_dir, dest_dir, list_pattern, verbose=VERB['none']):
    """
    recursive walk in the source_dir: cp whatever is in pattern
    otherwise touch

    parameters:
    -----------
    source_dir: string
        the directory to be "copied"
    dest_dir: string
        where it is copied
    list_pattern: list
        will only actually copy these patterns (strings)
    
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


""" Notes ---- 

1- associate data and model.
    - find most top level model (or just one model)
    - instanciate default model_dict
    - get list of data
        * for runs, get the nii.gz
        * for ses, sub, grp : 
            find all ses, sub, or grp directories, create a unique key for each
            for each, find data as list of files
            associate key and list of files
    - once list of data is found, with a top level model, for each of these element go down 
        the directory tree and update the model if necessary

    ASSOCIATION OF DATA AND MODEL SHOULD BE COMPLEMENTED BY state_dict{'run','ses','sub','grp'}

2. Once data and model are associated:
    - Create an internal data structure to represent the model
        this would be one object per set of data
        To be implementated
        -----------------------------
            X demean 
            X get movement parameter regressors 
            - 

3. Take this internal data structure and export it in spm / fsl like type of nipype inputs


Outstanding question for Satra/Chris
-------------------------------------
    - seems that pmod structure has only 1 element even when there are 2 runs (2 onsets, etc)
    - the model specification that is generic to both contains specific spm stuff ?

"""




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
