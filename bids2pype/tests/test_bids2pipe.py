from __future__ import print_function, division
from .. import utils
import numpy as np
import os 
import os.path as osp
import json


test_case_105a = {
    'base_dir': '/home/jb/data/bids/ds105',
    'model_pattern': "ds-105_level-run_sub*_model.json"
}
test_case_105b = {
    'base_dir': '/home/jb/data/bids/ds105',
    'model_pattern': "ds-105_level-all_model.json"
}
test_case_005 = {
    #'base_dir': '/home/jb/data/bids/ds005',
    'base_dir': './data/ds005',
    'model_pattern': "*_model.json"
}
test_case_005r = {
    #'base_dir': '/home/jb/data/bids/ds005',
    'base_dir': 'data/ds005',
    'model_pattern': "ds-005_type-russ*_model.json"
}

#test_case = test_case_005
test_case = test_case_005r
#test_case = test_case_105a
#test_case = test_case_105b

base_dir, model_pattern = test_case['base_dir'], test_case['model_pattern']

def test_associate_model_data():
    """
    """
    pattern_mvt = u'derivatives/mcflirt/par/_runcode_{0[run]:1d}/' + \
                    u'_subject_id_sub-{0[sub]}/sub-{0[sub]}' + \
                    u'_task-mixedgamblestask_run-{0[run]}_bold_mcf.nii.gz.par'

    expected = {u'Columns': { 
                    u'motion-param': { 
                            u'FileSelector': { 
                                    u'EntitiesKeys': { 
                                            u'run': u'run', 
                                            u'subject': u'sub'},
                                    u'pattern': 
                                            pattern_mvt
                                        },

                            u'Regressors': [u'all', u'deriv1']},

                    u'param-gain': { u'Demean': True,
                                     u'HRFModelling': u'Gamma+derivs',
                                     u'Level': u'n/a',
                                     u'ModulationOrder': 1,
                                     u'ModulationVar': u'gain',
                                     u'Variable': u'trial_type',
                                     u'duration': 1.3932149481304148},
                    u'param-loss': { u'Demean': True,
                                     u'HRFModelling': u'Gamma+derivs',
                                     u'Level': u'n/a',
                                     u'ModulationVar': u'loss',
                                     u'Variable': u'trial_type',
                                     u'duration': 1.3932149481304148},
                    u'param-rt': { u'Demean': True,
                                   u'HRFModelling': u'Gamma+derivs',
                                   u'Level': u'n/a',
                                   u'ModulationVar': u'RT',
                                   u'Variable': u'trial_type',
                                   u'duration': 1.3932149481304148},
                    u'task': { u'HRFModelling': u'Gamma',
                               u'Level': u'n/a',
                               u'Variable': u'trial_type',
                               u'duration': 1.3932149481304148}},
      u'Contrasts': { u'param-gain': { u'Columns': [u'param-gain'],
                                       u'Statistic': u'T',
                                       u'Weights': [1]},
                      u'param-loss-neg': { u'Columns': [u'param-loss'],
                                           u'Statistic': u'T',
                                           u'Weights': [-1]},
                      u'task>Baseline': { u'Columns': [u'task'],
                                          u'Statistic': u'T',
                                          u'Weights': [1]}},
      u'DependentVariable': u'task-mixedgamblestask',
      u'Error': { u'SerialCorrelations': True},
      u'HighPassFilterCutoff': 80,
      u'Level': u'Run'}

    assos_model_data = utils.associate_model_data(base_dir, model_pattern, 
                                                      verbose=utils.VERB['none'])

    data_dict = assos_model_data['data_dict']
    #print(data_dict)
    datafile0 = sorted(data_dict.keys())[0]
    read_from_disk  = data_dict[datafile0]
    
    expected_datafile = base_dir + '/sub-01/func/sub-01_task-mixedgamblestask_run-01_bold.nii.gz'
    assert datafile0 ==  expected_datafile


    list1 = read_from_disk['Columns'].keys()
    list2 = expected['Columns'].keys()
    assert set(list1) == set(list2)

    exp_contrasts = expected['Contrasts'].keys()
    read_contrasts = data_dict[datafile0]['Contrasts'].keys()
    assert set(exp_contrasts) == set(read_contrasts)
   

def test_get_nipype_specify_model_inputs(): 

    specifymodel_inputs, bunches, data = \
        utils._get_nipype_specify_model_inputs(base_dir, model_pattern, bunch_type='fsl', 
                                                                     verbose=utils.VERB['none'])

    sorted_bunch = [b for (d,b) in sorted(zip(data, bunches))]
    sorted_data = sorted(data)

    expected_data0 = base_dir + '/sub-01/func/sub-01_task-mixedgamblestask_run-01_bold.nii.gz'
    print(expected_data0)
    assert expected_data0 == sorted_data[0]
    
    exp_param_gain = np.loadtxt('./test_cond_amplitude.txt').astype(float)
    exp_param_gain -= exp_param_gain.mean()
    index_gain = sorted_bunch[0].conditions.index('param-gain')
    read_param_gain = np.asarray(sorted_bunch[0].amplitudes[index_gain])
    #print(read_param_gain, exp_param_gain)
    assert np.linalg.norm(read_param_gain - exp_param_gain) < 1.e-12
    
def test__get_dict_from_tsv_file():
    pass


def test_get_other_regressors():
    """
    
    specifymodel_inputs, bunches, data = \
        utils._get_nipype_specify_model_inputs(base_dir, model_pattern, 
                                bunch_type='fsl', verbose=utils.VERB['none'])

    """

    specifymodel_inputs, bunches, data = \
        utils._get_nipype_specify_model_inputs(base_dir, model_pattern, \
                                bunch_type='fsl', verbose=utils.VERB['none'])

    # sort both bunch and data to get predictable output
    sorted_bunch = [b for (d,b) in sorted(zip(data, bunches))]
    sorted_data = sorted(data)

    assert sorted_data[0] == base_dir + \
                '/sub-01/func/sub-01_task-mixedgamblestask_run-01_bold.nii.gz'
    assert sorted_data[2] == base_dir + \
                '/sub-01/func/sub-01_task-mixedgamblestask_run-03_bold.nii.gz'
    assert sorted_data[-1] == base_dir + \
                '/sub-16/func/sub-16_task-mixedgamblestask_run-03_bold.nii.gz'
    assert sorted_bunch[0].regressor_names[0] == "motion-param_01"
    assert sorted_bunch[0].regressor_names[1] == "motion-param_02"

    # read file corresponding to the first sub run mvt param:
    mvt = np.loadtxt(base_dir + \
        '/derivatives/mcflirt/par/_runcode_1/'+\
        '_subject_id_sub-01/sub-01_task-mixedgamblestask_run-01_bold_mcf.nii.gz.par')

    for col_idx, col_mvt in enumerate(mvt.T): 
        assert np.linalg.norm(col_mvt - sorted_bunch[0].regressors[col_idx]) < 1.e-12


    # read file corresponding to sub 01 run 03 mvt param:
    mvt_file = base_dir + \
        '/derivatives/mcflirt/par/_runcode_3/'+\
        '_subject_id_sub-01/sub-01_task-mixedgamblestask_run-03_bold_mcf.nii.gz.par'
    print(mvt_file)
    mvt = np.loadtxt(mvt_file) 


    for col_idx, col_mvt in enumerate(mvt.T): 
        assert np.linalg.norm(col_mvt - sorted_bunch[2].regressors[col_idx]) < 1.e-12


