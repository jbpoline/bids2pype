from __future__ import print_function, division
from .. import utils
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
    'base_dir': '/home/jb/data/bids/ds005',
    'model_pattern': "*_model.json"
}
test_case_005r = {
    'base_dir': '/home/jb/data/bids/ds005',
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
    

