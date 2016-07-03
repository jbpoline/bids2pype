

# def fill_Ydata_dict(json_model, nii_Y_prefix, nii_Y_suffix, the_dict={}, verbose=-1):
#     """
#     take a model json file, search for the 4d nii files that should be
#     processed with this model, and fills a dict with the dependant variable nii
#     as keys, and the dictionnary containing the model to be run on this
#     dependant variable as value
# 
#     for run level:
#         nii_Y_prefix = 'sub-*'
#         nii_Y_suffix = '_run-*.nii.gz'
# 
#     Parameters
#     ----------
#     json_model: string
#         the jason file containing a model
#   
#     Returns:
#     --------
#     the_dict: dictionary
#         contains the data as keys and model as value
#     """
#     
#     # directory from which to search for the data: assumes lower 
#     # in the disk hierarchy than the json model file
#     
#     nii_base_dir = osp.dirname(osp.dirname(json_model))
#     if verbose >= VERB['info']: 
#         print('base dir', nii_base_dir)
#         print('json_model', json_model)
# 
#     try: 
#         with open(json_model) as fjson:
#             model_dic = json.load(fjson)
#     except: ValueError, " {} cannot be loaded by json module".format(json_model)
#         
#     nii_to_search = nii_Y_prefix + model_dic['DependentVariable'] + nii_Y_suffix
#     if verbose >= VERB['warn']: print(nii_to_search)
#         
#     niis_for_this_model = glob_recursive(nii_base_dir, nii_to_search)
#     
#     for nii in niis_for_this_model:
#         the_dict[nii] = model_dic
#     
#     if verbose >= VERB['info'] : print("\n".join(niis_for_this_model))
#     
#     return the_dict
# 

#  import csv
#  import re
#  
#  the_dict = {}
#  
#  with open(tsv_file, 'rb') as tsvfile:
#      file = csv.reader(tsvfile, delimiter='\t')
#      keys = file.next()
#      for key in keys:
#          the_dict[key] = []
#      for row in file:
#          for idx, val in enumerate(row):
#              the_dict[keys[idx]].append(val)
#  
#  #print(the_dict)
#  
#  search = re.compile(r'[^\.\d]')
#  string = ''.join(the_dict['onset'])
#  
#  print(string)
#  newstr = string.replace('.','')
#  newstr.isdigit()
