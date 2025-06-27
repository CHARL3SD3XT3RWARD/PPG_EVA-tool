import os
import ast
import PPG_EVA_tool as eva
import numpy as np
import pandas as pd
from collections import defaultdict


master_ROC_values = defaultdict(dict) 
train_set_SQI = defaultdict(dict)

path = r'' # path to training_signals


signal_names = os.listdir(path)
df_data_anno = pd.read_excel('')#path to annotationfile

df_data_anno = df_data_anno.set_index('Name', drop=True)#deleting index-column cuz pandas ist fucked up

data_anno = defaultdict(dict, df_data_anno.to_dict(orient='index'))


#just some god damn string editing. You'll figure it out.
#The only importance is, taht you have a dict with the Names as keys and the annotation as array/list
#keep in mind that it have to match the signalchunks... so no shuffleing!
for name in data_anno: 
    string = data_anno[name]['annotation']
    string_with_commas = "[" + ",".join(string.strip("[]").split()) + "]"
    data_anno[name] = ast.literal_eval(string_with_commas )


#%%
all_data = []

for num, name in enumerate(signal_names):
        
    print(f'Processing {name} {num}/{len(signal_names)}')
    #signal import
    
    anno = data_anno['Signal-Name'] # danta_anno[name]
    
    somno_SQI = eva.preprocessing(path, name, plot=False, training=False) #to edit the meatadata, go to config in PPG_EVA-tool

    #extracting
    
    truth_matrix = np.vstack((somno_SQI.skewness,
                              somno_SQI.kurt,
                              somno_SQI.entropy_values,
                              somno_SQI.SNR,
                              somno_SQI.variance,
                              anno)).T 
    
    for row in truth_matrix:
        all_data.append(row)
    
    break
#exporting
df_all_data = pd.DataFrame(all_data, columns=['Skewness', 'Kurtosis', 'Entropy', 'SNR', 'ZCR', 'Annotation'])

df_all_data.to_excel(r'') #path to export. Keep in mind that it mus be the same as in config -> training_values_path

#%% generatin a example annotation

first_half = np.zeros(95, dtype=int)
second_half = np.ones(95, dtype=int)

example_data = np.concatenate((first_half, second_half))

np.random.shuffle(example_data)

example_dict = {'Name': 'Signal-Name', 'annotation': [example_data]}

df_example_data = pd.DataFrame.from_dict(example_dict)
df_example_data.to_excel('')# path to example_folders -> delete the index-column after annotation -> pandas is sometimes fucked up












