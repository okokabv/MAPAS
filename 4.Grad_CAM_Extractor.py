import os
import pandas as pd
import numpy as np
import pickle
import time
import gzip

from keras.models import *
from keras.layers import *
from keras.utils import *
from keras.optimizers import *
from keras.callbacks import *
from keras.layers import merge
from keras.layers.core import *
from keras.layers.recurrent import LSTM
from keras.models import *
import keras.backend as K
import tensorflow as tf
 
 
os.environ["CUDA_VISIBLE_DEVICES"]=''
 
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)





folder='data/mal/'
target = os.listdir(folder)

be_folder = 'data/be/'
be_target = os.listdir(be_folder)

with gzip.open('output_data/train_x.pickle', 'rb') as f:
    train_x = pickle.load(f)

with gzip.open('output_data/keyword_rev_dict.pickle', 'rb') as f:
    keyword_rev_dict = pickle.load(f)

with gzip.open('output_data/api_call_word_list.pickle', 'rb') as f:
    all= pickle.load(f)

model = load_model('CNN_output_data/CNN.model')
model.summary()

def grad_cam_conv1D(model, layer_nm, x, sample_weight=1, keras_phase=0):

    layers_wt = model.get_layer(layer_nm).weights
    layers = model.get_layer(layer_nm)

    grads = K.gradients(model.output[:, 0], layers_wt)[0]

    pooled_grads = K.mean(grads, axis=(0, 1))

    get_pooled_grads = K.function([model.input, model.sample_weights[0], K.learning_phase()],
                                  [pooled_grads, layers.output[0]])

    pooled_grads_value, conv_layer_output_value = get_pooled_grads([[x, ], [sample_weight, ], keras_phase])

    for i in range(conv_layer_output_value.shape[-1]):
        conv_layer_output_value[:, i] *= pooled_grads_value[i]

    heatmap = np.mean(conv_layer_output_value, axis=-1)

    return ((heatmap, pooled_grads_value))

all = all.iloc[0:, 0].values
val=[]

print("Grad CAM Extracting.....")

for a in range(0,len(all)):
    val.append([])

for idx in range(len(be_target),len(be_target)+len(target)):
    hm, graded = grad_cam_conv1D(model, 'conv1d_1', x=train_x[idx])
    kww = [keyword_rev_dict[i] for i in train_x[idx]]
    for j in range(0,len(kww)):
        for k in range(0,len(all)):
            if all[k] == kww[j]:
                try:
                    val[k].append(hm[j])
                except ValueError:
                    pass
            else :
                pass
last_val=[0]
print("Grad CAM Extracted")

print("Grad CAM Total.........")


for l in range(1,len(all)):
    temp = 0
    for m in range(0,len(val[l])):
        temp = temp + val[l][m]
    temp = temp/(len(all)-1)
    last_val.append(temp)




hm_tbl = pd.DataFrame({'heat': last_val, 'kw': all})
hm_tbl = hm_tbl.sort_values(by=['heat'], axis=0,ascending=False)
with gzip.open('Grad_CAM_output_data/heatmap.pickle', 'wb') as f:
    pickle.dump(hm_tbl, f)


