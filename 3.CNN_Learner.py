import pandas as pd
import numpy as np
import pickle
import time
import gzip
import matplotlib

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
from keras.utils.training_utils import multi_gpu_model
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"]='0'

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



#K.tensorflow_backend._get_available_gpus() # GPU 개수 확인

with gzip.open('output_data/train_x.pickle', 'rb') as f:
    train_x = pickle.load(f)

with gzip.open('output_data/train_y.pickle', 'rb') as f:
    train_y = pickle.load(f)

with gzip.open('output_data/keyword_dict.pickle', 'rb') as f:
    keyword_dict = pickle.load(f)


dim = train_x.shape[1]

epochs = 100
batch_size = 500
k_n=5

#train_x, train_y, test_x, test_y = train_test_split(train_x, train_y, test_size=0.2, random_state=1, stratify=y)


kfold = StratifiedKFold(n_splits=k_n, shuffle=True, random_state=1)



with K.tf_ops.device('/device:GPU:0'):
    result=[]
    for train, test in kfold.split(train_x, train_y):
        inputs = Input(shape=(train_x.shape[1],), name='input')
        embeddings_out = Embedding(input_dim=len(keyword_dict) , output_dim=128,name='embedding')(inputs)
        #embeddings_out = Embedding(input_dim=len(keyword_dict) , output_dim=64,name='embedding')(inputs)
        #embeddings_out = Embedding(input_dim=len(keyword_dict) , output_dim=32,name='embedding')(inputs)
        #embeddings_out = Embedding(input_dim=len(keyword_dict) , output_dim=16,name='embedding')(inputs)
        #embeddings_out = Embedding(input_dim=len(keyword_dict) , output_dim=8,name='embedding')(inputs)


        conv0 = Conv1D(128, 1, padding='same')(embeddings_out)
        #conv0 = Conv1D(64, 1, padding='same')(embeddings_out)
        #conv0 = Conv1D(32, 1, padding='same')(embeddings_out)
        #conv0 = Conv1D(16, 1, padding='same')(embeddings_out)
        #conv0 = Conv1D(8, 1, padding='same')(embeddings_out)

        pool0 = MaxPooling1D()(conv0)
        
        flat =Flatten()(pool0)

        out = Dense(1,activation='sigmoid')(flat)

        model = Model(inputs=[inputs,], outputs=out)
        model.compile(optimizer='rmsprop', loss='binary_crossentropy',metrics=['accuracy'])

        model_dir = './CNN_output_data'
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        model_path = model_dir + "/CNN.model"
        #checkpoint = ModelCheckpoint(filepath=model_path, monitor="val_acc", mode='max', verbose=1, save_best_only=True)
        early_stopping = EarlyStopping(monitor='val_acc', patience=3)

        #hist = model.fit(x=train_x[train],y=train_y[train], batch_size=batch_size, epochs=epochs, validation_data=(train_x[test],train_y[test]),callbacks=[checkpoint,early_stopping])
        
        hist = model.fit(x=train_x[train],y=train_y[train], batch_size=batch_size, epochs=epochs, validation_data=(train_x[test],train_y[test]),callbacks=[early_stopping])
        eva = model.evaluate(x=train_x[test],y=train_y[test],batch_size=batch_size)
        print("evaluation ; ", eva)
        result.append(eva)

model.save(model_dir+'/CNN_rcs_test_1.model')
model.summary()
#with open(model_dir+'/model_acc5.pickle', 'wb') as f:
#    pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)


#result = cross_val_score (model, train_x,train_y, cv=k_n)
print("cross : ",result)

'''
#hist = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, Y_val))
#val_loss = hist.history['val_loss']
val_acc = hist.history['val_acc']

#font_name = matplotlib.font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
#matplotlib.rc('font', family=font_name)

fig = plt.figure()
ax_acc = fig.add_subplot(111)

ax_acc.plot(range(epochs), val_acc, label='accuracy', color='darkblue')
plt.text(3, 14.7, "<----------------accuracy", verticalalignment='top', horizontalalignment='right')
plt.xlabel('epochs')
plt.ylabel('accuracy')
ax_acc.grid(linestyle='--', color='lavender')

#ax_loss = ax_acc.twinx()
#ax_loss.plot(range(epochs), val_loss, label='loss', color='darkred')
#plt.text(3, 2.2, "<----------------loss", verticalalignment='top', horizontalalignment='left')
#plt.ylabel('loss')
#ax_loss.yaxis.tick_right()
#ax_loss.grid(linestyle='--', color='lavender')

plt.legend(loc='lower left',bbox_to_anchor=(0.0, 0.1))
ax_acc.legend(loc='lower left',bbox_to_anchor=(0.0, 0.05))
#plt.show()
'''








