import keras
from keras import backend as K
import os


# load_path = ['cifar_keras/cifar_keras.h5', 
#              'mnist_custom/mnist_keras.h5']
        
# save_path = ['cifar_keras/', 
#              'mnist_custom/']

# for p in range(2):
#     model = keras.models.load_model(load_path[p])
#     file_name = load_path[p].split('/')[-1]
#     model_name = file_name.split('.')[0]
#     model_name = model_name + '_w'
#     new_name = '.'.join([model_name, 'h5'])
#     print(new_name)
#     print(save_path[p] + new_name)
#     model.save_weights(save_path[p] + new_name)
    
# model = keras.models.load_model(path)
# model.save_weights('cifar_keras_w.h5')

path = 'cifar_df2/model_32_0.75.h5'
save_path = 'cifar_df2/model_32_0.75_w.h5'

model = keras.models.load_model(path)

model.save_weights(save_path)
