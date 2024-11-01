<<<<<<< HEAD
#%%
import tensorflow as tf
import joblib

def make_model():
    model = tf.keras.models.Sequential()

    ''' _______________________________________________________________________ Input:'''
    model.add(tf.keras.layers.Conv2D(kernel_size        = 3, 
                                        filters            = 64,
                                        padding            = 'same', 
                                        kernel_initializer = 'he_normal',
                                        input_shape        = (184, 184, 1),
                                        activation         = 'relu'))
    # model.add(tf.keras.layers.Activation('ReLU'))
    model.add(tf.keras.layers.MaxPooling2D((2,2)))
        
    ''' _______________________________________________________________________ Convulational:'''                  
    for n_cov in [[128],[256,256],[512,512],[512,512]]:
        for filters in n_cov:
            model.add(tf.keras.layers.Conv2D(kernel_size        = 3, 
                                                filters            = filters,
                                                padding            = 'same', 
                                                kernel_initializer = 'he_normal',
                                                activation         = 'relu'))
            # model.add(tf.keras.layers.Activation('ReLU'))
        model.add(tf.keras.layers.MaxPooling2D((2,2)))

    model.add(tf.keras.layers.Flatten())

    ''' _______________________________________________________________________ Deep Layers:'''
    for units in [512,512]:
        model.add(tf.keras.layers.Dense(units              = units,
                                        kernel_initializer = 'he_normal',
                                        activation         = 'relu'))
        # model.add(tf.keras.layers.Activation('ReLU'))
        
    ''' _______________________________________________________________________ Output:'''

    model.add(tf.keras.layers.Dense(units=1))
    model.add(tf.keras.layers.Activation('linear', dtype='float32'))

    weights = joblib.load('.\weights.dat')

    model.set_weights(weights)

    return model
=======
#%%
import tensorflow as tf
import joblib

def make_model():
    model = tf.keras.models.Sequential()

    ''' _______________________________________________________________________ Input:'''
    model.add(tf.keras.layers.Conv2D(kernel_size        = 3, 
                                        filters            = 64,
                                        padding            = 'same', 
                                        kernel_initializer = 'he_normal',
                                        input_shape        = (184, 184, 1),
                                        activation         = 'relu'))
    # model.add(tf.keras.layers.Activation('ReLU'))
    model.add(tf.keras.layers.MaxPooling2D((2,2)))
        
    ''' _______________________________________________________________________ Convulational:'''                  
    for n_cov in [[128],[256,256],[512,512],[512,512]]:
        for filters in n_cov:
            model.add(tf.keras.layers.Conv2D(kernel_size        = 3, 
                                                filters            = filters,
                                                padding            = 'same', 
                                                kernel_initializer = 'he_normal',
                                                activation         = 'relu'))
            # model.add(tf.keras.layers.Activation('ReLU'))
        model.add(tf.keras.layers.MaxPooling2D((2,2)))

    model.add(tf.keras.layers.Flatten())

    ''' _______________________________________________________________________ Deep Layers:'''
    for units in [512,512]:
        model.add(tf.keras.layers.Dense(units              = units,
                                        kernel_initializer = 'he_normal',
                                        activation         = 'relu'))
        # model.add(tf.keras.layers.Activation('ReLU'))
        
    ''' _______________________________________________________________________ Output:'''

    model.add(tf.keras.layers.Dense(units=1))
    model.add(tf.keras.layers.Activation('linear', dtype='float32'))

    weights = joblib.load('.\weights.dat')

    model.set_weights(weights)

    return model
>>>>>>> f138422dcb7ac198875f8ee040550f475b0d7b3a
