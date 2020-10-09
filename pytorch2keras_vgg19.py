import numpy as np
import torch
from torch.autograd import Variable
from pytorch2keras.converter import pytorch_to_keras
import torchvision
import tensorflow as tf

class VGG(torchvision.models.vgg.VGG):
    def __init__(self, *args, **kwargs):
        super(VGG, self).__init__(*args, **kwargs)

    def forward(self, x):
        x = self.features(x)
        x = x.view([int(x.size(0)), -1])
        x = self.classifier(x)
        return x


def check_error(output, k_model, input_np, epsilon=1e-5):
    #    pytorch_output = output.data.numpy()
    keras_output = k_model.predict(input_np)

    error = np.max(pytorch_output - keras_output)
    print('Error:', error)

    assert error < epsilon
    return error

# currently, manually rename the layers after using pytorch2keras

def load_my_model(k_model):
    model = tf.keras.Sequential()
    predefined_layer_names = [
        'block0_input',

        'block1_conv1_zeropad',
        'block1_conv1_conv',
        'block1_conv1_relu',
        'block1_conv2_zeropad',
        'block1_conv2_conv',
        'block1_conv2_relu',
        'block1_pool',

        'block2_conv1_zeropad',
        'block2_conv1_conv',
        'block2_conv1_relu',
        'block2_conv2_zeropad',
        'block2_conv2_conv',
        'block2_conv2_relu',
        'block2_pool',
        
        'block3_conv1_zeropad',
        'block3_conv1_conv',
        'block3_conv1_relu',
        'block3_conv2_zeropad',
        'block3_conv2_conv',
        'block3_conv2_relu',
        'block3_conv3_zeropad',
        'block3_conv3_conv',
        'block3_conv3_relu',
        'block3_conv4_zeropad',
        'block3_conv4_conv3',
        'block3_conv4_relu',
        'block3_pool',

        'block4_conv1_zeropad',
        'block4_conv1_conv',
        'block4_conv1_relu',
        'block4_conv2_zeropad',
        'block4_conv2_conv',
        'block4_conv2_relu',
        'block4_conv3_zeropad',
        'block4_conv3_conv',
        'block4_conv3_relu',
        'block4_conv4_zeropad',
        'block4_conv4_conv',
        'block4_conv4_relu',
        'block4_pool',

        'block5_conv1_zeropad',
        'block5_conv1_conv',
        'block5_conv1_relu',
        'block5_conv2_zeropad',
        'block5_conv2_conv',
        'block5_conv2_relu',
        'block5_conv3_zeropad',
        'block5_conv3_conv',
        'block5_conv3_relu',
        'block5_conv4_zeropad',
        'block5_conv4_conv',
        'block5_conv4_relu',
        'block5_pool',

        'block6_pool',
        'block6_reshape',
        'block6_dense1_fc',
        'block6_dense1_relu',
        'block6_dense2_fc',
        'block6_dense2_relu',
        'block6_dense3_fc'
    ]


    for i, layer in enumerate(k_model.layers):
        ln = predefined_layer_names[i]
        ops = ln.split('_')[-1]
        blk = ln.split('_')[0]
        if blk == 'block6': # currently the change_ordering only works for feature layers
            break
        layer._name = ln # rename using ._name

        if ops == 'pool':
            replaced_layer = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', data_format='channels_last', name=ln)
            # replaced_layer = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid', data_format='channels_last', name=ln)
            model.add(replaced_layer)
        else:
            model.add(layer)

    print (model.summary())
    return model


if __name__ == '__main__':
    # max_error = 0

    model = torchvision.models.vgg19(pretrained=True)
    model.eval()
    
    for i in range(1):
        # model = VGG(torchvision.models.vgg.make_layers(torchvision.models.vgg.cfgs['A'], batch_norm=True))
        # model.eval()

        input_np = np.random.uniform(0, 1, (1, 3, 224, 224))
        
        input_var = Variable(torch.FloatTensor(input_np))
        output = model(input_var)

        k_model = pytorch_to_keras(model, input_var, [(3,224,224,)], verbose=True, name_policy='short', change_ordering=True)
        k_model.trainable = False
        print (k_model.summary())

        # k_model.save('vgg.h5')
        # k_model = tf.keras.models.load_model('vgg.h5')
        # print (k_model.summary())
        my_model = load_my_model(k_model)
        my_model.save('pytorch2keras_vgg19.h5')

        input_np = np.random.uniform(0, 1, (1, 224, 224, 3)) # order: channel_last
        my_model_output = my_model.predict(input_np)

        # error = check_error(output, k_model, input_np)
        # if max_error < error:
        #     max_error = error

    # print('Max error: {0}'.format(max_error))
    print('===> Done.')






