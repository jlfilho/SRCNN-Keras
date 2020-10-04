import os
import sys
#stderr = sys.stderr
#sys.stderr = open(os.devnull, 'w')
sys.path.append('libs/')  
import gc
import numpy as np
import matplotlib.pyplot as plt
# Import backend without the "Using X Backend" message
from argparse import ArgumentParser
from PIL import Image
from libs.srcnn import SRCNN
from libs.util import plot_test_images, DataLoader
from keras import backend as K


# Sample call
"""
# Train 2X SRCNN
python3 train.py --train ../../data/train_large/ --validation ../data/val_large/ --test ../data/benchmarks/Set5/  --log_test_path ./test/ --scale 2 --stage all

# Train the 4X SRCNN
python3 train.py --train ../../data/train_large/ --validation ../data/val_large/ --test ../data/benchmarks/Set5/  --log_test_path ./test/ --scale 4 --scaleFrom 2 --stage all

# Train the 8X SRCNN
python3 train.py --train ../../data/train_large/ --validation ../data/val_large/ --test ../data/benchmarks/Set5/  --log_test_path ./test/ --scale 8 --scaleFrom 4 --stage all
"""

def parse_args():
    parser = ArgumentParser(description='Training script for SRCNN')

    parser.add_argument(
        '-stage', '--stage',
        type=str, default='default',
        help='Which stage of training to run',
        choices=['all', 'default', 'finetune']
    )

    parser.add_argument(
        '-epochs', '--epochs',
        type=int, default=1000,
        help='Number epochs per train'
    )

    parser.add_argument(
        '-train', '--train',
        type=str, default='../../data/train_large/',
        help='Folder with training images'
    )

    parser.add_argument(
        '-steps_per_epoch', '--steps_per_epoch',
        type=int, default=625,
        help='Steps per epoch'
    )

    parser.add_argument(
        '-validation', '--validation',
        type=str, default='../data/val_large/',
        help='Folder with validation images'
    )

    parser.add_argument(
        '-steps_per_validation', '--steps_per_validation',
        type=int, default=10,
        help='Steps per validation'
    )
    
    parser.add_argument(
        '-test', '--test',
        type=str, default='../data/benchmarks/Set5/',
        help='Folder with testing images'
    )

    parser.add_argument(
        '-print_frequency', '--print_frequency',
        type=int, default=10,
        help='Frequency of print test images'
    )
        
    parser.add_argument(
        '-modelname', '--modelname',
        type=str, default='SRCNN_v1',
        help='SRCNN'
    )
        
    parser.add_argument(
        '-scale', '--scale',
        type=int, default=2,
        help='How much should we upscale images'
    )

    parser.add_argument(
        '-scaleFrom', '--scaleFrom',
        type=int, default=None,
        help='Perform transfer learning from lower-upscale model'
    )
        
    parser.add_argument(
        '-workers', '--workers',
        type=int, default=4,
        help='How many workers to user for pre-processing'
    )

    parser.add_argument(
        '-max_queue_size', '--max_queue_size',
        type=int, default=5,
        help='Max queue size to workers'
    )
        
    parser.add_argument(
        '-batch_size', '--batch_size',
        type=int, default=128,
        help='What batch-size should we use'
    )

    parser.add_argument(
        '-crops_per_image', '--crops_per_image',
        type=int, default=4,
        help='Increase in order to reduce random reads on disk (in case of slower SDDs or HDDs)'
    )           
        
    parser.add_argument(
        '-weight_path', '--weight_path',
        type=str, default='./model/',
        help='Where to output weights during training'
    )

    parser.add_argument(
        '-log_tensorboard_update_freq', '--log_tensorboard_update_freq',
        type=int, default=10,
        help='Frequency of update tensorboard weight'
    )
        
    parser.add_argument(
        '-log_path', '--log_path',
        type=str, default='./logs/',
        help='Where to output tensorboard logs during training'
    )

    parser.add_argument(
        '-log_test_path', '--log_test_path',
        type=str, default='./test/',
        help='Path to generate images in train'
    )


    parser.add_argument(
        '-height_lr', '--height_lr',
        type=int, default=16,
        help='height of lr crop'
    )

    parser.add_argument(
        '-width_lr', '--width_lr',
        type=int, default=16,
        help='width of lr crop'
    )

    parser.add_argument(
        '-channels', '--channels',
        type=int, default=3,
        help='channels of images'
    )

    parser.add_argument(
        '-colorspace', '--colorspace',
        type=str, default='RGB',
        help='Colorspace of images, e.g., RGB or YYCbCr'
    )

    parser.add_argument(
        '-media_type', '--media_type',
        type=str, default='i',
        help='Type of media i to image or v to video'
    )
        
    return  parser.parse_args()

def reset_layer_names(args):
    '''In case of transfer learning, it's important that the names of the weights match
    between the different networks (e.g. 2X and 4X). This function loads the lower-lever
    SR network from a reset keras session (thus forcing names to start from naming index 0),
    loads the weights onto that network, and saves the weights again with proper names'''

    # Find lower-upscaling model results
    BASE = os.path.join(args.weight_path, args.modelname+'_'+str(args.scaleFrom)+'X.h5')
    assert os.path.isfile(BASE), 'Could not find '+BASE

    
    # Load previous model with weights, and re-save weights so that name ordering will match new model
    prev_model = SRCNN(upscaling_factor=args.scaleFrom)
    prev_model.load_weights(BASE)
    prev_model.save_weights(args.weight_path+args.modelname)

    del prev_model
    K.reset_uids()
    gc.collect()
    return BASE

def model_freeze_layers(args, srcnn):
    '''In case of transfer learning, this function freezes lower-level generator
    layers according to the scaleFrom argument, and recompiles the model so that
    only the top layer is trained'''

    trainable=False
    for layer in srcnn.model.layers:
        if layer.name == 'conv3':
            trainable = True 
        layer.trainable = trainable

    # Compile generator with frozen layers
    srcnn.compile_model(srcnn.model)

def model_train(srcnn, args, epochs):
    '''Just a convenience function for training the SRCNN'''
    srcnn.train(
        epochs=epochs, 
        **args
    )



# Run script
if __name__ == '__main__':

    # Parse command-line arguments
    args = parse_args()
       
    # Common settings for all training stages
    args_train = {
        "model_name": args.modelname, 
        "batch_size": args.batch_size, 
        "steps_per_epoch": args.steps_per_epoch,
        "steps_per_validation": args.steps_per_validation,
        "crops_per_image": args.crops_per_image,
        "print_frequency": args.print_frequency,
        "log_tensorboard_update_freq": args.log_tensorboard_update_freq,
        "workers": args.workers,
        "max_queue_size": args.max_queue_size,
        "datapath_train": args.train,
        "datapath_validation": args.validation,
        "datapath_test": args.test,
        "log_weight_path": args.weight_path, 
        "log_tensorboard_path": args.log_path,        
        "log_test_path": args.log_test_path,        
        "media_type": args.media_type
    }

    args_model = {
        "height_lr": args.height_lr, 
        "width_lr": args.width_lr, 
        "channels": args.channels,
        "upscaling_factor": args.scale, 
        "colorspace": args.colorspace        
    }

    # Generator weight paths
    srcnn_path = os.path.join(args.weight_path, args.modelname+'_'+str(args.scale)+'X.h5')
    

    ## FIRST STAGE: TRAINING GENERATOR ONLY WITH MSE LOSS
    ######################################################

    # If we are doing transfer learning, only train top layer of the generator
    # And load weights from lower-upscaling model    
    if args.stage in ['all', 'default']:
        if args.scaleFrom:
            print(">> TRAIN DEFAULT MODEL SRCNN: scale {}X with transfer learning from {}X".format(args.scale,args.scaleFrom))

            # Ensure proper layer names
            BASE = reset_layer_names(args)

            # Load the properly named weights onto this model and freeze lower-level layers
            srcnn = SRCNN(lr=1e-4,**args_model)
            srcnn.load_weights(BASE, by_name=True)
            model_freeze_layers(args, srcnn)
            model_train(srcnn, args_train, epochs=3)

            # Train entire generator for 3 epochs
            srcnn = SRCNN(lr=1e-4,**args_model)
            srcnn.load_weights(srcnn_path)
            model_train(srcnn, args_train, epochs = 3)
        
        else:
            print(">> TRAIN DEFAULT MODEL SRCNN: scale {}X".format(args.scale))
            # As in paper - train for 10^-4 epochs
            srcnn = SRCNN(lr=1e-4,**args_model) 
            model_train(srcnn, args_train, epochs=args.epochs)
               
    ## SECOND STAGE: FINE TUNE SRCNN WITH LOW LEARNING RATE
    ######################################################    
    # Re-initialize & fine-tune GAN - load generator & discriminator weights
    if args.stage in ['all', 'finetune']:
        srcnn = SRCNN(lr=1e-4,**args_model)
        srcnn.load_weights(srcnn_path)
        print("FINE TUNE SRCNN WITH LOW LEARNING RATE")
        model_train(srcnn, args_train, epochs=args.epochs)
        
