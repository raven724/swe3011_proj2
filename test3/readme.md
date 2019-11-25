Main model
=================================================================
Layer (type)                 Output Shape              Param #   
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 32, 32, 32)        896       
conv2d_2 (Conv2D)            (None, 32, 32, 32)        9248      
max_pooling2d_1 (MaxPooling2 (None, 16, 16, 32)        0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 16, 16, 64)        18496     
conv2d_4 (Conv2D)            (None, 16, 16, 64)        36928     
max_pooling2d_2 (MaxPooling2 (None, 8, 8, 64)          0         
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 8, 8, 128)         73856     
conv2d_6 (Conv2D)            (None, 8, 8, 128)         147584    
_________________________________________________________________
conv2d_7 (Conv2D)            (None, 8, 8, 256)         295168    
conv2d_8 (Conv2D)            (None, 8, 8, 256)         590080    
_________________________________________________________________
dense_1 (Dense)              (None, 4096)              67112960  
_________________________________________________________________
dropout_1 (Dropout)          (None, 4096)              0         
_________________________________________________________________
dense_2 (Dense)              (None, 1024)              4195328   
_________________________________________________________________
dense_3 (Dense)              (None, 10)                10250     
_________________________________________________________________

FClight: delete 4096 FC layer 

lldel: delete redundant large conv layer(128, 256) 

winl: large window size(filter size up 3x3->5x5) in first layer of first two block 

nopad: disable padding in first two conv layer 