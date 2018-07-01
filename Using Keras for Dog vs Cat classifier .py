# first of all let's lay out the schema of the work that we're gonna do !! 
# keras is a high lvl framework that allows us to easily create and test models 
# so in keras you treat the inputs as something called a generator that with only the directory of the imgs 
# you can create appropriate data and labels that would be fed into a model 

# so the steps in order to create a keras model : 
# 1. you import the libraries required

# 2. you specify the constants like the img dimentions, the train/test/val directories, the number of epochs,
# batch size etc etc

# 3. you construct the architecture of the model you're gonna use, how many conv2d layers, the kind of activation,
# etc etc and then you compile this model or compile the still not instantiated computation graph 

# 4. then you create the image generator which is basically a tool to augment the images in real time and to normalize 
# them. PS: (for the test generator we don't augment the imgs we just rescale them, eg normalize them!! )  

# 5. You then use those augemntation configuration and apply them to your imgs in real time as you put them into 
# the appropriate settings of training data and labels (you put the augmented imgs in a numpy array and also the labels)
# you could do that using a method called 'flow_from_directory()'. Use case: 
# the generator.flow_from_directory(the directory of the data, target_size(height,width), batch_size, class_mode ='binary','softmax',etc)

# 6. and u do the same with validation dataset 

# 7. and now you fit those generators and train the model using a method called 'fit_generator'
# Use case : model.fit_generator(train_generator, steps_per_epoch = m1 // batch_size, epochs, validation_data = validation_generator
# validation_steps = m2 // batch_size )  m1/m2 : number of training examples/number of validation examples 

# 8. and last but not least you save the weights those weights that you trained !! using model.save_weights('name.h5')

# import libraries  
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras import applications
import os 
import numpy as np 


# specify constants !! 
Height, Width = 150, 150 
epochs = 10 
batch_size = 16 
m1 = 2000
m2 = 800
cwd = os.getcwd()
print(cwd)
train_dir = os.path.join(cwd,'kerasClassifier/train')
validation_dir = os.path.join(cwd,'kerasClassifier/validation')




# just a standard test case to check whether the inputs are (channels , height, width) or (height, width, channels): 

if K.image_data_format() == 'channels_first':
    input_shape = (3, Height, Width)
else:
    input_shape = (Height, Width, 3)




# now we construct the architecture !! 

model = Sequential()
model.add(Conv2D(32, (3,3), input_shape = input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2))) 

model.add(Conv2D(32, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2))) 

model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2))) 

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss = 'binary_crossentropy',
              optimizer = 'rmsprop',
              metrics = ['accuracy'])



# creating the data generator and augemntations 
train_datagen = ImageDataGenerator(rescale=1. / 255,
                               shear_range=0.2,
                               zoom_range=0.2,
                               horizontal_flip=True)


test_datagen = ImageDataGenerator(rescale = 1. / 255) 



# now we prepare the data and labels for training and applying the augmentations to the imgs in real time !! 
train_gen = train_datagen.flow_from_directory(train_dir, 
                                              target_size = (Height, Width),
                                              batch_size = batch_size,
                                              class_mode = 'binary')

val_gen = test_datagen.flow_from_directory(validation_dir, 
                                            target_size = (Height, Width),
                                            batch_size = batch_size,
                                            class_mode = 'binary')




#training...
model.fit_generator(train_gen,
                    steps_per_epoch = m1 // batch_size,
                    epochs = epochs,
                    validation_data = val_gen,
                    validation_steps = m2 // batch_size)

#saving the weights !! 

model.save_weights('First iteration.h5')

## reference : https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html