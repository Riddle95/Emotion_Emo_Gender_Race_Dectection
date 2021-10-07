from __future__ import print_function
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten,BatchNormalization
from keras.layers import Conv2D,MaxPooling2D
import os

num_classes = 7
img_rows,img_cols = 48,48
batch_size = 32

train_data_dir = '/Volumes/New/archive/train'
validation_data_dir = '/Volumes/New/archive/test'

train_datagen = ImageDataGenerator(
					rescale=1./255,
					rotation_range=30,
					shear_range=0.3,
					zoom_range=0.3,
					width_shift_range=0.4,
					height_shift_range=0.4,
					horizontal_flip=True,
					fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
					train_data_dir,
					color_mode='grayscale',
					target_size=(img_rows,img_cols),
					batch_size=batch_size,
					class_mode='categorical',
					shuffle=True)

validation_generator = validation_datagen.flow_from_directory(
							validation_data_dir,
							color_mode='grayscale',
							target_size=(img_rows,img_cols),
							batch_size=batch_size,
							class_mode='categorical',
							shuffle=True)

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
model = Sequential()

# Block-1

model.add(Conv2D(32,(3,3),padding='same',kernel_initializer='he_normal',input_shape=(img_rows,img_cols,1)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(32,(3,3),padding='same',kernel_initializer='he_normal',input_shape=(img_rows,img_cols,1)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

# Block-2 

model.add(Conv2D(64,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(64,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

# Block-3

model.add(Conv2D(128,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(128,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

# Block-4 

model.add(Conv2D(256,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(256,(3,3),padding='same',kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

# Block-5

model.add(Flatten())
model.add(Dense(64,kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# Block-6

model.add(Dense(64,kernel_initializer='he_normal'))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# Block-7

model.add(Dense(num_classes,kernel_initializer='he_normal'))
model.add(Activation('softmax'))

print(model.summary())

from keras.optimizers import RMSprop,SGD,Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Save the model
checkpoint = ModelCheckpoint('Riddle_Emo_Reco.h5',
                             monitor='val_loss',
                             mode='min',
                             save_best_only=True,
                             verbose=1)

earlystop = EarlyStopping(monitor='val_loss',
                          min_delta=0,
                          patience=3,
                          verbose=1,
                          restore_best_weights=True
                          )

reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.2,
                              patience=3,
                              verbose=1,
                              min_delta=0.0001)

callbacks = [earlystop,checkpoint,reduce_lr]

model.compile(loss='categorical_crossentropy',
              optimizer = Adam(lr=0.001),
              metrics=['accuracy'])

# Set up the training, testing, and epochs
nb_train_samples = 28709
nb_validation_samples = 7178
epochs=25

history=model.fit_generator(
                train_generator,
                steps_per_epoch=nb_train_samples//batch_size,
                epochs=epochs,
                callbacks=callbacks,
                validation_data=validation_generator,
                validation_steps=nb_validation_samples//batch_size)
# Save the model
# model.save('Riddle_emo_de_Kr.model')




###### Test #######

import os
import cv2
import numpy as np

test_path = '/Volumes/New/NSW Health R&D'

### Ideas:
    # Step 1. Data preparation and pre-processing
        # images:
            # scanning the path (folder containing the images)
            # convert the images data into the N-dimensional array
        # classes:
            # scanning the labelFile (folder containing classes)
            # convert the classes into the N-dimensional array
    # Step 2. Prepare the training, test, and validation datasets
        # prepare the training and testing datasets
            # apply the test ratio
        # prepare the validation dataset
            # apply the validation ratio

# Set the dimension of the images
imageDimesions = (48,48)  #width and height

def pre_processing(path):
    # count: the number of folders in the path
        # there are 3 folders in the path, naming 0, 1, and 2 respectively
            # each folders contains the images classified manually
                # Classes: 3
                    # 0: angry
                    # 1: disgust
                    # 2: fear
                    # 3: happy
                    # 4: neutral
                    # 5: sad
                    # 6: surprise
    count = 0 # base: 0
    # images: containing the N-dimensional array of images
    images = [] # create an empty list
    # classNo: containing the label of classes
    classNo = [] # create an empty list
    # create the for-loop
        # read all images and convert them into array
    for x in range(0,7): # set 7 as the limit as there are 7 classes
        # os.listdir: scan the path and provide the results in a list
            # Example: the name of the image will be record in the string type in the list
                # 'image-0-03109.jpg'
        myPicList = os.listdir(path+"/"+str(count)) # add count to scan through all folders
        for y in myPicList: # myPicList: list of all images in 1 folder
            if y.endswith(".jpg"): # the images are recorded in the jpg format
                curImg = cv2.imread(path+"/"+str(count)+"/"+y) # read the image, converting to the array type
                # resize the images to match with the functions introduced later
                curImg2 = cv2.resize(curImg, imageDimesions, interpolation = cv2.INTER_AREA)
                images.append(curImg2)
                classNo.append(count)
        # print(count, end =" "): checking the number of classes
        # after scanning through all the images in 1 folder, count will be added +1 to scan through the next folder
        count +=1
    return images, classNo
def preprocessing(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)     # CONVERT TO GRAYSCALE
    img = cv2.equalizeHist(img)                    # STANDARDIZE THE LIGHTING IN AN IMAGE
    img = img/255.0                                # TO NORMALIZE VALUES BETWEEN 0 AND 1 INSTEAD OF 0 TO 255
    return img

# Prepare the train, test, and validation datasets
    # Test dataset:
        # images_test: all images in the test folder
        # classNo_test: classified emotion of each images

# Prepare the test dataset

pre_list_test = pre_processing(test_path)
images_test = np.array(list(map(preprocessing,pre_list_test[0]))) # TO IRETATE AND PREPROCESS ALL IMAGES
classNo_test = np.array(pre_list_test[1])

print("Data Shapes")
print("Test",end = "");print(images_test.shape,classNo_test.shape)


# set dtype=float32 to match with the y_variables after transformation
X_test=images_test.reshape(images_test.shape[0],images_test.shape[1],images_test.shape[2],1).astype('float32')
# Converts a class vector (integers) to binary class matrix.
#y_test = to_categorical(classNo_test, num_classes = 7)
from tensorflow.keras.models import load_model
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
model = load_model('Riddle_Emo_Reco.h5')
test_batch_size = 10
emo_pred = model.predict(X_test, batch_size=len(X_test)//test_batch_size)
emo_pred_i = emo_pred.argmax(axis=-1)

from sklearn.metrics import classification_report
emo_ac = classification_report(classNo_test, emo_pred_i)
print(emo_ac) # Avg Accuracy: 64%
#                precision    recall  f1-score   support
#           0       0.00      0.00      0.00         0
#           2       0.43      0.17      0.24        18
#           3       0.73      0.79      0.76        24
#           4       0.56      0.68      0.61        28
#           5       0.64      0.51      0.57        41
#           6       0.82      0.67      0.74        21
#    accuracy                           0.58       132
#   macro avg       0.53      0.47      0.49       132
#weighted avg       0.64      0.58      0.59       132
