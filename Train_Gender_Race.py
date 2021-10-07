# Dataset: UTK Face - https://susanqq.github.io/UTKFace/
# consists of 20k+ face images in the wild (only single face in one image)
# provides the correspondingly aligned and cropped faces
# provides the corresponding landmarks (68 points)
# images are labelled by age, gender, and ethnicity
    # labels:
        # age: integer 0 - 116
        # gender: 0 - male | 1 = female
        # race: 0 - 4, white - black - asian - indian - others
        # date & time

#####
import pandas as pd
pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 10)

#####

Dataset_folder = '/Volumes/New/UTKFace/UTKFace'
Train_test_split = 0.7
Im_Width = Im_Height = 48
dataset_dict = {
    'race_id':{
        0: 'white',
        1: 'black',
        2: 'asian',
        3: 'indian',
        4: 'others'
    },
    'gender_id': {
        0: 'male',
        1: 'female'
    }
}

dataset_dict['gender_alias'] = dict((g,i) for i,g in dataset_dict['gender_id'].items())
dataset_dict['race_alias'] = dict((g,i) for i,g in dataset_dict['race_id'].items())

import numpy as np
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns


# Create function to exract data from the dataset
    # iterate over each file of the UTK dataset and
    # return a Pandas DataFrame containing all the fields (age, gender, and sex)

def parse_dataset(dataset_path, ext = 'jpg'):
    # extract info about the dataset
    def parse_info_from_file(path):
        # Parse info from a single file
        try:
            filename = os.path.split(path)[1]
            filename = os.path.splitext(filename)[0]
            age, gender, race, _ = filename.split('_')
            return int(age), dataset_dict['gender_id'][int(gender)],dataset_dict['race_id'][int(race)]
        except Exception as ex:
            return None, None, None

    files = glob.glob(os.path.join(dataset_path, '*.%s' % ext))
    records = []
    for file in files:
        info = parse_info_from_file(file)
        records.append(info)
    df = pd.DataFrame(records)
    df['file'] = files
    df.columns = ['age','gender','race','file']
    df = df.dropna()
    return df
# The master datasets, including all files
df = parse_dataset(Dataset_folder)

df['gender_id'] = df['gender'].map(lambda gender: dataset_dict['gender_alias'][gender])
df['race_id'] = df['race'].map(lambda race: dataset_dict['race_alias'][race])
df_new = df.sample(frac=1).reset_index(drop=True)

####### Prepare the datasets #######
gender_reco = df_new[['gender_id','file']]
race_reco = df_new[['race_id','file']]
age_reco = df_new[['age','file']]

####### Gender Detection #######
# converting images to arrays and labelling the categories
data = []
labels = []

labels_gender = gender_reco.iloc[:,0].tolist()
img_gender = gender_reco.iloc[:,1].tolist()

import cv2
from tensorflow.keras.preprocessing.image import img_to_array

for img in img_gender:
    image = cv2.imread(img)
    image = cv2.resize(image, (Im_Width,Im_Height))
    image = img_to_array(image)
    data.append(image)

for l in labels_gender:
    labels.append([l])

# pre-processing
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)


# split dataset for training and validation
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical, plot_model
(trainX_gender, testX_gender, trainY_gender, testY_gender) = train_test_split(data, labels, test_size=0.2,
                                                  random_state=42)

(trainX_gender, valX_gender, trainY_gender, valY_gender) = train_test_split(trainX_gender, trainY_gender, test_size=0.2,
                                                  random_state=42)

trainY_gender = to_categorical(trainY_gender, num_classes=2)  # [[1, 0], [0, 1], [0, 1], ...]
testY_gender = to_categorical(testY_gender, num_classes=2)
valY_gender = to_categorical(valY_gender, num_classes=2)

len(trainX_gender) # 15171
len(valX_gender) # 3793
len(testX_gender) # 4741

####### Train the model #######

# augmenting datset
from tensorflow.keras.preprocessing.image import ImageDataGenerator
aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                         horizontal_flip=True, fill_mode="nearest")

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense
from tensorflow.keras import backend as K
import cv2
import os
import glob
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

def build(width, height, depth, classes):
    model = Sequential()
    inputShape = (height, width, depth)
    chanDim = -1

    if K.image_data_format() == "channels_first":  # Returns a string, either 'channels_first' or 'channels_last'
        inputShape = (depth, height, width)
        chanDim = 1

    # The axis that should be normalized, after a Conv2D layer with data_format="channels_first",
    # set axis=1 in BatchNormalization.

    model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))

    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))

    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(classes))
    model.add(Activation("sigmoid"))

    return model

# activate the model
model = build(width=Im_Width, height=Im_Width, depth=3, classes=2)

# fit
from tensorflow.keras.optimizers import Adam
epochs = 100
lr = 1e-3
batch_size = 64

opt = Adam(lr=lr, decay=lr / epochs)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# train the model
H = model.fit_generator(aug.flow(trainX_gender, trainY_gender, batch_size=batch_size),
                        validation_data=(valX_gender, valY_gender),
                        steps_per_epoch=len(trainX_gender) // batch_size,
                        epochs=epochs, verbose=1)
# accuracy epoch 100: 91.31%
model.save('Riddle_gender_detection.model')

# Test on the test dataset

# load model
from tensorflow.keras.models import load_model
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
model = load_model('Riddle_gender_detection.model')

test_batch_size = 128
gender_pred = model.predict_generator(testX_gender, steps=len(testX_gender)//128)
gender_pred_i = gender_pred.argmax(axis=-1)

from sklearn.metrics import classification_report
cr_gender = classification_report(testY_gender, gender_pred_i)
print(cr_gender)
print(cr_gender)
#              precision    recall  f1-score   support
#           0       0.97      0.89      0.93      2479
#           1       0.89      0.97      0.93      2262
#    accuracy                           0.93      4741
#   macro avg       0.93      0.93      0.93      4741
#weighted avg       0.93      0.93      0.93      4741


####### Races Detection #######

# converting images to arrays and labelling the categories
data_races = []
labels_races = []

labels_races_pd = race_reco.iloc[:,0].tolist()
img_races_pd = race_reco.iloc[:,1].tolist()

import cv2
from tensorflow.keras.preprocessing.image import img_to_array

for img in img_races_pd:
    image = cv2.imread(img)
    image = cv2.resize(image, (Im_Width,Im_Height))
    image = img_to_array(image)
    data_races.append(image)

for l in labels_races_pd:
    labels_races.append([l])

# pre-processing
data_races = np.array(data_races, dtype="float") / 255.0
labels_races = np.array(labels_races)

# split dataset for training and validation
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical, plot_model
(trainX_race, testX_race, trainY_race, testY_race) = train_test_split(data_races, labels_races, test_size=0.2,
                                                  random_state=42)

(trainX_race, valX_race, trainY_race, valY_race) = train_test_split(trainX_race, trainY_race, test_size=0.2,
                                                  random_state=42)

trainY_race_ct = to_categorical(trainY_race, num_classes=5)  # [[1, 0, 0, 0, 0], [0, 1 ,0 ,0 ,0], ...]
testY_race_ct = to_categorical(testY_race, num_classes=5)
valY_race_ct = to_categorical(valY_race, num_classes=5)

len(trainX_race) # 15171
len(valX_race) # 3793
len(testY_race) # 4741

### buil classification model ###

model_race = build(width=Im_Width, height=Im_Width, depth=3, classes=5)

# fit
from tensorflow.keras.optimizers import Adam
epochs = 100
lr = 1e-3
batch_size = 64

opt = Adam(lr=lr, decay=lr / epochs)
model_race.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# train the model
H = model_race.fit_generator(aug.flow(trainX_race, trainY_race_ct, batch_size=batch_size),
                        validation_data=(valX_race, valY_race_ct),
                        steps_per_epoch=len(trainX_race) // batch_size,
                        epochs=epochs, verbose=1)
# accuracy epoch 100: 81.03%

model_race.save('Riddle_race_detection.model')

# load model
from tensorflow.keras.models import load_model
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
modeli = load_model('Riddle_race_detection.model')

test_batch_size = 128
race_pred = modeli.predict_generator(testX_race, steps=len(testX_race)//128)
race_pred_i = race_pred.argmax(axis=1)

from sklearn.metrics import classification_report
cr_race = classification_report(testY_race, race_pred_i)
print(cr_race)
# precision    recall  f1-score   support
#           0       0.78      0.94      0.85      1999
#           1       0.84      0.89      0.86       921
#           2       0.90      0.71      0.79       703
#           3       0.79      0.71      0.75       785
#           4       0.43      0.15      0.22       333
#    accuracy                           0.80      4741
#   macro avg       0.75      0.68      0.70      4741
#weighted avg       0.79      0.80      0.79      4741

