# Set up the working environment
import pandas as pd
pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 10)

# Pre-processing
import os
import cv2
import numpy as np

imageDimesions = (48,48)  #width and height

test_path = '/Volumes/New/NSW_HEALTH_R&D/NSW Health R&D'

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

def pre_processing(path):
    # count: the number of folders in the path
        # there are 3 folders in the path, naming 0, 1, and 2 respectively
            # each folders contains the images classified manually
                # Classes: 7
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
print(emo_ac)
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

## Stress or not stress ##
# stress: 2 (Fear) or 5 (Sad)
# not stress: others classes
import numpy as np
Pred_stress = np.where((emo_pred_i == 2) | (emo_pred_i == 5), 0, 1)
True_stress = np.where((classNo_test == 2) | (classNo_test == 5), 0, 1)
pred_acc = classification_report(True_stress, Pred_stress)
print(pred_acc) # stress's detection accuracy: 77%

#               precision    recall  f1-score   support
#           0       0.82      0.56      0.67        59
#           1       0.72      0.90      0.80        73
#    accuracy                           0.75       132
#   macro avg       0.77      0.73      0.73       132
#weighted avg       0.77      0.75      0.74       132

### Generate Bio-metrics ###
### Gender ###

## Convert back:
    # input's shape: (48, 48, 3)
    # feed into the model
    # predict race
    # predict gender

def preprocessing_gender_race(img):
    img = img/255.0
    return img
New_input = pre_processing(test_path)

images_test_new = np.array(list(map(preprocessing_gender_race,pre_list_test[0]))) # TO IRETATE AND PREPROCESS ALL IMAGES

print("Data Shapes")
print("Test",end = "");print(images_test_new.shape)

# race
model_race = load_model('Riddle_race_detection.model')
test_batch_size = 10
race_pred = model_race.predict(images_test_new, batch_size=len(images_test_new)//test_batch_size)
race_pred_i = race_pred.argmax(axis=-1)

# gender
model_gender = load_model('Riddle_gender_detection.model')
gender_pred = model_gender.predict(images_test_new, batch_size=len(images_test_new)//test_batch_size)
gender_pred_i = gender_pred.argmax(axis=-1)

# convert arrays into dataframe
    # gender_pred_i: gender
        # 0: male
        # 1: female
    # race_pred_i: race
        # 0: White
        # 1: Black
        # 2: Asian
        # 3: Indian
        # 4: others
    # True_stress: stress condition
        # 0: stress
        # 1: not stress


gender = pd.DataFrame(gender_pred_i, columns = ['gender'])
race = pd.DataFrame(race_pred_i, columns = ['race'])
Stress = pd.DataFrame(True_stress, columns = ['stress_condition'])
Emotion = pd.DataFrame(classNo_test,columns = ['Emotion'])

df_temp = pd.merge(gender, race, left_index=True, right_index=True)
df_all = pd.merge(df_temp, Stress, left_index=True, right_index=True)
df_all = pd.merge(df_all, Emotion, left_index=True, right_index=True)

np.random.seed(245)
df_all['BMI'] = df_all['stress_condition'].apply(lambda x:np.random.randint(24.5, 30.0) if x ==0 else np.random.randint(18.5, 24.9))
df_all['Age'] = np.random.randint(20, 25, df_all.shape[0])
df_all['Coffee_intake'] = np.random.randint(0, 2, df_all.shape[0])
df_all['Smoking'] = np.random.randint(0, 2, df_all.shape[0])
df_all['Steps'] = df_all['stress_condition'].apply(lambda x:np.random.randint(5000, 8000) if x ==0 else np.random.randint(8000, 10000))
df_all['Blood_Pressure'] = df_all['stress_condition'].apply(lambda x:np.random.randint(90, 159) if x ==0 else np.random.randint(80, 120))
df_all['Heart_Rate'] = df_all['Steps'].apply(lambda x:np.random.randint(114, 133) if x < 8000 else np.random.randint(152, 171))
df_all['Excercise'] = np.where(df_all['Steps'] > 8000, 1, 0)
df_all.to_csv('df_bio_metrics.csv',index=False)