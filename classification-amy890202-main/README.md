
# 311581024_The Simpsons Characters Recognition Challenge II



### 檔案說明
- 311581024_王映絜.ipynb 
    - Task1 
    - 主執行檔，包含訓練測試模型與生成submit file都使用該檔執行
- requirements.txt
    - 需要安裝的套件
    - 使用如下指令安裝
        ```
        pip install -r requirements.txt
        ```
- 311581024.csv
    - submit的預測結果
- matrix.png
    - Task2 
    - confusion matrix
- filter.png
    - Task3 
    - 第一層filters權重


### 作法說明

讀入資料集，進行初步圖片處理，並將訓練資料切出訓練集與驗證集，再搭建cnn機器學習模型，或套用vgg16模型，以訓練集資料訓練模型、使用驗證集資料驗證模型，最後輸出測試集的預測結果。


## 311581024_王映絜.ipynb 程式寫法說明

import 所需函式庫
```
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential 
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Dense
# import package
import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras.optimizers import SGD
from keras.applications.vgg16 import preprocess_input

```

設置path路徑
```
# from google drive import files
# from google.colab import drive
# drive.mount('/content/drive')
train_path = "/home/amy890202/2022ML_env/machine-learning-nycu-2022-classification/theSimpsons-train/train"
test_path = "/home/amy890202/2022ML_env/machine-learning-nycu-2022-classification/theSimpsons-te
```


透過資料夾名稱將資料集讀入成ImageGenerator的格式並分類
```
datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
origin_datasets = datagen.flow_from_directory(train_path,class_mode="categorical",color_mode = "rgb", batch_size=32,target_size = (224,224))
test_datasets = datagen.flow_from_directory(test_path,class_mode="categorical",color_mode = "rgb",shuffle = "false", batch_size=1,target_size = (224,224))
test_datasets.samples
```


查看每位character的數量，分配是否過於不平均，決定是否要進行進一步處理。
```

# 查看每位character的數量
chara_count = {}
for data in origin_datasets.filenames:
    chara = data.split('/')[0]
    if chara in chara_count.keys():
        chara_count[chara] += 1
    else:
        chara_count[chara] = 1
print('average character : %i' %(np.mean(list(chara_count.values()))))
#把超過average以上的chara區分出來
color_list = []
chara_list = []
for key,value in zip(chara_count.keys(),chara_count.values()):
    chara_list.append(key)
    if value > 1938 :
        color_list.append('r')
    else:
        color_list.append('b')
# draw bar chart
fig, ax = plt.subplots(figsize=(70, 15))
plt.bar(chara_count.keys(),chara_count.values(),color=color_list)
plt.show()
```
![](https://i.imgur.com/hy4RApp.png)

output結果數據分布圖如上，紅色代表高於平均數量，藍色代表低於平均數量，整體數量分配並無過於不平均。


將train的資料集做Data augmentation(翻轉角度等等)，並切分出3成的訓練集資料作為驗證集。
```

# Data augmentation for training
train_datagen = ImageDataGenerator(rescale=1. / 255.0,shear_range=0.2,zoom_range=0.2,width_shift_range = 0.2,height_shift_range = 0.2,fill_mode = 'nearest',horizontal_flip=True,preprocessing_function=preprocess_input,validation_split=0.3)
#val_datagen = ImageDataGenerator(rescale=1./255,validation_split=0.3)

test_datagen = ImageDataGenerator(rescale=1./255,preprocessing_function=preprocess_input)
#valid_datagen = ImageDataGenerator(rescale=1./255,validation_split=0.33)

train_datasets = train_datagen.flow_from_directory(train_path,class_mode="categorical",color_mode = "rgb", batch_size=32,target_size = (224,224),classes=chara_list,subset='training')
valid_datasets = train_datagen.flow_from_directory(train_path,class_mode="categorical",color_mode = "rgb", batch_size=32,target_size = (224,224),classes=chara_list,subset='valid
```

import建構模型用的套件

```
# import 套件
from keras.models import Sequential 
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Activation
from keras.callbacks import EarlyStopping ,ModelCheckpoint, ReduceLROnPlateau
```

測試gpu是否可用，並調整一些gpu的參數設置。
```
import tensorflow as tf
#tf.__version__
#model = Sequential() #建立一個model
tf.test.is_gpu_available()
gpu_options = tf.GPUOptions(allow_growth=True)
session = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.per_process_gpu_memory_fraction = 0.40
config.gpu_options.allow_growth=True 
```


設定early stop條件
設置ModelCheckpoint(每輪epoch結束後與之前的模型比較儲存當前val_acc最好的模型)
設定ReduceLROnPlateau，即learnig rate的降低條件(0.001 → 0.005 → 0.0025 → 0.00125 → 下限：0.0001)，避免overfitting。

```
#設定early stop
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=20, verbose=1, mode='auto')
# 設定模型儲存條件(儲存最佳模型)
# 設定earlystop條件
#early = EarlyStopping(monitor='val_loss', patience=10,mode='min', verbose=1)
checkpoint = ModelCheckpoint('cnn_checkpoint.h5', verbose=1,
                              monitor='val_acc', save_best_only=True,
                              mode='auto')
# 設定lr降低條件(0.001 → 0.005 → 0.0025 → 0.00125 → 下限：0.0001) auto val_acc
reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5,
                           patience=5, mode='auto', verbose=1,
                           min_lr=1e-4)
```

show_train_history函式會回傳訓練過程的訓練集acc與loss變化圖。
```

import matplotlib.pyplot as plt
def show_train_history(train_history, train):
    plt.plot(train_history.history[train])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train'], loc='center right')
    plt.show()
```

引入VGG16 model並調整他的輸入與輸出成我們資料集要的格式(input layer與output layer)。
```
from keras.models import Sequential
model = Sequential()
model.add(VGG16(weights = 'imagenet',include_top = False,input_shape = (224,224,3)))
model.add(Flatten())
model.add(Dense(units = 4096,activation = 'relu'))
model.add(Dense(units = 4096,activation = 'relu'))
model.add(Dense(units = len(chara_list),activation = 'softmax'))
```

看整個模型組成。
```
model.summary()

```
![](https://i.imgur.com/gHdF4vc.png)




實際上的VGG16模型構成如下
```
from keras.models import Sequential
from keras.layers import MaxPool2D
model = Sequential()
# two conv64 with a pooling layer
model.add(Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
#two conv128 with a pooling layer
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
#conv256 with a pooling layer
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
#four conv512 with a pooling layer
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
#four conv512 with a pooling layer
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
# Two Dense layers with 4096 nodes and an output layer
model.add(Flatten())
model.add(Dense(units=4096,activation="relu"))
model.add(Dense(units=4096,activation="relu"))
model.add(Dense(units = len(chara_list),activation = 'softmax'))
#model.add(Dense(units=50, activation="softmax")) 
```
```
model.summary()

```
![](https://i.imgur.com/hPklPS2.png)


編譯vgg16模型準備訓練。
```
model.compile(loss = "categorical_crossentropy",optimizer = SGD(lr = 0.001),metrics=['accuracy'])
for layer in model.layers:
    layer.trainable = True
```
使用fit_genrator訓練與驗證模型
```
VVGG16_history1 = model.fit_generator(train_datasets,epochs=10, verbose=1,steps_per_epoch=train_datasets.samples//32,validation_data=valid_datasets,validation_steps=valid_datasets.samples//32,use_multiprocessing = True,workers = 16 ,callbacks=[checkpoint, early, reduce_lr])
```
訓練過程圖:
![](https://i.imgur.com/fuopzt9.png)




訓練過程需要很多時間，有時程式會當掉，重開會遺失之前所有儲存的變數並需要重新train，這種情況的話就可以用load_model的方式把先前所儲存的當前訓練到最好的模型取出來繼續train。
```

from keras.models import load_model
#model = load_model("cnn_checkpoint_vgllll.h5")
```
Freeze core VGG16 layers(最後的4層)準備再train一次。 
```
Freeze core VGG16 layers and train again 
for layer in model.layers:
    layer.trainable = False

for layer in model.layers[-4:]:
    layer.trainable = True

model.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr = 0.001), 
              metrics=['accuracy'])
```

再train一次。
```
VGG16_history2 = model.fit_generator(train_datasets,
                   epochs=20, verbose=1,
                   steps_per_epoch=train_datasets.samples//32,
                   validation_data=valid_datasets,
                   validation_steps=valid_datasets.samples//32,
                   callbacks=[checkpoint, early, reduce_lr],use_multiprocessing = True,work
```


畫圖畫出訓練過程的訓練集acc與loss變化圖
```
show_train_history(VGG16_history2, 'acc')
show_train_history(VGG16_history2, 'loss')

```
結果如下:
![](https://i.imgur.com/OR1Id35.png)


評估訓練好的模型對驗證集的最後預測結果。
```
scores = model.evaluate_generator(valid_datasets,use_multiprocessing=True,workers=10,verbose =0,steps=valid_datasets.samples//32)
print("loss =",scores[0],",acc = ",scores[1])
```
對驗證集結果:
loss = 0.20696830116355963 ,acc =  0.9461384911894273



把測試集資料一一讀入並處理後，將使用訓練好的模型(model)預測出的結果存在list中。
```
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
# #from keras.applications.vgg16 import preprocess_input
saveDir = test_path
# #load the image
csv_ans =[]
for i in range(test_datasets.samples):
    img = cv2.imread(saveDir+'/test/'+str(i+1)+'.jpg')
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = preprocess_input(img)
    img = cv2.resize(img,(224,224))
    img = img/255.0
    a = np.reshape(img,(1,224,224,3))
    #make the prediction
    prediction = model.predict(a)
    #(prediction)
    prediction = prediction[0].tolist()
    tmp = max(prediction)
    ans = chara_list[prediction.index(tmp)]
    print(ans)
    csv_ans.append(ans)
```

將該list寫入311581024.csv檔，作為submit檔。
```

import csv
print(test_datasets.samples)
print(len(csv_ans))
test_id = list(range(1,test_datasets.samples+1))
with open('311581024.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['id', 'character'])
    for x,y in zip (test_id,csv_ans):
        writer.writerow([x,y])
```



使用驗證集資料預測結果畫出Confusion Matrix


```
#Plot the confusion matrix. Set Normalize = True/False
import matplotlib.pyplot as plt
def plot_confusion_matrix(cm, classes, normalize=True, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure(figsize=(20,20))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.around(cm, decimals=2)
        cm[np.isnan(cm)] = 0.0
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
            horizontalalignment="center",color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
#Print the Target names
from sklearn.metrics import classification_report, confusion_matrix
import itertools 
#shuffle=False
target_names = []
for key in train_datasets.class_indices:
    target_names.append(key)
# print(target_names)
#Confution Matrix
testing_generator = train_datagen.flow_from_directory(train_path,class_mode="categorical",shuffle=False,color_mode = "rgb", batch_size=1,target_size = (224,224),classes=chara_list,subset='validation')


Y_pred = model.predict_generator(testing_generator,steps = testing_generator.samples//1)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
cm = confusion_matrix(testing_generator.classes, y_pred)
plot_confusion_matrix(cm, target_names, title='Confusion Matrix')
#Print Classification Report
print('Classification Report')
print(classification_report(testing_generator.classes, y_pred, target_names=target_names))
```
![](https://i.imgur.com/hXnQLBR.png)
Confusion_Matrix如下:
![](https://i.imgur.com/rsZ4XAa.png)

看model中的filter
```
# summarize filter shapes
for layer in model.layers: 
    # check for convolutional layer
    print(layer)
    if 'conv' not in layer.name:
        continue   
    # get filter weights
    filters, biases = layer.get_weights()
    print(layer.name, filters.shape)
# normalize filter values to 0-1 so we can visualize them
```
![](https://i.imgur.com/w8lg55S.png)


取出model第一層的filters
```
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 200

# retrieve weights from the second hidden layer  1->0
filters, biases = model.layers[0].get_weights()  

# normalize filter values to 0-1 so we can visualize them
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)

# plot first few filters
n_filters, ix = 6, 1
for i in range(n_filters):
    # get the filter
    f = filters[:, :, :, i]
    
    # plot each channel separately
    for j in range(3):
        # specify subplot and turn of axis
        ax = plt.subplot(n_filters, 3, ix)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # plot filter channel in grayscale
        plt.imshow(f[:, :, j], cmap='gray') # coolwarm
        ix += 1
        
# show the figure
plt.show()
```
畫出的第一層的filter如下圖:
![](https://i.imgur.com/XlPgmc6.png)



### 結果分析

我有嘗試用自己搭建的cnn模型訓練過，最後對測試集的準確率大約83%，套用vgg16模型後準確率有增加到96%，參考並套用pretrain模型確實會對圖片分類有所幫助。
我自己的cnn模型設計如下
```

model = Sequential()
model.add(Conv2D(32, (5, 5), input_shape=(224, 224, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Activation('softmax'))
```
![](https://i.imgur.com/pg5MeTp.png)
上圖為我自己的cnn模型訓練過程圖。
我自己的cnn模型訓練過程剛開始驗證集資料的準確率高於訓練集的準確率，可能是因為訓練集的資料做了data argumentation變得更加複雜，之後訓練集準確率就都普遍高於驗證集的準確率，開始有over-fitting的現象產生。


套用vgg16的訓練過程中每個epoch的訓練集準確率都是普遍高於驗證集的準確率，可以看出稍有些overfitting的狀況 。因此可以透過適度減少learning rate，或使用kfold的方式來降低over-fitting對模型的影響，但kfold花的訓練時間會較長故在這次作業中沒有嘗試。

