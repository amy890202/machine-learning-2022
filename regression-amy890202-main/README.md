
# 311581024_House Price Prediction



### 檔案說明
- 311581024_王映絜.ipynb 
    - 主執行檔，包含訓練測試模型與生成submit file都使用該檔執行)
- requirements.txt
    - 需要安裝的套件
    - 使用如下指令安裝
        ```
        pip install -r requirements.txt
        ```
- train-v3
    - 訓練資料集 
- test-v3
    - 測試資料集
- valid-v3
    - 驗證資料集
- 311581024.zip
    - submit的預測結果   

### 作法說明

讀入資料集，根據關係係數進行初步資料分析，並進行資料前處理與正規化，再使用keras搭建機器學習模型，並以訓練集資料訓練模型、與使用驗證集資料驗證模型，最後輸出測試集的預測結果。


## house_price.py 程式寫法說明

import 所需函式庫
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

```

將資料集讀入成dataframe格式
```
train = pd.read_csv("train-v3.csv", encoding = "ISO-8859-1", engine='python')
test = pd.read_csv("test-v3.csv", encoding = "ISO-8859-1", engine='python')
valid =  pd.read_csv("valid-v3.csv", encoding = "ISO-8859-1", engine='python')
```


檢查訓練集是否有空值
```
print(train[train.isnull().T.any()])#檢查是否有空值，並印出有空值
```

output結果為Empty DataFrame，代表該訓練集無空值，無須作填補空值的動作。


查看各欄位相關性
```

print(train.corr(method='kendall'))
```
部分output結果如下

![](https://i.imgur.com/6sBsGY2.png)


我們專注於紅框所圈各欄位與price之間的相關係數，若為負可考慮直接drop掉或再行資料處理。



去除離群值

```
# 將所有特徵超出1.5倍IQR的概念將這些Outlier先去掉，避免對Model造成影響。
print ("Shape Of The Before Ouliers: ",train.shape)
#print(train['price'])
n=1.5
#IQR = Q3-Q1
IQR = np.percentile(train['price'],75) - np.percentile(train['price'],25)
# outlier = Q3 + n*IQR 
transform_data=train[train['price'] < np.percentile(train['price'],75)+n*IQR]
# outlier = Q1 - n*IQR 
transform_data=transform_data[transform_data['price'] > np.percentile(transform_data['price'],25)-n*IQR]
print ("Shape Of The After Ouliers: ",transform_data.shape)
train = transform_data.reset_index(drop=True) 

# print(train.shape)
# print(test.shape)
# print(valid.shape)
#print(train)
```

Drop掉不必要的欄位(如id)，並分出訓練集的train_x,train_y與validation的valid_x,valid_y (將price欄位設成y)


```
train_y = train.iloc[0:,1:2]
valid_y = valid.iloc[0:,1:2]
train_x = train.drop(columns= ["price","id"])
valid_x = valid.drop(columns= ["price","id"])
test_id = test["id"].values.tolist()
#print(test_id)
test_x = test.drop(columns= ["id"])
```


renovate function會將沒翻新過房子的翻新年份紀錄成建造時間(原先沒翻新過的話該欄為0) 經過該function後yr_renovated欄位與price的相關係數由0.077189增加至0.106372

```
def renovate(df):
    #df = train_x
    count = 0 
    for a in df["yr_renovated"]:
        #print(count)
        if a == 0:
            df["yr_renovated"][count] = df["yr_built"][count]
        count=count+1

    #     if i["yr_renovated"]==0:
    #         i["yr_renovated"] = i["yr_built"]
    df["yr_renovated"][99]
    return df
train_x =  renovate(train_x)
valid_x =  renovate(valid_x)
test_x =  renovate(test_x)
```
zipcode function 會將zipcode的欄位做one hot encoding(因郵遞區號比起數值更偏向類別型資料)

```
def zipcode(df):
    df_dum=pd.get_dummies(df['zipcode'])
    #print(df_dum.shape)
    df.drop(columns= ["zipcode"])
    df_new=pd.concat([df,df_dum],axis=1)
    return df_new

train_x =  zipcode(train_x)
valid_x =  zipcode(valid_x)
test_x =  zipcode(test_x)
```

drop函數將"sale_yr", "sale_month","sale_day","condition","sqft_lot15"五個與price關聯性較低的欄位drop掉。( "sale_month","sale_day"與price關係係數為負，而將sale_yr轉為int後與price關係係數也僅0.006183，過低因此直接drop掉)
```
def drop(df_date):
    df_date = df_date.drop(columns= ["sale_yr", "sale_month","sale_day","condition","sqft_lot15"])  
    return df_date
train_x = drop(train_x)
test_x = drop(test_x)
valid_x = drop(valid_x)
from sklearn import preprocessing
#建立MinMaxScaler物件
minmax = preprocessing.MinMaxScaler()#新資料=（原資料-最小值）/（最大值-最小值）
# 資料標準化
train_x = minmax.fit_transform(train_x)
test_x = minmax.fit_transform(test_x)
valid_x = minmax.fit_transform(valid_x)
```

將資料集做正規化處理，以避免數值本身範圍大小影響模型判斷。
```
from sklearn import preprocessing
#建立MinMaxScaler物件
minmax = preprocessing.MinMaxScaler()#新資料=（原資料-最小值）/（最大值-最小值）
# 資料標準化
train_x = minmax.fit_transform(train_x)
test_x = minmax.fit_transform(test_x)
valid_x = minmax.fit_transform(valid_x)

```
import 所需函式並設置EarlyStopping(EarlyStopping機制避免overfitting)
```
from keras import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Dense
from keras.layers import Dropout
from matplotlib import pyplot as plt
earlystop_callback = EarlyStopping(monitor='val_mae',min_delta=0.0001,patience=1)
```
構建keras模型，避免overfitting所以每層都會dropout部分神經元
```
def build_deep_model():
    model = Sequential()
    model.add(Dense(128, input_shape=(train_x.shape[1],), activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation="relu"))
    model.add(Dropout(0.1))
    model.add(Dense(16, activation="relu"))
    model.add(Dropout(0.1))
    model.add(Dense(16, activation="relu"))
    model.add(Dropout(0.1))
    model.add(Dense(16, activation="relu"))
    #odel.add(Dropout(0.1))
    model.add(Dense(8, activation="relu"))
    #model.add(Dropout(0.1))
    model.add(Dense(8, activation="relu"))
    #model.add(Dropout(0.1))
    model.add(Dense(8, activation="relu"))
    #model.add(Dropout(0.1))
    model.add(Dense(8, activation="relu"))
    #model.add(Dropout(0.1))
    model.add(Dense(1))
    # 編譯模型
    model.compile(loss="mse", optimizer="adam",
    metrics=["mae"])
    return model
```
使用kfold訓練與驗證模型(k設為3，將train_x、train_y切成兩輪的訓練集X_train_p,Y_train_p 搭配驗證集X_val,Y_val)，並將這兩輪資料拿去fit，再印出訓練完的模型對原訓練集train_x、train_y、與原驗證集valid_x, valid_y的MSE與MAE
```
k = 3
nb_val_samples = len(train_x) // k
nb_epochs = 150
mse_scores = []
mae_scores = []
for i in range(k):
    print("Processing Fold #" + str(i))
    # 取出驗證資料集
    X_val = train_x[i*nb_val_samples: (i+1)*nb_val_samples]
    Y_val = train_y[i*nb_val_samples: (i+1)*nb_val_samples]
    # 結合出訓練資料集
    X_train_p = np.concatenate(
    [train_x[:i*nb_val_samples],
    train_x[(i+1)*nb_val_samples:]], axis=0)
    Y_train_p = np.concatenate(
    [train_y[:i*nb_val_samples],
    train_y[(i+1)*nb_val_samples:]], axis=0)
    model = build_deep_model()
    # 訓練模型
    history = model.fit(X_train_p, Y_train_p, epochs=nb_epochs,callbacks=[earlystop_callback],validation_data = (X_val, Y_val),
    validation_freq=1,batch_size=16, verbose=1)
    # 評估模型
    #plot(history)
    mse, mae = model.evaluate(X_val, Y_val)
    mse_scores.append(mse)
    mae_scores.append(mae)
print("MSE_kfold: ", np.mean(mse_scores))
print("MAE_kfold: ", np.mean(mae_scores))

mse, mae = model.evaluate(train_x, train_y)
print("MSE_train: ", mse)
print("MAE_train: ", mae)
# 使用測試資料評估模型
mse, mae = model.evaluate(valid_x, valid_y)
print("MSE_valid: ", mse)
print("MAE_valid: ", mae)
```
MAE output result:
![](https://i.imgur.com/bzgJgfO.png)


train資料集MAE :94981
valid 資料集MAE :98510

畫出訓練過程的mae變化圖
```
def plot(history):
    #print(history.history.keys())
    plt.plot(history.history['mae'])
    plt.plot(history.history['val_mae'])
    plt.title('model mae')
    plt.ylabel('mae')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
plot(history)
```
訓練過程mae變化圖如下
![](https://i.imgur.com/poqvDL0.png)

MAE function 計算真實的y與用模型預測出來的y的差異MAE值
使用訓練好的模型預測出驗證集與訓練集的結果後，可使用該function來計算出與真實結果的MAE值
```
def MAE(real, predict):
    print('MAE',np.mean(abs(real[:len(predict)]  - predict)))
y_pred_v = model.predict(valid_x)
y_pred = model.predict(test_x)
y_pred_train = model.predict(train_x)

MAE(valid_y,y_pred_v)
MAE(train_y,y_pred_train)
```

畫圖畫出train的實際答案與用model predict出的答案的散佈圖(x軸為id y軸為price)以觀察差異
```
for i in range(len(train)):
    train['id'][i] = i
train_ans = []
for i in y_pred_train:
    train_ans.append(int(round(i[0])))
#print(train_ans)
#print(train['price'])
miny = min(np.concatenate([train_ans,train['price']]))
maxy = max(np.concatenate([train_ans,train['price']]))
print(miny ,maxy)
plt.axis([min(train['id']), max(train['id']), miny, maxy])
#plt.xlim(min(train['id']), max(train['id']))
#plt.ylim(miny, maxy)
tick_arr = np.arange(miny, maxy,1000000) #產生刻度陣列(npArray 類似list)
plt.yticks(tick_arr)
plt.scatter(train['id'], train['price'], label='Real')
plt.scatter(train['id'], train_ans, label='Predict')
plt.xlabel('id')
plt.ylabel('Price')
plt.legend()
```
train資料集結果如下:
![](https://i.imgur.com/Wc1o1j8.png)

畫圖畫出valid的實際答案與用model predict出的答案的散佈圖(x軸為id y軸為price)以觀察差異
```
for i in range(len(valid)):
    valid['id'][i] = i
valid_ans = []
for i in y_pred_v:
    valid_ans.append(int(round(i[0])))
miny = min(np.concatenate([valid_ans,valid['price']]))
maxy = max(np.concatenate([valid_ans,valid['price']]))
#print(miny ,maxy)
plt.axis([min(valid['id']), max(valid['id']), miny, maxy])
tick_arr = np.arange(miny, maxy,1000000) #產生刻度陣列(npArray 類似list)
plt.yticks(tick_arr)
plt.scatter(valid['id'], valid['price'], label='Real')
plt.scatter(valid['id'], valid_ans, label='Predict')
plt.xlabel('id')
plt.ylabel('Price')
plt.legend()

```
valid資料集結果如下:
![](https://i.imgur.com/sPoaHSo.png)




將測試集(test_x)使用訓練好的模型(model)預測出的結果(y_pred)存在list中，並將該list寫入311581024.csv檔，作為submit檔。
```
submit_ans = []
for i in y_pred:
    submit_ans.append(int(round(i[0])))
submit_ans
#MAE(valid_y,submit_ans)
import csv

# co_id_np = np.array(co_id)
# year_np = np.array(year)

# data = np.array([co_id_np, year_np])

# np.savetxt("sample.csv", data.T, fmt='%s', delimiter='\t')
    
    
#with open('submit.csv', 'w', newline='',encoding='UTF-8-sig') as test_file:
with open('311581024.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['id', 'price'])
    for x,y in zip (test_id,submit_ans):
        writer.writerow([x,y])
```
### 結果分析
valid資料集結果如下:
![](https://i.imgur.com/sPoaHSo.png)

train資料集結果如下:
![](https://i.imgur.com/Wc1o1j8.png)



透過以下程式碼我們可以觀察原始資料的資料分布
```
# skewness 與 kurtosis
transform_data = train['price']
# skewness 與 kurtosis
skewness = round(transform_data.skew(), 2)
kurtosis = round(transform_data.kurt(), 2)
print(f"偏度(Skewness): {skewness}, 峰度(Kurtosis): {kurtosis}")

# 繪製分布圖
sns.histplot(transform_data, kde=True)
plt.show()# 繪製分布圖
```
結果如下
![](https://i.imgur.com/rdHDiqN.png)

由train跟valid的資料集預測結果散佈圖可發現模型對於真實中特別高房價的預測誤差特別大，無法成功預測高房價，整體的預測價錢偏低。由原始資料分布圖可以發現我們資料集中大部分的資料皆集中於低房價，因此可以推測模型為了降低整體MAE會傾向直接預測低房價。或許下次在資料前處理時若是能預先調整訓練集使其在各價位的的資料分布數量更平均而不是低價數量的偏多，例如預先進行平方根或立方根轉換，能讓模型有較佳的預測表現而不是傾向於普遍預測低房價以降低MAE。




訓練過程mae變化圖如下
![](https://i.imgur.com/poqvDL0.png)



而訓練過程的詳細模型訓練與驗證MAE數據如下
![](https://i.imgur.com/xvsDTVG.png)



可以看出在kflod的訓練過程每個epoch kfold訓練集的MAE普遍高於 kfold驗證集的MAE，這可能是因為kfold的驗證集比訓練集更容易或kfold的訓練過程有over-regularizing 。因此我有嘗試透過減少正則化約束，包括增加模型容量（即通過更多參數使其更深），減少dropout的方式調整過模型參數。

但最後用真實的valid資料集測試預測結果的MAE(MAE_valid=98510)又稍略高於train資料集的MAE(MAE_train = 94981)，代表在kfold後的實際最後預測結果是有些微overfitting情況的。
