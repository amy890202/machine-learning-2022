
# 311581024_Translation



### 檔案說明
- 311581024_王映絜.ipynb 
    - google colab 程式
    - 主執行檔，包含訓練測試模型與生成submit file都使用該檔執行
- requirements.txt
    - 環境需要安裝的套件
    - 使用如下指令安裝
        ```
        pip install -r requirements.txt
        ```
    - 程式在colab環境運行時需安裝的套件指令已包含於程式碼中
- 311581024.csv
    - submit的預測結果



### 作法說明

讀入資料集，進行初步處理，並將訓練資料整理成訓練集與驗證集兩個檔案，先分別幫中文和台語建立字典，再根據字典將文字進行編碼轉為數字，搭建Transformer機器學習模型，再以編碼玩得訓練集資料訓練模型，最後將模型預測結果再解碼轉回文字即為翻譯結果。最後輸出所有測試集的預測結果。

### 前置處理

因google colab免費版會有閒置一段時間就重啟，而導致丟失所有暫存變數需要重跑整個程式的問題，因此有另外用pyCharm寫了一個Python script控制滑鼠每過一段時間去點擊一次colab視窗的腳本以避免colab閒置。
```
import pyautogui
pyautogui.PAUSE = 1#指令間隔 1 秒
from time import sleep
print('mo')
sleep(3)
print('go')
loc = pyautogui.position()
sleep(1)
print('mo new')
sleep(5)
print('go')
iloc = pyautogui.position()
while(True):
    sleep(300)
    if(loc):
        pyautogui.moveTo(loc)
        pyautogui.mouseDown()
        sleep(0.5)
        pyautogui.mouseUp()
    sleep(300)
    if(iloc):
        pyautogui.moveTo(iloc)
        pyautogui.mouseDown()
        sleep(0.5)
        pyautogui.mouseUp()
```


## 311581024_王映絜.ipynb 程式寫法說明



import所需套件
```
import os
import time
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pprint import pprint
from IPython.display import clear_output

```

安裝並使用tensorflow 2.0
```
!pip install tensorflow-gpu==2.0.0-beta0
clear_output()

import tensorflow as tf
import tensorflow_datasets as tfds
print(tf.__version__)
```
設定資料集路徑與各種資訊存放位置
```
ch_vocab_file = "/content/drive/MyDrive/nycu/2022ML/translate/ml-2022-nycu-translation/ch_vocab_new"
tai_vocab_file = "/content/drive/MyDrive/nycu/2022ML/translate/ml-2022-nycu-translation/tai_vocab_new"
checkpoint_path = "/content/drive/MyDrive/nycu/2022ML/translate/ml-2022-nycu-translation/checkpoints_new"
log_dir = "/content/drive/MyDrive/nycu/2022ML/translate/ml-2022-nycu-translation/logs"
download_dir = "/content/drive/MyDrive/nycu/2022ML/translate/ml-2022-nycu-translation/tensorflow-datasets/downloads"

# if not os.path.exists(output_dir):
#   os.makedirs(output_dir)
```
設定colab檔案於google drive雲端的位置 並下載與import所需函式庫
```
from google.colab import drive
drive.mount('/content/drive')

```

寫中文和台語分別的字串處理函式，中文的話將訓練集資料的空格(斷詞)去掉
```

del_list =[0]
def processString(txt):
  # specialChars = "！#$%^&*()；，。ＥＣＦＡＢ、﹔、,＊《 」" 
  # for specialChar in specialChars:
  #   txt = txt.replace(specialChar,'')
  #print(txt) # A,Quick,brown,fox,jumped,over,the,lazy,dog
  #txt = txt.replace('-', ' ')
  #print(txt) # A Quick brown fox jumped over the lazy dog 
  return txt 
def processString_ch(txt):
  #specialChars = "#$%^&*()（）⿰「」！、⿳；，。 ，,_。',、，。、'」「；！？：《》ＥＣＦＡＢ、﹔、,＊《 」" #-123456789
  specialChars = " "
  for specialChar in specialChars:
    txt = txt.replace(specialChar,'')
  #print(txt) # A,Quick,brown,fox,jumped,over,the,lazy,dog
  #print(txt) # A Quick brown fox jumped over the lazy dog 
  return txt 
```

用正則表達式判斷中文字串中是否含有其他英文或數字，有的話之後要drop掉該筆訓練資料。
```
import re

my_re = re.compile(r'[a-zA-Z0-9_]')

my_str_1 = '我是as誰'
my_str_2 = '我是誰。'
print(bool(re.search(my_re, my_str_1)))
print(bool(re.search(my_re, my_str_2)))
```
讀入中文訓練集資料，並拆出訓練與驗證集
```
import pandas as pd
data_path = "/content/drive/MyDrive/nycu/2022ML/translate/ml-2022-nycu-translation/"
train_y_path = "/content/drive/MyDrive/nycu/2022ML/translate/ml-2022-nycu-translation/train-TL.csv"
train_x_path = "/content/drive/MyDrive/nycu/2022ML/translate/ml-2022-nycu-translation/train-ZH.csv"
test_x_path = "/content/drive/MyDrive/nycu/2022ML/translate/ml-2022-nycu-translation/test-ZH-nospace.csv"
train_path = "/content/drive/MyDrive/nycu/2022ML/translate/ml-2022-nycu-translation/train_translate.csv"
valid_path = "/content/drive/MyDrive/nycu/2022ML/translate/ml-2022-nycu-translation/valid_translate.csv"
# wavs_path = data_path + "/train/new/"
# metadata_path = data_path + "/train-toneless.csv"


# Read metadata file and parse it
train_x = pd.read_csv(train_x_path, sep=",", header=None)#, quoting=2
#metadata_df.head(3)
train_x.columns = ["id", "txt"]
train_x = train_x[["id", "txt"]]
#train_x = train_x.drop(0)
count = 0;
for index,row in train_x.iterrows():
  id = row["id"]
  txt = row["txt"]
  row["txt"] = processString_ch(txt)
  if(bool(re.search(my_re, txt))and count not in del_list):# 
    del_list.append(count)
  count = count+1;
print(del_list)
train_x = train_x.drop(del_list)
# data_df = metadata_df.drop(del_list)
# print(data_df.shape)

train_xx = train_x.iloc[:60000]
#train_xx = train_x
valid_x = train_x.iloc[60001:]
#train_y = train_y.sample(frac=1)#.reset_index(drop=True)
#train_y.head(3)
context_raw = train_xx["txt"].values
valid_x = valid_x["txt"].values
print(len(context_raw))
print(len(valid_x))




```
讀入台語訓練集資料，並拆出訓練與驗證集
```
# Read metadata file and parse it
train_y = pd.read_csv(train_y_path, sep=",", header=None)#, quoting=2
#metadata_df.head(3)

train_y.columns = ["id", "txt"]
train_y = train_y[["id", "txt"]]
#train_y = train_y.drop(0)

# count = 0;
# for index,row in train_y.iterrows():
#   id = row["id"]
#   text = row["text"]
#   if((text.islower()==False) and int(id) not in del_list):
#     del_list.append(count)
#   else:
#     for chara in text:
#       if (chara not in lexicon_list )and chara.isalpha():
#         del_list.append(count)
#   count = count+1;
# print(del_list)

# data_df = metadata_df.drop(del_list)
# print(data_df.shape)
for index,row in train_y.iterrows():
  id = row["id"]
  txt = row["txt"]
  row["txt"] = processString(txt)
  #print(row["txt"])


# for index,row in train_y.iterrows():
#   print(row["txt"])
train_y = train_y.drop(del_list)

#train_yy = train_y.iloc[:60000]
train_yy = train_y
valid_y = train_y.iloc[60001:]
#train_y = train_y.sample(frac=1)#.reset_index(drop=True)
#train_y.head(3)
target_raw = train_yy["txt"].values
valid_y = valid_y["txt"].values
print(len(target_raw))
print(len(valid_y))
```
將中台的訓練資料(train_x,train_y)合併整理到train_translate.csv檔。
```
import csv
train_x = pd.read_csv(train_x_path, sep=",", header=None)
train_y = pd.read_csv(train_x_path, sep=",", header=None)

with open("/content/drive/MyDrive/nycu/2022ML/translate/ml-2022-nycu-translation/train_translate.csv", 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['ch', 'tw'])
    for x,y in zip (context_raw,target_raw):
        writer.writerow([x,y])
```
將中台的驗證資料(valid_x,valid_y)合併整理到valid_translate.csv檔。
```
import csv
train_x = pd.read_csv(train_x_path, sep=",", header=None)
train_y = pd.read_csv(train_x_path, sep=",", header=None)

with open("/content/drive/MyDrive/nycu/2022ML/translate/ml-2022-nycu-translation/valid_translate.csv", 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['ch', 'tw'])
    for x,y in zip (valid_x,valid_y):
        writer.writerow([x,y])
```
最後的lexicon_list包含['a', 'h', 'i', 'n', 'k', 'm', 'g', 'p', 't', 'u', 'b', 'e', 'o', 'j', 'l', 's', ' ', '─', '？', '。']



從csv檔讀入整理好的訓練集dataset
```
train_examples = tf.data.experimental.CsvDataset(
  train_path,
  [
   tf.string,tf.string  # Required field, use dtype or empty tensor
  ],
  header = True,
  select_cols=[0,1]  # Only parse last three columns
)
train_examples
```
從csv檔讀入整理好的驗證集dataset
```
val_examples = tf.data.experimental.CsvDataset(
  valid_path,
  [
   tf.string,tf.string  # Required field, use dtype or empty tensor
  ],
  header = True,
  select_cols=[0,1]  # Only parse last three columns
)
val_examples
```
印幾筆訓練集資料出來看看確定格式正確。
```
for element in train_examples.as_numpy_iterator():
  print(element)
  break

```

紀錄log檔要紀錄的訓練集與驗證集拆分比率
```
train_perc = 20
val_prec = 1
drop_prec = 100 - train_perc - val_prec

```

印幾筆預計的翻譯結果sample看看
```
sample_examples = []
num_samples = 10

for ch_t, zh_t in train_examples.take(num_samples):
  ch = ch_t.numpy().decode("utf-8")
  zh = zh_t.numpy().decode("utf-8")
  
  print(ch)
  print(zh)
  print('-' * 10)
  
  # 之後用來簡單評估模型的訓練情況
  sample_examples.append((ch, zh))
```

幫中文資料建立字典
```
%%time
try:
  subword_encoder_ch = tfds.deprecated.text.SubwordTextEncoder.load_from_file(ch_vocab_file)
  print(f"載入已建立的字典： {ch_vocab_file}")
except:
  print("沒有已建立的字典，從頭建立。")
  subword_encoder_ch = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
      (en.numpy() for en, _ in train_examples), 
      target_vocab_size=2**13) # 有需要可以調整字典大小
  
  # 將字典檔案存下以方便下次 warmstart
  subword_encoder_ch.save_to_file(ch_vocab_file)
  

print(f"字典大小：{subword_encoder_ch.vocab_size}")
print(f"前 10 個 subwords：{subword_encoder_ch.subwords[:10]}")
print()

```


將任意一個中文句子根據我們建立的字典編碼看看結果。

```
sample_string = '佇菜園種有機青菜'
indices = subword_encoder_ch.encode(sample_string)
indices
```
再將編碼結果解碼回來看看是否正確
```
print("{0:10}{1:6}".format("Index", "Subword"))
print("-" * 15)
for idx in indices:
  subword = subword_encoder_ch.decode([idx])
  print('{0:5}{1:6}'.format(idx, ' ' * 5 + subword))
```
驗證編碼解碼結果對應正確無誤。
```
sample_string = '佇菜園種有機青菜'
indices = subword_encoder_ch.encode(sample_string)
decoded_string = subword_encoder_ch.decode(indices)
assert decoded_string == sample_string
pprint((sample_string, decoded_string))
```

幫台語也建立字典集。
```
%%time
try:
  subword_encoder_zh = tfds.deprecated.text.SubwordTextEncoder.load_from_file(tai_vocab_file)
  print(f"載入已建立的字典： {tai_vocab_file}")
except:
  print("沒有已建立的字典，從頭建立。")
  subword_encoder_zh = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
      (zh.numpy() for _, zh in train_examples), 
      target_vocab_size=2**13, # 有需要可以調整字典大小
      max_subword_length=1) # 每一個台語字就是字典裡的一個單位
  
  # 將字典檔案存下以方便下次 warmstart 
  subword_encoder_zh.save_to_file(tai_vocab_file)

print(f"字典大小：{subword_encoder_zh.vocab_size}")
print(f"前 10 個 subwords：{subword_encoder_zh.subwords[:10]}")
print()
```

編碼看看結果。
```
sample_string = sample_examples[0][1]
indices = subword_encoder_zh.encode(sample_string)
print(sample_string)
print(indices)
```


測試一下中台編碼轉換
```
ch = "佇菜園種有機青菜"
zh = "ti7 tshai3-hng5 tsing3 iu2-ki1-tshenn1-tshai3"

# 將文字轉成為 subword indices
ch_indices = subword_encoder_ch.encode(ch)
zh_indices = subword_encoder_zh.encode(zh)

print("[中台原文]（轉換前）")
print(en)
print(zh)
print()
print('-' * 20)
print()
print("[中台序列]（轉換後）")
print(ch_indices)
print(zh_indices)
```

前處理數據
在處理序列數據時在一個序列的前後各加入一個特殊的 token，以標記該序列的開始與完結

    -開始 token、Begin of Sentence、BOS、<start>
    -結束 token、End of Sentence、EOS、<end>
定義一個將被 tf.data.Dataset 使用的 encode 函式，它的輸入是一筆包含 2 個 string Tensors 的例子，輸出則是 2 個包含 BOS / EOS 的索引序列
```
def encode(ch_t, zh_t):
  # 因為字典的索引從 0 開始，
  # 我們可以使用 subword_encoder_ch.vocab_size 這個值作為 BOS 的索引值
  # 用 subword_encoder_ch.vocab_size + 1 作為 EOS 的索引值
  ch_indices = [subword_encoder_ch.vocab_size] + subword_encoder_ch.encode(
      ch_t.numpy()) + [subword_encoder_ch.vocab_size + 1]
  # 同理，不過是使用台語字典的最後一個索引 + 1
  zh_indices = [subword_encoder_zh.vocab_size] + subword_encoder_zh.encode(
      zh_t.numpy()) + [subword_encoder_zh.vocab_size + 1]
  
  return ch_indices, zh_indices
```
從訓練集裡任意取一組中台Tensors 看看函式的實際輸出
```
ch_t, zh_t = next(iter(train_examples))
ch_indices, zh_indices = encode(ch_t, zh_t)
print('中文 BOS 的 index：', subword_encoder_ch.vocab_size)
print('中文 EOS 的 index：', subword_encoder_ch.vocab_size + 1)
print('台語 BOS 的 index：', subword_encoder_zh.vocab_size)
print('台語 EOS 的 index：', subword_encoder_zh.vocab_size + 1)

print('\n輸入為 2 個 Tensors：')
pprint((ch_t, zh_t))
print('-' * 15)
print('輸出為 2 個索引序列：')
pprint((ch_indices, zh_indices))
```
下圖為該函式輸出結果，可以看到中台的序列前後都被各自加上了BOS、EOS
![](https://i.imgur.com/AnaqExJ.png)

由於目前 tf.data.Dataset.map 函式裡頭的計算是在計算圖模式（Graph mode）下執行，所以裡頭的 Tensors 並不會有 Eager Execution 下才有的 numpy 屬性，因此encode函式裡的numpy會抱錯。

因此需要使用 tf.py_function 將我們剛剛定義的 encode 函式包成一個以 eager 模式執行的 TensorFlow Op
```
def tf_encode(ch_t, zh_t):
  # 在 `tf_encode` 函式裡頭的 `ch_t` 與 `zh_t` 都不是 Eager Tensors
  # 要到 `tf.py_funtion` 裡頭才是
  # 另外因為索引都是整數，所以使用 `tf.int64`
  return tf.py_function(encode, [ch_t, zh_t], [tf.int64, tf.int64])

# `tmp_dataset` 為說明用資料集，說明完所有重要的 func，
# 我們會從頭建立一個正式的 `train_dataset`
tmp_dataset = train_examples.map(tf_encode)
ch_indices, zh_indices = next(iter(tmp_dataset))
print(ch_indices)
print(zh_indices)
```


filter_max_length函式過濾掉資料集過長的句子。
```
MAX_LENGTH = 40

def filter_max_length(ch, zh, max_length=MAX_LENGTH):
  # ch, zh 分別代表中文和台語的索引序列
  return tf.logical_and(tf.size(ch) <= max_length,
                        tf.size(zh) <= max_length)

# tf.data.Dataset.filter(func) 只會回傳 func 為真的例子
tmp_dataset = tmp_dataset.filter(filter_max_length)
```


使用padded_batch 函式幫每個 batch 裡頭的序列都補 0 到跟當下 batch 裡頭最長的序列一樣長。
```
BATCH_SIZE = 64
# 將 batch 裡的所有序列都 pad 到同樣長度
tmp_dataset = tmp_dataset.padded_batch(BATCH_SIZE, padded_shapes=([-1], [-1]))
ch_batch, zh_batch = next(iter(tmp_dataset))
print("中文索引序列的 batch")
print(ch_batch)
print('-' * 20)
print("台語索引序列的 batch")
print(zh_batch)
```






根據我們前面做的前處理重新建立訓練集與驗證集
```

MAX_LENGTH = 40
BATCH_SIZE = 128
BUFFER_SIZE = 15000

# 訓練集
train_dataset = (train_examples  # 輸出：(中文句子, 台語句子)
                 .map(tf_encode) # 輸出：(英文索引序列, 中文索引序列)
                 .filter(filter_max_length) # 同上，且序列長度都不超過 40
                 .cache() # 加快讀取數據
                 .shuffle(BUFFER_SIZE) # 將例子洗牌確保隨機性
                 .padded_batch(BATCH_SIZE, # 將 batch 裡的序列都 pad 到一樣長度
                               padded_shapes=([-1], [-1]))
                 .prefetch(tf.data.experimental.AUTOTUNE)) # 加速
# 驗證集
val_dataset = (val_examples
               .map(tf_encode)
               .filter(filter_max_length)
               .padded_batch(BATCH_SIZE, 
                             padded_shapes=([-1], [-1])))
```

簡單檢查是否有序列超過我們指定的長度，順便計算過濾掉過長序列後剩餘的訓練集筆數
```
# # 因為我們數據量小可以這樣 count
# num_examples = 0
# for ch_indices, zh_indices in train_dataset:
#   cond1 = len(ch_indices) <= MAX_LENGTH
#   cond2 = len(zh_indices) <= MAX_LENGTH
#   #assert cond1 and cond2
#   #if cond1 and cond2:
#   num_examples += 1

# print(f"所有序列長度都不超過 {MAX_LENGTH} 個 tokens")
# print(f"訓練資料集裡總共有 {num_examples} 筆數據")
```
看看最後建立出來的資料集長什麼樣子
```
ch_batch, zh_batch = next(iter(train_dataset))
print("中文索引序列的 batch")
print(ch_batch)
print('-' * 20)
print("台語索引序列的 batch")
print(zh_batch)
```

Transformer
建立兩個要拿來持續追蹤的中台平行句子
```
demo_examples = [
    ("佇菜園種有機青菜 ", "ti tshai hng tsing iu ki1 tshenn tshai"),
    ("反倒轉利用環保的生態防治法", "huan to tng li iong khuan po e senn thai hong ti huat"),
]
pprint(demo_examples)
```
接著利用之前建立資料集的方法將這 2 組中台句子做些前處理並以 Tensor 的方式讀出
```
batch_size = 2
demo_examples = tf.data.Dataset.from_tensor_slices((
    [ch for ch, _ in demo_examples], [zh for _, zh in demo_examples]
))

# 將兩個句子透過之前定義的字典轉換成子詞的序列（sequence of subwords）
# 並添加 padding token: <pad> 來確保 batch 裡的句子有一樣長度
demo_dataset = demo_examples.map(tf_encode)\
  .padded_batch(batch_size, padded_shapes=([-1], [-1]))

# 取出這個 demo dataset 裡唯一一個 batch
inp, tar = next(iter(demo_dataset))
print('inp:', inp)
print('' * 10)
print('tar:', tar)
```
視覺化 3 維詞嵌入張量
在將索引序列丟入神經網路之前，先做word embedding，將一個維度為字典大小的高維離散空間「嵌入」到低維的連續空間裡頭。

為中文與台語分別建立一個詞嵌入層並實際對 inp 及 tar 做轉換：
```
# + 2 是因為我們額外加了 <start> 以及 <end> tokens
vocab_size_ch = subword_encoder_ch.vocab_size + 2
vocab_size_zh = subword_encoder_zh.vocab_size + 2

# 為了方便 demo, 將詞彙轉換到一個 4 維的詞嵌入空間
d_model = 4
embedding_layer_ch = tf.keras.layers.Embedding(vocab_size_ch, d_model)
embedding_layer_zh = tf.keras.layers.Embedding(vocab_size_zh, d_model)

emb_inp = embedding_layer_ch(inp)
emb_tar = embedding_layer_zh(tar)
emb_inp, emb_tar
```
```
print("tar[0]:", tar[0][-3:])
print("-" * 20)
print("emb_tar[0]:", emb_tar[0][-3:])
```
create_padding_mask :建立Transformer 裡的 mask
```
def create_padding_mask(seq):
  # padding mask 的工作就是把索引序列中為 0 的位置設為 1
  mask = tf.cast(tf.equal(seq, 0), tf.float32)
  return mask[:, tf.newaxis, tf.newaxis, :] #　broadcasting

inp_mask = create_padding_mask(inp)
inp_mask
```
將inp_mask額外的維度去掉以方便跟 inp 作比較
```
print("inp:", inp)
print("-" * 20)
print("tf.squeeze(inp_mask):", tf.squeeze(inp_mask))
```


注意函式
注意力機制（或稱注意函式，attention function）概念上就是拿一個查詢（query）去跟一組 key-values 做運算，最後產生一個輸出。只是我們會利用矩陣運算同時讓多個查詢跟一組 key-values 做運算，最大化計算效率。

而不管是查詢（query）、鍵值（keys）還是值（values）或是輸出，全部都是向量（vectors）。該輸出是 values 的加權平均，而每個 value 獲得的權重則是由當初 value 對應的 key 跟 query 計算匹配程度所得來的。
![](https://i.imgur.com/drOeitN.jpg)

拿已經被轉換成詞嵌入空間的中文張量 emb_inp 來充當左圖中的 Q 以及 K，讓它自己跟自己做匹配。V 則隨機產生一個 binary 張量（裡頭只有 1 或 0）來當作每個 K 所對應的值，方便直觀解讀 scaled dot product attention 的輸出結果
```
# 設定一個 seed 確保我們每次都拿到一樣的隨機結果
tf.random.set_seed(9527)

# 自注意力機制：查詢 `q` 跟鍵值 `k` 都是 `emb_inp`
q = emb_inp
k = emb_inp
# 簡單產生一個跟 `emb_inp` 同樣 shape 的 binary vector
v = tf.cast(tf.math.greater(tf.random.uniform(shape=emb_inp.shape), 0.5), tf.float32)
v
```



Scaled dot product attention：注意函式
邏輯:
- 將 q 和 k 做點積得到 matmul_qk
- 將 matmul_qk 除以 scaling factor sqrt(dk)
- 有遮罩的話在丟入 softmax 前套用
- 通過 softmax 取得加總為 1 的注意權重
- 以該權重加權平均 v 作為輸出結果
- 回傳輸出結果以及注意權重
```
def scaled_dot_product_attention(q, k, v, mask):
  """Calculate the attention weights.
  q, k, v must have matching leading dimensions.
  k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
  The mask has different shapes depending on its type(padding or look ahead) 
  but it must be broadcastable for addition.
  
  Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable 
          to (..., seq_len_q, seq_len_k). Defaults to None.
    
  Returns:
    output, attention_weights
  """
  # 將 `q`、 `k` 做點積再 scale
  matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
  
  dk = tf.cast(tf.shape(k)[-1], tf.float32)  # 取得 seq_k 的序列長度
  scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)  # scale by sqrt(dk)

  # 將遮罩「加」到被丟入 softmax 前的 logits
  if mask is not None:
    scaled_attention_logits += (mask * -1e9)

  # 取 softmax 是為了得到總和為 1 的比例之後對 `v` 做加權平均
  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)
  
  # 以注意權重對 v 做加權平均（weighted average）
  output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

  return output, attention_weights
  ```


際將 q, k, v 輸入此函式看看得到的結果。假設沒有遮罩的存在
```
mask = None
output, attention_weights = scaled_dot_product_attention(q, k, v, mask)
print("output:", output)
print("-" * 20)
print("attention_weights:", attention_weights)
```
加入遮罩
```
def create_padding_mask(seq):
  # padding mask 的工作就是把索引序列中為 0 的位置設為 1
  mask = tf.cast(tf.equal(seq, 0), tf.float32)
  return mask[:, tf.newaxis, tf.newaxis, :] #　broadcasting

print("inp:", inp)
inp_mask = create_padding_mask(inp)
print("-" * 20)
print("inp_mask:", inp_mask)

# 這次讓我們將 padding mask 放入注意函式並觀察
# 注意權重的變化
mask = tf.squeeze(inp_mask, axis=1) # (batch_size, 1, seq_len_q)
_, attention_weights = scaled_dot_product_attention(q, k, v, mask)
print("attention_weights:", attention_weights)

# 事實上也不完全是上句話的翻譯，
# 因為我們在第一個維度還是把兩個句子都拿出來方便你比較
attention_weights[:, :, -2:]

```
另一種遮罩 look ahead mask，用 look ahead mask 的結果就是讓序列 q 裡的每個字詞只關注包含自己左側的子詞，在自己之後的位置的字詞都不看。
```
# 建立一個 2 維矩陣，維度為 (size, size)，
# 其遮罩為一個右上角的三角形
def create_look_ahead_mask(size):
  mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
  return mask  # (seq_len, seq_len)

seq_len = emb_tar.shape[1] # 注意這次我們用中文的詞嵌入張量 `emb_tar`
look_ahead_mask = create_look_ahead_mask(seq_len)
print("emb_tar:", emb_tar)
print("-" * 20)
print("look_ahead_mask", look_ahead_mask)
```

用目標語言（台語）的 batch來模擬 Decoder 處理的情況
```
# 用目標語言（台語）的 batch
# 來模擬 Decoder 處理的情況
temp_q = temp_k = emb_tar
temp_v = tf.cast(tf.math.greater(
    tf.random.uniform(shape=emb_tar.shape), 0.5), tf.float32)

# 將 look_ahead_mask 放入注意函式
_, attention_weights = scaled_dot_product_attention(
    temp_q, temp_k, temp_v, look_ahead_mask)

print("attention_weights:", attention_weights)

attention_weights[:, 0, :]
attention_weights[:, 1, :]
```
為了實現 multi-head attention，先使用split_heads函式把一個 head 變成多個 heads。實際上就是把一個 d_model 維度的向量「折」成 num_heads 個 depth 維向量，使得：num_heads * depth = d_model
然後將中文詞嵌入張量 emb_inp 實際丟進去看看
```
def split_heads(x, d_model, num_heads):
  # x.shape: (batch_size, seq_len, d_model)
  batch_size = tf.shape(x)[0]
  
  # 我們要確保維度 `d_model` 可以被平分成 `num_heads` 個 `depth` 維度
  assert d_model % num_heads == 0
  depth = d_model // num_heads  # 這是分成多頭以後每個向量的維度 
  
  # 將最後一個 d_model 維度分成 num_heads 個 depth 維度。
  # 最後一個維度變成兩個維度，張量 x 從 3 維到 4 維
  # (batch_size, seq_len, num_heads, depth)
  reshaped_x = tf.reshape(x, shape=(batch_size, -1, num_heads, depth))
  
  # 將 head 的維度拉前使得最後兩個維度為子詞以及其對應的 depth 向量
  # (batch_size, num_heads, seq_len, depth)
  output = tf.transpose(reshaped_x, perm=[0, 2, 1, 3])
  
  return output

# 我們的 `emb_inp` 裡頭的子詞本來就是 4 維的詞嵌入向量
d_model = 4
# 將 4 維詞嵌入向量分為 2 個 head 的 2 維矩陣
num_heads = 2
x = emb_inp

output = split_heads(x, d_model, num_heads)  
print("x:", x)
print("output:", output)
```
multi-head attention 實現
```
# 實作一個執行多頭注意力機制的 keras layer
# 在初始的時候指定輸出維度 `d_model` & `num_heads，
# 在呼叫的時候輸入 `v`, `k`, `q` 以及 `mask`
# 輸出跟 scaled_dot_product_attention 函式一樣有兩個：
# output.shape            == (batch_size, seq_len_q, d_model)
# attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
class MultiHeadAttention(tf.keras.layers.Layer):
  # 在初始的時候建立一些必要參數
  def __init__(self, d_model, num_heads):
    super(MultiHeadAttention, self).__init__()
    self.num_heads = num_heads # 指定要將 `d_model` 拆成幾個 heads
    self.d_model = d_model # 在 split_heads 之前的基底維度
    
    assert d_model % self.num_heads == 0  # 前面看過，要確保可以平分
    
    self.depth = d_model // self.num_heads  # 每個 head 裡子詞的新的 repr. 維度
    
    self.wq = tf.keras.layers.Dense(d_model)  # 分別給 q, k, v 的 3 個線性轉換 
    self.wk = tf.keras.layers.Dense(d_model)  # 注意我們並沒有指定 activation func
    self.wv = tf.keras.layers.Dense(d_model)
    
    self.dense = tf.keras.layers.Dense(d_model)  # 多 heads 串接後通過的線性轉換
  
  # 這跟我們前面看過的函式有 87% 相似
  def split_heads(self, x, batch_size):
    """Split the last dimension into (num_heads, depth).
    Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
    """
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])
  
  # multi-head attention 的實際執行流程，注意參數順序（這邊跟論文以及 TensorFlow 官方教學一致）
  def call(self, v, k, q, mask):
    batch_size = tf.shape(q)[0]
    
    # 將輸入的 q, k, v 都各自做一次線性轉換到 `d_model` 維空間
    q = self.wq(q)  # (batch_size, seq_len, d_model)
    k = self.wk(k)  # (batch_size, seq_len, d_model)
    v = self.wv(v)  # (batch_size, seq_len, d_model)
    
    # 前面看過的，將最後一個 `d_model` 維度分成 `num_heads` 個 `depth` 維度
    q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
    k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
    v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
    
    # 利用 broadcasting 讓每個句子的每個 head 的 qi, ki, vi 都各自進行注意力機制
    # 輸出會多一個 head 維度
    scaled_attention, attention_weights = scaled_dot_product_attention(
        q, k, v, mask)
    # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
    # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
    
    # 跟我們在 `split_heads` 函式做的事情剛好相反，先做 transpose 再做 reshape
    # 將 `num_heads` 個 `depth` 維度串接回原來的 `d_model` 維度
    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
    # (batch_size, seq_len_q, num_heads, depth)
    concat_attention = tf.reshape(scaled_attention, 
                                  (batch_size, -1, self.d_model)) 
    # (batch_size, seq_len_q, d_model)

    # 通過最後一個線性轉換
    output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
        
    return output, attention_weights
```
初始一個 multi-head attention layer 並將中文詞嵌入向量 emb_inp 輸入進去看看
```
# emb_inp.shape == (batch_size, seq_len, d_model)
#               == (2, 8, 4)
assert d_model == emb_inp.shape[-1]  == 4
num_heads = 2

print(f"d_model: {d_model}")
print(f"num_heads: {num_heads}\n")

# 初始化一個 multi-head attention layer
mha = MultiHeadAttention(d_model, num_heads)

# 簡單將 v, k, q 都設置為 `emb_inp`
# 順便看看 padding mask 的作用。
# 別忘記，第一個英文序列的最後兩個 tokens 是 <pad>
v = k = q = emb_inp
padding_mask = create_padding_mask(inp)
print("q.shape:", q.shape)
print("k.shape:", k.shape)
print("v.shape:", v.shape)
print("padding_mask.shape:", padding_mask.shape)

output, attention_weights = mha(v, k, q, mask)
print("output.shape:", output.shape)
print("attention_weights.shape:", attention_weights.shape)

print("\noutput:", output)
```

Position-wise Feed-Forward Network
建立 Transformer 裡 Encoder / Decoder layer 都要使用到的 Feed Forward 元件
```
def point_wise_feed_forward_network(d_model, dff):
  
  # 此 FFN 對輸入做兩個線性轉換，中間加了一個 ReLU activation func
  return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
  ])
```
```
batch_size = 64
seq_len = 10
d_model = 512
dff = 2048

x = tf.random.uniform((batch_size, seq_len, d_model))
ffn = point_wise_feed_forward_network(d_model, dff)
out = ffn(x)
print("x.shape:", x.shape)
print("out.shape:", out.shape)

```

建立一個全連接前饋神經網路（Fully-connected Feed Forward Network，FFN）試試
```
batch_size = 64
seq_len = 10
d_model = 512
dff = 2048

x = tf.random.uniform((batch_size, seq_len, d_model))
ffn = point_wise_feed_forward_network(d_model, dff)
out = ffn(x)
print("x.shape:", x.shape)
print("out.shape:", out.shape)

```
同一個子詞不會因為位置的改變而造成 FFN 的輸出結果產生差異。但因為我們實際上會有多個 Encoder / Decoder layers，而每個 layers 都會有不同參數的 FFN，因此每個 layer 裡頭的 FFN 做的轉換都會有所不同。
```
d_model = 4 # FFN 的輸入輸出張量的最後一維皆為 `d_model`
dff = 6

# 建立一個小 FFN
small_ffn = point_wise_feed_forward_network(d_model, dff)
dummy_sentence = tf.constant([[5, 5, 6, 6], 
                              [5, 5, 6, 6], 
                              [9, 5, 2, 7], 
                              [9, 5, 2, 7],
                              [9, 5, 2, 7]], dtype=tf.float32)
small_ffn(dummy_sentence)
```

Encoder layer 實作
```
# Encoder 裡頭會有 N 個 EncoderLayers，而每個 EncoderLayer 裡又有兩個 sub-layers: MHA & FFN
class EncoderLayer(tf.keras.layers.Layer):
  # Transformer 論文內預設 dropout rate 為 0.1
  def __init__(self, d_model, num_heads, dff, rate=0.1):
    super(EncoderLayer, self).__init__()

    self.mha = MultiHeadAttention(d_model, num_heads)
    self.ffn = point_wise_feed_forward_network(d_model, dff)

    # layer norm 很常在 RNN-based 的模型被使用。一個 sub-layer 一個 layer norm
    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    
    # 一樣，一個 sub-layer 一個 dropout layer
    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)
    
  # 需要丟入 `training` 參數是因為 dropout 在訓練以及測試的行為有所不同
  def call(self, x, training, mask):
    # 除了 `attn`，其他張量的 shape 皆為 (batch_size, input_seq_len, d_model)
    # attn.shape == (batch_size, num_heads, input_seq_len, input_seq_len)
    
    # sub-layer 1: MHA
    # Encoder 利用注意機制關注自己當前的序列，因此 v, k, q 全部都是自己
    # 另外別忘了我們還需要 padding mask 來遮住輸入序列中的 <pad> token
    attn_output, attn = self.mha(x, x, x, mask)  
    attn_output = self.dropout1(attn_output, training=training) 
    out1 = self.layernorm1(x + attn_output)  
    
    # sub-layer 2: FFN
    ffn_output = self.ffn(out1) 
    ffn_output = self.dropout2(ffn_output, training=training)  # 記得 training
    out2 = self.layernorm2(out1 + ffn_output)
    
    return out2
```
把詞嵌入張量丟進去Encode layer看看
```
# 之後可以調的超參數。這邊為了 demo 設小一點
d_model = 4
num_heads = 2
dff = 8

# 新建一個使用上述參數的 Encoder Layer
enc_layer = EncoderLayer(d_model, num_heads, dff)
padding_mask = create_padding_mask(inp)  # 建立一個當前輸入 batch 使用的 padding mask
enc_out = enc_layer(emb_inp, training=False, mask=padding_mask)  # (batch_size, seq_len, d_model)

print("inp:", inp)
print("-" * 20)
print("padding_mask:", padding_mask)
print("-" * 20)
print("emb_inp:", emb_inp)
print("-" * 20)
print("enc_out:", enc_out)
assert emb_inp.shape == enc_out.shape
```
Decode Layer實作
```
# Decoder 裡頭會有 N 個 DecoderLayer，
# 而 DecoderLayer 又有三個 sub-layers: 自注意的 MHA, 關注 Encoder 輸出的 MHA & FFN
class DecoderLayer(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, dff, rate=0.1):
    super(DecoderLayer, self).__init__()

    # 3 個 sub-layers 的主角們
    self.mha1 = MultiHeadAttention(d_model, num_heads)
    self.mha2 = MultiHeadAttention(d_model, num_heads)
    self.ffn = point_wise_feed_forward_network(d_model, dff)
 
    # 定義每個 sub-layer 用的 LayerNorm
    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    
    # 定義每個 sub-layer 用的 Dropout
    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)
    self.dropout3 = tf.keras.layers.Dropout(rate)
    
    
  def call(self, x, enc_output, training, 
           combined_mask, inp_padding_mask):
    # 所有 sub-layers 的主要輸出皆為 (batch_size, target_seq_len, d_model)
    # enc_output 為 Encoder 輸出序列，shape 為 (batch_size, input_seq_len, d_model)
    # attn_weights_block_1 則為 (batch_size, num_heads, target_seq_len, target_seq_len)
    # attn_weights_block_2 則為 (batch_size, num_heads, target_seq_len, input_seq_len)

    # sub-layer 1: Decoder layer 自己對輸出序列做注意力。
    # 我們同時需要 look ahead mask 以及輸出序列的 padding mask 
    # 來避免前面已生成的子詞關注到未來的子詞以及 <pad>
    attn1, attn_weights_block1 = self.mha1(x, x, x, combined_mask)
    attn1 = self.dropout1(attn1, training=training)
    out1 = self.layernorm1(attn1 + x)
    
    # sub-layer 2: Decoder layer 關注 Encoder 的最後輸出
    # 記得我們一樣需要對 Encoder 的輸出套用 padding mask 避免關注到 <pad>
    attn2, attn_weights_block2 = self.mha2(
        enc_output, enc_output, out1, inp_padding_mask)  # (batch_size, target_seq_len, d_model)
    attn2 = self.dropout2(attn2, training=training)
    out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)
    
    # sub-layer 3: FFN 部分跟 Encoder layer 完全一樣
    ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)

    ffn_output = self.dropout3(ffn_output, training=training)
    out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)
    
    # 除了主要輸出 `out3` 以外，輸出 multi-head 注意權重方便之後理解模型內部狀況
    return out3, attn_weights_block1, attn_weights_block2
```
產生 comined_mask ，把兩個遮罩取大的
```
tar_padding_mask = create_padding_mask(tar)
look_ahead_mask = create_look_ahead_mask(tar.shape[-1])
combined_mask = tf.maximum(tar_padding_mask, look_ahead_mask)

print("tar:", tar)
print("-" * 20)
print("tar_padding_mask:", tar_padding_mask)
print("-" * 20)
print("look_ahead_mask:", look_ahead_mask)
print("-" * 20)
print("combined_mask:", combined_mask)
```
把台語（目標語言）的詞嵌入張量以及相關的遮罩丟進去Decoder layer看看
```
# 超參數
d_model = 4
num_heads = 2
dff = 8
dec_layer = DecoderLayer(d_model, num_heads, dff)

# 來源、目標語言的序列都需要 padding mask
inp_padding_mask = create_padding_mask(inp)
tar_padding_mask = create_padding_mask(tar)

# masked MHA 用的遮罩，把 padding 跟未來子詞都蓋住
look_ahead_mask = create_look_ahead_mask(tar.shape[-1])
combined_mask = tf.maximum(tar_padding_mask, look_ahead_mask)

# 實際初始一個 decoder layer 並做 3 個 sub-layers 的計算
dec_out, dec_self_attn_weights, dec_enc_attn_weights = dec_layer(
    emb_tar, enc_out, False, combined_mask, inp_padding_mask)

print("emb_tar:", emb_tar)
print("-" * 20)
print("enc_out:", enc_out)
print("-" * 20)
print("dec_out:", dec_out)
assert emb_tar.shape == dec_out.shape
print("-" * 20)
print("dec_self_attn_weights.shape:", dec_self_attn_weights.shape)
print("dec_enc_attn_weights:", dec_enc_attn_weights.shape)
```

Positional encoding位置編碼公式
```
# 以下直接參考 TensorFlow 官方 tutorial 
def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
  return pos * angle_rates

def positional_encoding(position, d_model):
  angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)
  
  # apply sin to even indices in the array; 2i
  sines = np.sin(angle_rads[:, 0::2])
  
  # apply cos to odd indices in the array; 2i+1
  cosines = np.cos(angle_rads[:, 1::2])
  
  pos_encoding = np.concatenate([sines, cosines], axis=-1)
  
  pos_encoding = pos_encoding[np.newaxis, ...]
    
  return tf.cast(pos_encoding, dtype=tf.float32)


seq_len = 50
d_model = 512

pos_encoding = positional_encoding(seq_len, d_model)
pos_encoding
```
把位置編碼畫出
```
plt.pcolormesh(pos_encoding[0], cmap='RdBu')
plt.xlabel('d_model')
plt.xlim((0, 512))
plt.ylabel('Position')
plt.colorbar()
plt.show()
```
![](https://i.imgur.com/0xp70KE.png)


Encoder Class
主要包含 3 個元件：
- 輸入的詞嵌入層
- 位置編碼
- N 個 Encoder layers
```
class Encoder(tf.keras.layers.Layer):
  # Encoder 的初始參數除了本來就要給 EncoderLayer 的參數還多了：
  # - num_layers: 決定要有幾個 EncoderLayers, 前面影片中的 `N`
  # - input_vocab_size: 用來把索引轉成詞嵌入向量
  def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, 
               rate=0.1):
    super(Encoder, self).__init__()

    self.d_model = d_model
    
    self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
    self.pos_encoding = positional_encoding(input_vocab_size, self.d_model)
    
    # 建立 `num_layers` 個 EncoderLayers
    self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) 
                       for _ in range(num_layers)]

    self.dropout = tf.keras.layers.Dropout(rate)
        
  def call(self, x, training, mask):
    # 輸入的 x.shape == (batch_size, input_seq_len)
    # 以下各 layer 的輸出皆為 (batch_size, input_seq_len, d_model)
    input_seq_len = tf.shape(x)[1]
    
    # 將 2 維的索引序列轉成 3 維的詞嵌入張量，並依照論文乘上 sqrt(d_model)
    # 再加上對應長度的位置編碼
    x = self.embedding(x)
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x += self.pos_encoding[:, :input_seq_len, :]

    # 對 embedding 跟位置編碼的總合做 regularization
    # 這在 Decoder 也會做
    x = self.dropout(x, training=training)
    
    # 通過 N 個 EncoderLayer 做編碼
    for i, enc_layer in enumerate(self.enc_layers):
      x = enc_layer(x, training, mask)
      # 以下只是用來 demo EncoderLayer outputs
      #print('-' * 20)
      #print(f"EncoderLayer {i + 1}'s output:", x)
      
    
    return x 
```
將索引序列 inp 丟入 Encoder
```
# 超參數
num_layers = 2 # 2 層的 Encoder
d_model = 4
num_heads = 2
dff = 8
input_vocab_size = subword_encoder_ch.vocab_size + 2 # 記得加上 <start>, <end>

# 初始化一個 Encoder
encoder = Encoder(num_layers, d_model, num_heads, dff, input_vocab_size)

# 將 2 維的索引序列丟入 Encoder 做編碼
enc_out = encoder(inp, training=False, mask=None)
print("inp:", inp)
print("-" * 20)
print("enc_out:", enc_out)
```
Decoder class
在 Decoder 裡頭我們只需要建立一個專門給台語用的詞嵌入層以及位置編碼即可。我們在呼叫每個 Decoder layer 的時候也順便把其注意權重存下來，以方便了解模型訓練完後是怎麼做翻譯的。
```
class Decoder(tf.keras.layers.Layer):
  # 初始參數跟 Encoder 只差在用 `target_vocab_size` 而非 `inp_vocab_size`
  def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size, 
               rate=0.1):
    super(Decoder, self).__init__()

    self.d_model = d_model
    
    # 為中文（目標語言）建立詞嵌入層
    self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
    self.pos_encoding = positional_encoding(target_vocab_size, self.d_model)
    
    self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) 
                       for _ in range(num_layers)]
    self.dropout = tf.keras.layers.Dropout(rate)
  
  # 呼叫時的參數跟 DecoderLayer 一模一樣
  def call(self, x, enc_output, training, 
           combined_mask, inp_padding_mask):
    
    tar_seq_len = tf.shape(x)[1]
    attention_weights = {}  # 用來存放每個 Decoder layer 的注意權重
    
    # 這邊跟 Encoder 做的事情完全一樣
    x = self.embedding(x)  # (batch_size, tar_seq_len, d_model)
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x += self.pos_encoding[:, :tar_seq_len, :]
    x = self.dropout(x, training=training)

    
    for i, dec_layer in enumerate(self.dec_layers):
      x, block1, block2 = dec_layer(x, enc_output, training,
                                    combined_mask, inp_padding_mask)
      
      # 將從每個 Decoder layer 取得的注意權重全部存下來回傳，方便我們觀察
      attention_weights['decoder_layer{}_block1'.format(i + 1)] = block1
      attention_weights['decoder_layer{}_block2'.format(i + 1)] = block2
    
    # x.shape == (batch_size, tar_seq_len, d_model)
    return x, attention_weights
```
初始並呼叫一個 Decoder 看看
```
# 超參數
num_layers = 2 # 2 層的 Decoder
d_model = 4
num_heads = 2
dff = 8
target_vocab_size = subword_encoder_zh.vocab_size + 2 # 記得加上 <start>, <end>

# 遮罩
inp_padding_mask = create_padding_mask(inp)
tar_padding_mask = create_padding_mask(tar)
look_ahead_mask = create_look_ahead_mask(tar.shape[1])
combined_mask = tf.math.maximum(tar_padding_mask, look_ahead_mask)

# 初始化一個 Decoder
decoder = Decoder(num_layers, d_model, num_heads, dff, target_vocab_size)

# 將 2 維的索引序列以及遮罩丟入 Decoder
print("tar:", tar)
print("-" * 20)
print("combined_mask:", combined_mask)
print("-" * 20)
print("enc_out:", enc_out)
print("-" * 20)
print("inp_padding_mask:", inp_padding_mask)
print("-" * 20)
dec_out, attn = decoder(tar, enc_out, training=False, 
                        combined_mask=combined_mask,
                        inp_padding_mask=inp_padding_mask)
print("dec_out:", dec_out)
print("-" * 20)
for block_name, attn_weights in attn.items():
  print(f"{block_name}.shape: {attn_weights.shape}")
```

實作 Transformer 
- Encoder
- Decoder
- Final linear layer
```
# Transformer 之上已經沒有其他 layers 了，我們使用 tf.keras.Model 建立一個模型
class Transformer(tf.keras.Model):
  # 初始參數包含 Encoder & Decoder 都需要超參數以及中英字典數目
  def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, 
               target_vocab_size, rate=0.1):
    super(Transformer, self).__init__()

    self.encoder = Encoder(num_layers, d_model, num_heads, dff, 
                           input_vocab_size, rate)

    self.decoder = Decoder(num_layers, d_model, num_heads, dff, 
                           target_vocab_size, rate)
    # 這個 FFN 輸出跟台語字典一樣大的 logits 數，等通過 softmax 就代表每個台語字的出現機率
    self.final_layer = tf.keras.layers.Dense(target_vocab_size)
  
  # enc_padding_mask 跟 dec_padding_mask 都是中文序列的 padding mask，
  # 只是一個給 Encoder layer 的 MHA 用，一個是給 Decoder layer 的 MHA 2 使用
  def call(self, inp, tar, training, enc_padding_mask, 
           combined_mask, dec_padding_mask):

    enc_output = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)
    
    # dec_output.shape == (batch_size, tar_seq_len, d_model)
    dec_output, attention_weights = self.decoder(
        tar, enc_output, training, combined_mask, dec_padding_mask)
    
    # 將 Decoder 輸出通過最後一個 linear layer
    final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)
    
    return final_output, attention_weights
```
建一個 Transformer，並假設我們已經準備好用 demo 數據來訓練它做中翻台

```
# 超參數
num_layers = 1
d_model = 4
num_heads = 2
dff = 8

# + 2 是為了 <start> & <end> token
input_vocab_size = subword_encoder_ch.vocab_size + 2
output_vocab_size = subword_encoder_zh.vocab_size + 2

# 重點中的重點。訓練時用前一個字來預測下一個台語字
tar_inp = tar[:, :-1]
tar_real = tar[:, 1:]

# 來源 / 目標語言用的遮罩。注意 `comined_mask` 已經將目標語言的兩種遮罩合而為一
inp_padding_mask = create_padding_mask(inp)
tar_padding_mask = create_padding_mask(tar_inp)
look_ahead_mask = create_look_ahead_mask(tar_inp.shape[1])
combined_mask = tf.math.maximum(tar_padding_mask, look_ahead_mask)

# 初始化我們的第一個 transformer
transformer = Transformer(num_layers, d_model, num_heads, dff, 
                          input_vocab_size, output_vocab_size)

# 將中文台語序列丟入取得 Transformer 預測下個台語字的結果
predictions, attn_weights = transformer(inp, tar_inp, False, inp_padding_mask, 
                                        combined_mask, inp_padding_mask)

print("tar:", tar)
print("-" * 20)
print("tar_inp:", tar_inp)
print("-" * 20)
print("tar_real:", tar_real)
print("-" * 20)
print("predictions:", predictions)
```
定義損失函數
loss_object 實際算 cross entropy 
```
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

# 假設我們要解的是一個 binary classifcation， 0 跟 1 個代表一個 label
real = tf.constant([1, 1, 0], shape=(1, 3), dtype=tf.float32)
pred = tf.constant([[0, 1], [0, 1], [0, 1]], dtype=tf.float32)
loss_object(real, pred)
```
另外一個函式來建立遮罩並加總序列裡頭不包含 token 位置的損失
```
def loss_function(real, pred):
  # 這次的 mask 將序列中不等於 0 的位置視為 1，其餘為 0 
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  # 照樣計算所有位置的 cross entropy 但不加總
  loss_ = loss_object(real, pred)
  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask  # 只計算非 <pad> 位置的損失 
  
  return tf.reduce_mean(loss_)
```

另外再定義兩個 tf.keras.metrics，方便之後使用 TensorBoard 來追蹤模型 performance
```
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='train_accuracy')

```
設置超參數
```
num_layers = 4 
d_model = 128
dff = 512
num_heads = 8

input_vocab_size = subword_encoder_ch.vocab_size + 2
target_vocab_size = subword_encoder_zh.vocab_size + 2
dropout_rate = 0.1  # 預設值

print("input_vocab_size:", input_vocab_size)
print("target_vocab_size:", target_vocab_size)
```
設置 Optimizer，使用 Adam optimizer 以及自定義的 learning rate schedule
```
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  # 論文預設 `warmup_steps` = 4000
  def __init__(self, d_model, warmup_steps=4000):
    super(CustomSchedule, self).__init__()
    
    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps
    
  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)
    
    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
  
# 將客製化 learning rate schdeule 丟入 Adam opt.
# Adam opt. 的參數都跟論文相同
learning_rate = CustomSchedule(d_model)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, 
                                     epsilon=1e-9)
```

觀察看看這個 schedule 是怎麼隨著訓練步驟而改變 learning rate 的
```
d_models = [128, 256, 512]
warmup_steps = [1000 * i for i in range(1, 4)]

schedules = []
labels = []
colors = ["blue", "red", "black"]
for d in d_models:
  schedules += [CustomSchedule(d, s) for s in warmup_steps]
  labels += [f"d_model: {d}, warm: {s}" for s in warmup_steps]

for i, (schedule, label) in enumerate(zip(schedules, labels)):
  plt.plot(schedule(tf.range(10000, dtype=tf.float32)), 
           label=label, color=colors[i // 3])

plt.legend()

plt.ylabel("Learning Rate")
plt.xlabel("Train Step")
```
![](https://i.imgur.com/7LrYaH8.png)
所有 schedules 都先經過 warmup_steps 個步驟直線提升 learning rate，接著逐漸平滑下降。另外我們也會給比較高維的 d_model 維度比較小的 learning rate。

使用前面已經定義好的超參數來初始化一個全新的 Transformer：
```
transformer = Transformer(num_layers, d_model, num_heads, dff,
                          input_vocab_size, target_vocab_size, dropout_rate)

print(f"""這個 Transformer 有 {num_layers} 層 Encoder / Decoder layers
d_model: {d_model}
num_heads: {num_heads}
dff: {dff}
input_vocab_size: {input_vocab_size}
target_vocab_size: {target_vocab_size}
dropout_rate: {dropout_rate}

""")
```

設置 checkpoint 來定期儲存 / 讀取模型及 optimizer
```
train_perc = 10
val_prec = 1
drop_prec = 100 - train_perc - val_prec

# 方便比較不同實驗/ 不同超參數設定的結果
run_id = f"{num_layers}layers_{d_model}d_{num_heads}heads_{dff}dff_{train_perc}train_perc"
checkpoint_path = os.path.join(checkpoint_path, run_id)
log_dir = os.path.join(log_dir, run_id)

# tf.train.Checkpoint 可以幫我們把想要存下來的東西整合起來，方便儲存與讀取
# 一般來說你會想存下模型以及 optimizer 的狀態
ckpt = tf.train.Checkpoint(transformer=transformer,
                           optimizer=optimizer)

# ckpt_manager 會去 checkpoint_path 看有沒有符合 ckpt 裡頭定義的東西
# 存檔的時候只保留最近 5 次 checkpoints，其他自動刪除
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# 如果在 checkpoint 路徑上有發現檔案就讀進來
if ckpt_manager.latest_checkpoint:
  ckpt.restore(ckpt_manager.latest_checkpoint)
  
  # 用來確認之前訓練多少 epochs 了
  last_epoch = int(ckpt_manager.latest_checkpoint.split("-")[-1])
  print(f'已讀取最新的 checkpoint，模型已訓練 {last_epoch} epochs。')
else:
  last_epoch = 0
  print("沒找到 checkpoint，從頭訓練。")
```




create_masks函式產生所有的遮罩
```
# 為 Transformer 的 Encoder / Decoder 準備遮罩
def create_masks(inp, tar):
  # 英文句子的 padding mask，要交給 Encoder layer 自注意力機制用的
  enc_padding_mask = create_padding_mask(inp)
  
  # 同樣也是英文句子的 padding mask，但是是要交給 Decoder layer 的 MHA 2 
  # 關注 Encoder 輸出序列用的
  dec_padding_mask = create_padding_mask(inp)
  
  # Decoder layer 的 MHA1 在做自注意力機制用的
  # `combined_mask` 是中文句子的 padding mask 跟 look ahead mask 的疊加
  look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
  dec_target_padding_mask = create_padding_mask(tar)
  combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
  
  return enc_padding_mask, combined_mask, dec_padding_mask
```
train_step 函式處理一個batch：
- 對輸入數據做些前處理（本文中的遮罩、將輸出序列左移當成正解 etc.）
- 利用 tf.GradientTape 輕鬆記錄數據被模型做的所有轉換並計算 loss
- 將梯度取出並讓 optimzier 對可被訓練的權重做梯度下降（上升）
Step:
- 對訓練數據做些必要的前處理
- 將數據丟入模型，取得預測結果
- 用預測結果跟正確解答計算 loss
- 取出梯度並利用 optimizer 做梯度下降

```
@tf.function  # 讓 TensorFlow 幫我們將 eager code 優化並加快運算
def train_step(inp, tar):
  # 前面說過的，用去尾的原始序列去預測下一個字的序列
  tar_inp = tar[:, :-1]
  tar_real = tar[:, 1:]
  
  # 建立 3 個遮罩
  enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
  
  # 紀錄 Transformer 的所有運算過程以方便之後做梯度下降
  with tf.GradientTape() as tape:
    # 注意是丟入 `tar_inp` 而非 `tar`。記得將 `training` 參數設定為 True
    predictions, _ = transformer(inp, tar_inp, 
                                 True, 
                                 enc_padding_mask, 
                                 combined_mask, 
                                 dec_padding_mask)
    # 跟影片中顯示的相同，計算左移一個字的序列跟模型預測分佈之間的差異，當作 loss
    loss = loss_function(tar_real, predictions)

  # 取出梯度並呼叫前面定義的 Adam optimizer 幫我們更新 Transformer 裡頭可訓練的參數
  gradients = tape.gradient(loss, transformer.trainable_variables)    
  optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
  
  # 將 loss 以及訓練 acc 記錄到 TensorBoard 上，非必要
  train_loss(loss)
  train_accuracy(tar_real, predictions)
```
用迴圈將數據集跑幾次
```
# 定義我們要看幾遍數據集
EPOCHS = 150
print(f"此超參數組合的 Transformer 已經訓練 {last_epoch} epochs。")
print(f"剩餘 epochs：{min(0, last_epoch - EPOCHS)}")


# 用來寫資訊到 TensorBoard，非必要但十分推薦
summary_writer = tf.summary.create_file_writer(log_dir)

# 比對設定的 `EPOCHS` 以及已訓練的 `last_epoch` 來決定還要訓練多少 epochs
for epoch in range(last_epoch, EPOCHS):
  start = time.time()
  
  # 重置紀錄 TensorBoard 的 metrics
  train_loss.reset_states()
  train_accuracy.reset_states()
  
  # 一個 epoch 就是把我們定義的訓練資料集一個一個 batch 拿出來處理，直到看完整個數據集 
  for (step_idx, (inp, tar)) in enumerate(train_dataset):
    
    # 每次 step 就是將數據丟入 Transformer，讓它生預測結果並計算梯度最小化 loss
    train_step(inp, tar)  

  # 每個 epoch 完成就存一次檔    
  if (epoch + 1) % 1 == 0:
    ckpt_save_path = ckpt_manager.save()
    print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                         ckpt_save_path))
    
  # 將 loss 以及 accuracy 寫到 TensorBoard 上
  with summary_writer.as_default():
    tf.summary.scalar("train_loss", train_loss.result(), step=epoch + 1)
    tf.summary.scalar("train_acc", train_accuracy.result(), step=epoch + 1)
  
  print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, 
                                                train_loss.result(), 
                                                train_accuracy.result()))
  print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))
```

evaluate 函式:輸入是一個完全沒有經過處理的中文句子（以字串表示），輸出則是一個索引序列，裡頭的每個索引就代表著 Transformer 預測的台語字
```
# 給定一個中文句子，輸出預測的台語索引數字序列以及注意權重 dict
def evaluate(inp_sentence):
  
  # 準備中文句子前後會加上的 <start>, <end>
  start_token = [subword_encoder_ch.vocab_size]
  end_token = [subword_encoder_ch.vocab_size + 1]
  
  # inp_sentence 是字串，我們用 Subword Tokenizer 將其變成子詞的索引序列
  # 並在前後加上 BOS / EOS
  inp_sentence = start_token + subword_encoder_ch.encode(inp_sentence) + end_token
  encoder_input = tf.expand_dims(inp_sentence, 0)
  
  # Decoder 在第一個時間點吃進去的輸入
  # 是一個只包含一個台語 <start> token 的序列
  decoder_input = [subword_encoder_zh.vocab_size]
  output = tf.expand_dims(decoder_input, 0)  # 增加 batch 維度
  
  # auto-regressive，一次生成一個台語字並將預測加到輸入再度餵進 Transformer
  for i in range(MAX_LENGTH):
    # 每多一個生成的字就得產生新的遮罩
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
        encoder_input, output)
  
    # predictions.shape == (batch_size, seq_len, vocab_size)
    predictions, attention_weights = transformer(encoder_input, 
                                                 output,
                                                 False,
                                                 enc_padding_mask,
                                                 combined_mask,
                                                 dec_padding_mask)
    

    # 將序列中最後一個 distribution 取出，並將裡頭值最大的當作模型最新的預測字
    predictions = predictions[: , -1:, :]  # (batch_size, 1, vocab_size)

    predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
    
    # 遇到 <end> token 就停止回傳，代表模型已經產生完結果
    if tf.equal(predicted_id, subword_encoder_zh.vocab_size + 1):
      return tf.squeeze(output, axis=0), attention_weights
    
    #將 Transformer 新預測的台語索引加到輸出序列中，讓 Decoder 可以在產生
    # 下個中文字的時候關注到最新的 `predicted_id`
    output = tf.concat([output, predicted_id], axis=-1)

  # 將 batch 的維度去掉後回傳預測的台語索引序列
  return tf.squeeze(output, axis=0), attention_weights
```

實際透過Transformer 做翻譯
```
# 要被翻譯的中文句子
sentence = "反倒轉利用環保的生態防治法"

# 取得預測的台語索引序列
predicted_seq, _ = evaluate(sentence)

# 過濾掉 <start> & <end> tokens 並用台語的 subword tokenizer 幫我們將索引序列還原回中文句子
target_vocab_size = subword_encoder_zh.vocab_size
predicted_seq_without_bos_eos = [idx for idx in predicted_seq if idx < target_vocab_size]
predicted_sentence = subword_encoder_zh.decode(predicted_seq_without_bos_eos)

print("sentence:", sentence)
print("-" * 20)
print("predicted_seq:", predicted_seq)
print("-" * 20)
print("predicted_sentence:", predicted_sentence)
```

讀入測試集資料，並一筆一筆進行翻譯
```
test = pd.read_csv(test_x_path, sep=",", header=None)#, quoting=2
#metadata_df.head(3)
test.columns = ["id", "txt"]
test = test[["id", "txt"]]
test = test.drop(0)

test_x = test["txt"].values
print(len(test_x))
submit_ans = []
count = 1
for i in test_x:
    # 取得預測的台語索引序列
  predicted_seq, _ = evaluate(i)

  # 過濾掉 <start> & <end> tokens 並用台語的 subword tokenizer 幫我們將索引序列還原回台語句子
  target_vocab_size = subword_encoder_zh.vocab_size
  predicted_seq_without_bos_eos = [idx for idx in predicted_seq if idx < target_vocab_size]
  predicted_sentence = subword_encoder_zh.decode(predicted_seq_without_bos_eos)

  # print("sentence:", i)
  print(count)
  # print("-" * 20)
  #print("predicted_seq:", predicted_seq)
  # # print("-" * 20)
  print("predicted_sentence:", predicted_sentence)
  submit_ans.append(predicted_sentence)
  count = count + 1



#MAE(valid_y,submit_ans)

```
將翻譯結果輸出到csv檔作為submit file。 

```

import csv

# co_id_np = np.array(co_id)
# year_np = np.array(year)

# data = np.array([co_id_np, year_np])

# np.savetxt("sample.csv", data.T, fmt='%s', delimiter='\t')
test_id = list(range(1,641+1))    
#with open('submit.csv', 'w', newline='',encoding='UTF-8-sig') as test_file:
os.chdir("/content/drive/MyDrive/nycu/2022ML/translate/ml-2022-nycu-translation")
# !ls
with open('311581024.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['id', 'txt'])
    for x,y in zip (test_id,submit_ans):
        writer.writerow([x,y])
```

將環境所需套件寫入requirements.txt。
```
!pip install pipreqs
!pipreqs --force
!pip freeze > requirements.txt
import sys
print(sys.modules.keys())
```







