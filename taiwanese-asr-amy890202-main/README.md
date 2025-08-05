
# 311581024_Taiwanese ASR



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

讀入資料集，進行初步處理，並將訓練資料切出訓練集與驗證集，再搭建rnn機器學習模型，以訓練集資料訓練模型、使用驗證集資料驗證模型，最後輸出測試集的預測結果。

### 前置處理

在linux環境撰寫以下test.sh腳本，腳本內容為使用sox轉換音檔格式為16 kHz sampling, signed-integer, 16 bits
```
vi test.sh
```

```
#!/bin/sh
for((c=1;c<=3119;c++))
do
    sox "$c".wav -r 16000 -e signed-integer -b 16 new/"$c".wav
done
```
運行test.sh腳本執行程式，將轉換好的音檔上傳google drive。
```
bash test.sh
```

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



colab程式 執行階段類型選gpu

檢查gpu 與 設定環境與gpu參數(避免模型訓練過程出現RecvAsync is cancelled error )
```
! nvidia-smi

import tensorflow as tf
tf.test.is_gpu_available()

gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0],True)

import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="true"
TF_FORCE_GPU_ALLOW_GROWTH=1

```





設定colab檔案於google drive雲端的位置 並下載與import所需函式庫
```
from google.colab import drive
drive.mount('/content/drive')


!pip install jiwer
import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from IPython import display
from jiwer import wer

import os
# os.chdir("/content/drive/MyDrive/nycu/2022ML/rnn/train/new/")
# !ls

```

設置path路徑
```

data_path = "/content/drive/MyDrive/nycu/2022ML/rnn"#/kaldi-taiwanese-asr
train_data_path = "/content/drive/MyDrive/nycu/2022ML/rnn/train/new"#/kaldi-taiwanese-asr/train/new
test_data_path = "/content/drive/MyDrive/nycu/2022ML/rnn/test/new"#/kaldi-taiwanese-asr
```

讀入lexicon.txt檔案，將檔案中出現過的子母寫入lexixon_list中。
```
f = open(data_path + "/lexicon.txt")
text = []

for line in f:
  line_list = line.split()
  if len(line_list)>2:
    for chr in line_list[0]:
      if chr not in text:
        text.append(chr)
#print(text)
lexicon_list = text
```
設定訓練集與測試集資料夾路徑，讀入train-toneless.csv作為訓練集的答案，並進行資料處理，刪除含有中文亂碼、含有大寫英文、含有未存在於lexicon.txt中的英文字元(這種情況是因為該句含有英文單字)，或長度>300字元的訓練資料。
```
wavs_path = data_path + "/train/new/"
test_wavs_path = data_path + "/test/new/"
metadata_path = data_path + "/train-toneless.csv"


# Read metadata file and parse it
metadata_df = pd.read_csv(metadata_path, sep=",", header=None)#, quoting=2
#metadata_df.head(3)
metadata_df.columns = ["id", "text"]
metadata_df = metadata_df[["id", "text"]]
print(metadata_df.shape)
#data_df = metadata_df.drop(0)#


del_list =[0,335,649,703,739,764,1028,1243,1467,2067,2087,2136,2513,2764,2778,2956]


count = 0;

for index,row in metadata_df.iterrows():
  id = row["id"]
  text = row["text"]
  #if(text.islower()==False and int(id) not in del_list):
  if((text.islower()==False or len(text)>300) and int(id) not in del_list):
    del_list.append(count)
  else:
    for chara in text:
      if (chara not in lexicon_list )and chara.isalpha():
        del_list.append(count)
  count = count+1;
print(del_list)

data_df = metadata_df.drop(del_list)
print(data_df.shape)
```
lexixon_list再append上所有已處理過後的資料中所有原先未包含在lexicon.txt中的字元。(例如各種標點符號與空白等)
```

for index,row in data_df.iterrows():
  id = row["id"]
  text = row["text"]
  if(id == "id"):
    print("????",id)
  for chr in text:
    if chr not in lexicon_list:
      lexicon_list.append(chr)
print(lexicon_list)

data_df = data_df.sample(frac=1).reset_index(drop=True)
data_df.head(3)
```
最後的lexicon_list包含['a', 'h', 'i', 'n', 'k', 'm', 'g', 'p', 't', 'u', 'b', 'e', 'o', 'j', 'l', 's', ' ', '─', '？', '。']



將訓練集再切分出訓練集與資料集
```

split = int(len(data_df) * 0.90)
df_train = data_df[:split]
df_val = data_df[split:]

print(f"training set size: {len(df_train)}")
print(f"validation set size: {len(df_val)}")

```


根據lexicon_list作為StringLookup的分類
StringLookup的char_to_num可將字元轉換為數字、num_to_char可以將數字轉換回字元。
```

# The set of characters accepted in the transcription.
characters = [x for x in lexicon_list]#lexicon_list
print(characters)
# Mapping characters to integers
char_to_num = keras.layers.StringLookup(vocabulary=characters, oov_token="")
# Mapping integers back to original characters
num_to_char = keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)

print(
    f"The vocabulary is: {char_to_num.get_vocabulary()} "
    f"(size ={char_to_num.vocabulary_size()})"
)
#label = tf.strings.lower("li be e mih kiann lan lan san san long be tsiau tsng")
    # 8. Split the label
label = tf.strings.unicode_split("li be e mih kiann lan lan san san long be tsiau tsng", input_encoding="UTF-8")
label = char_to_num(label)
#label
print(num_to_char(label).numpy())
tf.strings.reduce_join(num_to_char(label)).numpy().decode("utf-8")
#tf.strings.reduce_join(num_to_char(label)).numpy().decode("utf-8")
#print(label)
```

encode_single_sample函式讀入wav音檔，並進行解碼(decode_wav)將音檔轉換為float最後得到spectorgram的magnitude，並進行標準化。
並且將該音檔對應的羅馬拼音文字(label)根據前面的char_to_num轉為數字。
設置frame_length、frame_step、fft_length(須根據資料集進行調整)
```
# An integer scalar Tensor. The window length in samples.
frame_length = 1024
# An integer scalar Tensor. The number of samples to step.
frame_step = 160
# An integer scalar Tensor. The size of the FFT to apply.
# If not provided, uses the smallest power of 2 enclosing frame_length.
fft_length = 1024


def encode_single_sample(wav_file, label,train = 1):
    ###########################################
    ##  Process the Audio
    ##########################################
    # 1. Read wav file:
    #if train == 1:
    file = tf.io.read_file(wavs_path + wav_file + ".wav")
    # else:
    #   file = tf.io.read_file(test_wavs_path + wav_file + ".wav")
    # 2. Decode the wav file
    audio, _ = tf.audio.decode_wav(file)
    audio = tf.squeeze(audio, axis=-1)
    # 3. Change type to float
    audio = tf.cast(audio, tf.float32)
    # 4. Get the spectrogram
    spectrogram = tf.signal.stft(
        audio, frame_length=frame_length, frame_step=frame_step, fft_length=fft_length
    )
    # 5. We only need the magnitude, which can be derived by applying tf.abs
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.math.pow(spectrogram, 0.5)
    # 6. normalisation
    means = tf.math.reduce_mean(spectrogram, 1, keepdims=True)
    stddevs = tf.math.reduce_std(spectrogram, 1, keepdims=True)
    spectrogram = (spectrogram - means) / (stddevs + 1e-10)


    # ########PADDING
    # audio_len = tf.shape(spectrogram)[0]
    # # padding to 10 seconds
    # pad_len = 2754*3
    # paddings = tf.constant([[0, pad_len], [0, 0]])
    # spectrogram = tf.pad(spectrogram, paddings, "CONSTANT")[:pad_len, :]
    # ########PADDING
    ###########################################
    ##  Process the label
    ##########################################
    # 7. Convert label to Lower case
    #print(label)
    #if train == 1:
    label = tf.strings.lower(label)
    # 8. Split the label
    label = tf.strings.unicode_split(label, input_encoding="UTF-8")
    #print(label) Tensor("UnicodeSplit/UnicodeEncode/UnicodeEncode/UnicodeEncode:0", shape=(None,), dtype=string)
    # 9. Map the characters in label to numbers
    label = char_to_num(label)
    #print(label) Tensor("string_lookup/Identity:0", shape=(None,), dtype=int64)
    # 10. Return a dict as our model is expecting two inputs
    return spectrogram, label
print(encode_single_sample("1", "li be e mih kiann lan lan san san long be tsiau tsng"))



label = tf.strings.lower("li be e mih kiann lan lan san san long be tsiau tsng")
# # 8. Split the label
label = tf.strings.unicode_split(label, input_encoding="UTF-8")

label = char_to_num(label)
print(label)
#num_to_char(label).numpy()#.decode("utf-8")
#label = tf.strings.reduce_join(num_to_char(label)).numpy().decode("utf-8")
#print(label)
print(tf.strings.reduce_join(num_to_char(label)).numpy().decode("utf-8"))

```

將train與validation的資料集根據encode_single_sample進行資料處理、轉換數據後，將格式轉為dataset。
```
batch_size = 8
# Define the trainig dataset
train_dataset = tf.data.Dataset.from_tensor_slices(
    (list(df_train["id"]), list(df_train["text"]))
)

train_dataset = (
    train_dataset.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
    .padded_batch(batch_size)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)
# for x,y in train_dataset:
#   print(y)



# Define the validation dataset
validation_dataset = tf.data.Dataset.from_tensor_slices(
    (list(df_val["id"]), list(df_val["text"]))
)
validation_dataset = (
    validation_dataset.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
    .padded_batch(batch_size)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)


```


印出一個train_dataset的例子確定資料正確。

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
                           min_lr=1e-4)fig = plt.figure(figsize=(8, 5))
for batch in train_dataset.take(1):
  spectrogram = batch[0][0].numpy()
  spectrogram = np.array([np.trim_zeros(x) for x in np.transpose(spectrogram)])
  label = batch[1][0]
  # Spectrogram
  label = tf.strings.reduce_join(num_to_char(label)).numpy().decode("utf-8")
  print(label)
  ax = plt.subplot(2, 1, 1)
  ax.imshow(spectrogram, vmax=1)
  ax.set_title(label)
  ax.axis("off")
  # Wav
  file = tf.io.read_file(wavs_path + list(df_train["id"])[0] + ".wav")
  audio, _ = tf.audio.decode_wav(file)
  audio = audio.numpy()
  ax = plt.subplot(2, 1, 2)
  plt.plot(audio)
  ax.set_title("Signal Wave")
  ax.set_xlim(0, len(audio))
  display.display(display.Audio(np.transpose(audio), rate=16000))
plt.show()
```

設定loss function。
```

def CTCLoss(y_true, y_pred):
  # Compute the training-time loss value
  batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
  input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
  label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

  input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
  label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

  loss = keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
  return loss

```

build_model建立rnn模型。
```
def build_model(input_dim, output_dim, rnn_layers=5, rnn_units=128):
    """Model similar to DeepSpeech2."""
    # Model's input
    input_spectrogram = layers.Input((None, input_dim), name="input")
    # Expand the dimension to use 2D CNN.
    x = layers.Reshape((-1, input_dim, 1), name="expand_dim")(input_spectrogram)
    # Convolution layer 1
    x = layers.Conv2D(
        filters=32,
        kernel_size=[11, 41],
        strides=[2, 2],
        padding="same",
        use_bias=False,
        name="conv_1",
    )(x)
    x = layers.BatchNormalization(name="conv_1_bn")(x)
    x = layers.ReLU(name="conv_1_relu")(x)
    # Convolution layer 2
    x = layers.Conv2D(
        filters=32,
        kernel_size=[11, 21],
        strides=[1, 2],
        padding="same",
        use_bias=False,
        name="conv_2",
    )(x)
    x = layers.BatchNormalization(name="conv_2_bn")(x)
    x = layers.ReLU(name="conv_2_relu")(x)
    # Reshape the resulted volume to feed the RNNs layers
    x = layers.Reshape((-1, x.shape[-2] * x.shape[-1]))(x)
    # RNN layers
    for i in range(1, rnn_layers + 1):
        recurrent = layers.GRU(
            units=rnn_units,
            activation="tanh",
            recurrent_activation="sigmoid",
            use_bias=True,
            return_sequences=True,
            reset_after=True,
            name=f"gru_{i}",
        )
        x = layers.Bidirectional(
            recurrent, name=f"bidirectional_{i}", merge_mode="concat"
        )(x)
        if i < rnn_layers:
            x = layers.Dropout(rate=0.5)(x)
    # Dense layer
    x = layers.Dense(units=rnn_units * 2, name="dense_1")(x)
    x = layers.ReLU(name="dense_1_relu")(x)
    x = layers.Dropout(rate=0.5)(x)
    # Classification layer
    output = layers.Dense(units=output_dim + 1, activation="softmax")(x)
    # Model
    model = keras.Model(input_spectrogram, output, name="DeepSpeech_2")

    # learning_rate = CustomSchedule(
    # init_lr=0.00001,
    # lr_after_warmup=0.001,
    # final_lr=0.00001,
    # warmup_epochs=15,
    # decay_epochs=85,
    # steps_per_epoch=2531//8,
    # )

    # opt = keras.optimizers.Adam(learning_rate)
    
    # Optimizer
    opt = keras.optimizers.Adam(learning_rate=1e-4)
    # Compile the model and return
    model.compile(optimizer=opt, loss=CTCLoss)
    return model


# Get the model
model = build_model(
    input_dim=fft_length // 2 + 1,
    output_dim=char_to_num.vocabulary_size(),
    rnn_units=512,
)
model.summary(line_length=110)
```
![](https://i.imgur.com/6qSduwK.png)


CallbackEval函式: 訓練時的call back fucntion，讓每個epoch跑完時都印出這個epoch對資料集中隨機兩個音檔預測的結果與正確答案與計算WER。

Word error rate (WER) = (D + S + I) / N × 100%
- N - total number of labels （總詞數）
- D - dele2on errors （刪除錯誤）
- S - subs2tu2on errors （替換錯誤）
- I - Inser2on errors （插入錯誤）

decode_batch_predictions 函式:將模型預測結果做decode並使用前面的num_to_char將數值預測結果轉換回字元。
```
# A utility function to decode the output of the network
def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0]
    # Iterate over the results and get back the text
    output_text = []
    for result in results:
        result = tf.strings.reduce_join(num_to_char(result)).numpy().decode("utf-8")
        output_text.append(result)
    return output_text


# A callback class to output a few transcriptions during training
class CallbackEval(keras.callbacks.Callback):
    """Displays a batch of outputs after every epoch."""

    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def on_epoch_end(self, epoch: int, logs=None):
        predictions = []
        targets = []
        for batch in self.dataset:
            X, y = batch
            batch_predictions = model.predict(X)
            batch_predictions = decode_batch_predictions(batch_predictions)
            predictions.extend(batch_predictions)
            for label in y:
                label = (
                    tf.strings.reduce_join(num_to_char(label)).numpy().decode("utf-8")
                )
                targets.append(label)
        wer_score = wer(targets, predictions)
        print("-" * 100)
        print(f"Word Error Rate: {wer_score:.4f}")
        print("-" * 100)
        for i in np.random.randint(0, len(predictions), 2):
            print(f"Target    : {targets[i]}")
            print(f"Prediction: {predictions[i]}")
            print("-" * 100)
```


設置其他訓練時的call back function
early_stop 避免overfitting
checkpoint 每次epoch結束儲存目前訓練的validation loss最佳的一次model(避免程式或gpu用量超過google colab免費版限額時需要全部重train)
reduce_lr 設定learning rate根據validation loss的降低條件
```
os.chdir("/content/drive/MyDrive/nycu/2022ML/rnn")
from keras.callbacks import EarlyStopping ,ModelCheckpoint, ReduceLROnPlateau
#設定early stop
early = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=1, mode='auto')
# 設定模型儲存條件(儲存最佳模型)
# 設定earlystop條件
#early = EarlyStopping(monitor='val_loss', patience=10,mode='min', verbose=1)
checkpoint = ModelCheckpoint('rnn_checkpoint.h5', verbose=1,
                              monitor='val_loss', save_best_only=True,
                              mode='auto')
# 設定lr降低條件(0.001 → 0.005 → 0.0025 → 0.00125 → 下限：0.0001) auto val_acc
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                           patience=5, mode='auto', verbose=1,
                           min_lr=1e-5)


```
根據需要load進之前訓練過程用checkpoint存的model，因為訓練過程需要很多時間，有時程式會當掉或超過colab免費gpu用量限制，重開會遺失之前所有儲存的變數並需要重新train，這種情況的話就可以用load_model的方式把先前所儲存的當前訓練到最好的模型取出來繼續train。
```

from keras.models import load_model

model = load_model('rnn_checkpoint1.h5',custom_objects={'CTCLoss': CTCLoss})

```



model.fit訓練模型，並設置前面設定的所有call back functoin。
```
# Define the number of epochs.
epochs = 50
# Callback function to check transcription on the val set.
validation_callback = CallbackEval(validation_dataset)
# Train the model
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=epochs,
    callbacks=[validation_callback,early,checkpoint,reduce_lr],
)
```
訓練過程圖:
![](https://i.imgur.com/8BFipFq.png)


用訓練好的model預測驗證集結果並計算驗證集Word Error Rate(WER)。
```
# Let's check results on more validation samples
predictions = []
targets = []
for batch in validation_dataset:
    X, y = batch
    batch_predictions = model.predict(X)
    batch_predictions = decode_batch_predictions(batch_predictions)
    predictions.extend(batch_predictions)
    for label in y:
        label = tf.strings.reduce_join(num_to_char(label)).numpy().decode("utf-8")
        targets.append(label)
wer_score = wer(targets, predictions)
print("-" * 100)
print(f"Word Error Rate: {wer_score:.4f}")
print("-" * 100)
for i in np.random.randint(0, len(predictions), 5):
    print(f"Target    : {targets[i]}")
    print(f"Prediction: {predictions[i]}")
    print("-" * 100)
```
驗證集結果如下圖:
![](https://i.imgur.com/nJ5XE7A.png)





使用encode_single_test函式(與encode_single_sample只差在一個針對訓練集資料做處理，一個對測試集資料做處理)讀入測試集音檔，將測試集音檔轉換為我們要的數值格式(char_to_num)，然後再轉成模型需要的dataset格式，最後用model.predict預測每個測試集的結果，並將預測結果存在predictions的list中。
```

def encode_single_test(wav_file, label):
    ###########################################
    ##  Process the Audio
    ##########################################
    # 1. Read wav file:
    #if train == 1:
    #file = tf.io.read_file(wavs_path + wav_file + ".wav")
    file = tf.io.read_file(test_wavs_path + wav_file + ".wav")
    # else:
    #   file = tf.io.read_file(test_wavs_path + wav_file + ".wav")
    # 2. Decode the wav file
    audio, _ = tf.audio.decode_wav(file)
    audio = tf.squeeze(audio, axis=-1)
    # 3. Change type to float
    audio = tf.cast(audio, tf.float32)
    # 4. Get the spectrogram
    spectrogram = tf.signal.stft(
        audio, frame_length=frame_length, frame_step=frame_step, fft_length=fft_length
    )
    # 5. We only need the magnitude, which can be derived by applying tf.abs
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.math.pow(spectrogram, 0.5)
    # 6. normalisation
    means = tf.math.reduce_mean(spectrogram, 1, keepdims=True)
    stddevs = tf.math.reduce_std(spectrogram, 1, keepdims=True)
    spectrogram = (spectrogram - means) / (stddevs + 1e-10)


    # ########PADDING
    # audio_len = tf.shape(spectrogram)[0]
    # # padding to 10 seconds
    # pad_len = 2754*3
    # paddings = tf.constant([[0, pad_len], [0, 0]])
    # spectrogram = tf.pad(spectrogram, paddings, "CONSTANT")[:pad_len, :]
    # ########PADDING
    ###########################################
    ##  Process the label
    ##########################################
    # 7. Convert label to Lower case
    #print(label)
    #if train == 1:
    label = tf.strings.lower(label)
    # 8. Split the label
    label = tf.strings.unicode_split(label, input_encoding="UTF-8")
    #print(label) Tensor("UnicodeSplit/UnicodeEncode/UnicodeEncode/UnicodeEncode:0", shape=(None,), dtype=string)
    # 9. Map the characters in label to numbers
    label = char_to_num(label)
    #print(label) Tensor("string_lookup/Identity:0", shape=(None,), dtype=int64)
    # 10. Return a dict as our model is expecting two inputs
    return spectrogram, label

id_to_text_test = []
for i in range(1,347):
  id_to_text_test.append("")




test_id = []
for i in range(1,346+1):
  test_id.append(str(i))


test_dataset = tf.data.Dataset.from_tensor_slices(   
    (test_id, id_to_text_test)
)

test_dataset = (
    test_dataset.map(encode_single_test, num_parallel_calls=tf.data.AUTOTUNE)
    .padded_batch(batch_size)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)


predictions = []
for batch in test_dataset:
  X, y = batch
  batch_predictions = model.predict(X)
  batch_predictions = decode_batch_predictions(batch_predictions)
  predictions.extend(batch_predictions)




```
將predicitons存的預測結果輸出到csv檔作為submit file。 

```

submit_ans = []
for i in predictions:
  print(i)
  submit_ans.append(i)
submit_ans
#MAE(valid_y,submit_ans)
import csv

# co_id_np = np.array(co_id)
# year_np = np.array(year)

# data = np.array([co_id_np, year_np])

# np.savetxt("sample.csv", data.T, fmt='%s', delimiter='\t')
test_id = list(range(1,346+1))    
#with open('submit.csv', 'w', newline='',encoding='UTF-8-sig') as test_file:
os.chdir("/content/drive/MyDrive/nycu/2022ML/rnn")
# !ls
with open('311581024.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['id', 'text'])
    for x,y in zip (test_id,submit_ans):
        writer.writerow([x,y])
```

將環境所需套件寫入requirements.txt。
```
#!pip install pipreqs
#!pipreqs --force
!pip freeze > requirements.txt
import sys
print(sys.modules.keys())
```




### 結果分析

訓練集的WER最後可達到0.03，但驗證集的WER一直維持於0.04左右，可見得是有些overfitting的狀況。因此我認為可以嘗試再透過適度減少learning rate，或使用kfold的方式來降低over-fitting對模型的影響，另外我認為如果能再增加訓練集的資料數量或套用更適合於台語的資料前處理，也許也有助於達到更好的辨識成果。
另外因為gpu硬體及各種colab環境問題，這次的batch size只能設在8才跑得動，除了訓練時間增加的問題，也不大有調整batch size以達到最終收斂精度上最優的機會，可能因為太小了容易修正方向而導致不收斂，或者需要經過很大的epoch才能收斂，是我覺得比較可惜的部分。
後來經上網查詢這樣的情況，當batch_size因顯卡問題無法提高的時候，可以把solver裡面的iter_size調大一些。因為在每個隨機梯度下降步驟中通過iter_size*batch_size實現累加梯度，所以增加iter_size也可以得到更穩定的梯度。



