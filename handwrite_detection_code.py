import os   #os模組
import random
import numpy as np 
import keras
from PIL import Image   #PIL提供處理image的模組
from keras.models import Sequential   #建立最簡單的線性模組(Sequential),就會一層層往下執行,沒有分叉(If),也沒有迴圈(loop)
                                      #標準一層一層傳遞的神經網路叫 Sequnetial
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D   #CNN的卷基層和池化層
from keras.models import load_model   #載入模組
from keras.utils import np_utils   #用來後續將label標籤轉為one-hot-encoding
from matplotlib import pyplot as plt

#data_x與data_y(label)前處理函式
def data_x_y_preprocess(datapath):
    img_row, img_col = 28, 28   #定義圖片大小
    datapath = datapath   #訓練資料路徑
    data_x = np.zeros((28,28)).reshape(1,28,28)   #儲存圖片
    pictureCount = 0   #紀錄圖片張數
    data_y = []   #紀錄label
    num_class=10   #數字種類有10種
    #讀取image資料夾裡所有檔案
    for root, dirs, files in os.walk(datapath):
        for f in files:
            label=int(root.split("\\")[3])   #取得label
            data_y.append(label)
            fullpath=os.path.join(root,f)   #取得檔案路徑
            img=Image.open(fullpath)   #開啟img
            img =(np.array(img)/225).reshape(1,28,28)   #取得資料時順便做正規畫與reshape
            data_x =np.vstack((data_x, img))
            pictureCount+=1
    data_x =np.delete(data_x,[0],0)   #刪除一開始宣告的np.zeros
    #調整資料格式(圖篇張數,img_row,img_col,色彩通道=1(灰階))
    data_x=data_x.reshape(pictureCount,img_row,img_col,1)
    data_y=np_utils.to_categorical(data_y,num_class)   #將label轉為one-hot encoding
    return data_x,data_y

data_train_X, data_train_Y = data_x_y_preprocess("handwrite__detect\\train_image")      
data_test_X, data_test_Y = data_x_y_preprocess("handwrite__detect\\test_image")

#建立簡單的線性執行的模型
model = Sequential()
#建立卷基層 filter=32, 即 output space深度, Kernal Size: 3x3, activation function採用relu
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(28,28,1)))
#建立池化層,池畫大小=2x2,取最大值
model.add(MaxPooling2D(pool_size=(2,2)))
#建立捲積層,filter=64,即 output size, Kernal Size: 3x3, activation function relu
model.add(Conv2D(64, (3,3), activation='relu'))
#建立池化層,池畫大小=2x2,取最大值
model.add(MaxPooling2D(pool_size=(2,2)))
#Dropout層隨機端開輸入神經元,用於防止過度擬合,斷開比例:0.25
model.add(Dropout(0.25))
#Flatten層大多為的輸入一維化,常用於從卷基層到全連接層的過度
model.add(Flatten())
#Dropout層隨機端開輸入神經元,用於防止過度擬合,斷開比例:0.1
model.add(Dropout(0.1))
#全連接層: 128個 output
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
#使用 softmax activation function, 將結果分類(units=10,表示分10類)
model.add(Dense(units=10, activation='softmax'))

#編譯: 選擇損失函數、優化方法及成校衡量方式
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

#進行訓練,訓練過程會存在train_history變數中
train_history = model.fit(data_train_X, data_train_Y, batch_size =32, epochs=150, verbose=1, validation_split=0.1)
#顯示損失函數,訓練成果(分數)
score = model.evaluate(data_test_X, data_test_Y, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])

#優化過程曲線
plt.plot(train_history.history['loss'])
plt.plot(train_history.history['val_loss'])
plt.title('Train History')
plt.ylabel('loss')
plt.xlabel('Epoch')
plt.legend(['loss, val_loss'], loc='upper left')
plt.show()
