# Convolutional Neural Network

# Part 1 - Building the CNN
import sys
from PIL import Image
import numpy as np
from sklearn.model_selection import GridSearchCV
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dense,Dropout
from keras import optimizers
# from keras.layers.normalization import BatchNormalization
from keras.wrappers.scikit_learn import KerasClassifier

# def  create_model(optimizer ="adam"):
    # Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))
# classifier.add(Dropout(0.5))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
# classifier.add(Dense(units = 256, activation = 'relu'))

classifier.add(Dense(units = 3, activation = 'softmax'))

# Compiling the CNN
# ada =optimizers.Adam(lr=0.0002, beta_1=0.9, beta_2=0.999, decay=0.00001)
# ada =optimizers.Adam(lr=0.0002, beta_1=0.9, beta_2=0.999, decay=0.01)

# for layer in classifier.layers[0:6]:
#     layer.trainable = False
    
classifier.compile(optimizer = "adam",loss = 'sparse_categorical_crossentropy', metrics=['accuracy'] )
# classifier.compile(optimizer = optimizer,loss = 'sparse_categorical_crossentropy', metrics=['accuracy'] )

classifier.summary()
    
    # return classifier
        
#%%

# model = KerasClassifier(build_fn=create_model, nb_epoch=100, batch_size=10, verbose=0)
# # define the grid search parameters
# optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
# param_grid = dict(optimizer=optimizer)
# grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
# grid_result = grid.fit(training_set,test_set)


# # summarize results
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
#     print("%f (%f) with: %r" % (mean, stdev, param))

#%%
from keras.utils import plot_model
plot_model(classifier, to_file='MODEL.png',show_shapes=True)


# 將已訓練model參數載入後，就會基於這參數再去訓練新的
# classifier.load_weights('weights-best3.hdf5')


# #classifier.reset_states() 重置model所有層狀態
# from sklearn.cluster import KMeans
# # 原文網址：https://kknews.cc/news/blr5ljm.html
# n_clusters = 4

# # Runs in parallel 4 CPUs
# kmeans = KMeans(n_clusters=n_clusters, n_init=20, n_jobs=4)
# # Train K-Means.
# y_pred_kmeans = kmeans.fit_predict(training_set)
# # Evaluate the K-Means clustering accuracy.
# metrics.acc(y, y_pred_kmeans)


#%%
# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True,
                                   vertical_flip= True,
                                   rotation_range = 360,
                                    # samplewise_std_normalization = True,
                                   validation_split=0.2)

# test_datagen = ImageDataGenerator(rescale = 1./255,
#                                   samplewise_std_normalization = True,)

training_set = train_datagen.flow_from_directory(r'C:\Users\Jason\Desktop\CNN\CNN\dataset\training_set\Cervical Cancer_stage2-4',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary',)
                                                 # subset='training')


validation_set = train_datagen.flow_from_directory(r'C:\Users\Jason\Desktop\CNN\CNN\dataset\training_set\Cervical Cancer_stage2-4',
                                            target_size = (64, 64),
                                            batch_size = 25,
                                            class_mode = 'binary'
                                            ,subset='validation')

# 一個 epoch 要跑完全部的訓練資料，所以若訓練資料有 n 筆，batch size 為 m ，
# 那麼每一個 epoch 應該要跑 n/m 次
# ，亦即 steps_per_epoch*batch_size= len(training data)，validation data 亦然。


from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=50, verbose=2,mode='auto',min_delta=0.01)#0.005

# min_delta 評斷監控的數據是否有改善標準，唯有當數據變動幅度大於 min_delta 才算是有改善。
from keras.callbacks import ModelCheckpoint
filepath="weights.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True,)
                             
history = classifier.fit_generator(training_set,
                         steps_per_epoch = 20,
                         epochs = 300,
                         validation_data = validation_set,
                         validation_steps = validation_set.samples,
                         callbacks=[early_stopping,checkpoint])
                         # validation_steps = 5)


   # train是訓練集，val是訓練過程中的測試集，是為了讓你在邊訓練邊看到訓練的結果，及時判斷學習狀態。
   # test就是訓練模型結束後，用於評價模型結果的測試集。只有train就可以訓練，val不是必須的，比例也可以設置很小。
   # test對於model訓練也不是必須的，但是一般都要預留一些用來檢測，通常推薦比例是8:1:1
   # https://blog.csdn.net/qq_37995260/article/details/100146401



#%%
from matplotlib import pyplot as plt

def train_trend(history):
       plt.figure()
       plt.plot(history.history['acc'])
       plt.plot(history.history['val_acc'])
       plt.title('Model accuracy')
       plt.ylabel('Accuracy')
       plt.xlabel('Epoch')
       plt.legend(['Train', 'Test'], loc='upper left')
       plt.show()
       
       plt.figure()
       plt.plot(history.history['loss'])
       plt.plot(history.history['val_loss'])
       plt.title('Model loss')
       plt.ylabel('loss')
       plt.xlabel('Epoch')
       plt.legend(['Train', 'Test'], loc='upper left')
       
train_trend(history)

# 當train loss不斷下降，test loss也不斷下降：網絡仍在學習，讓他繼續學;

# train loss不斷下降，test loss趨於不變：說明網絡過擬合;

# train loss趨於不變，test loss不斷下降：不用想數據集250%有問題;

# train loss趨於不變，test loss趨於不變：說明學習遇到瓶頸，需要減小學習率或批量數目;或者是數據集有問題（數據集標註錯誤數據比較多）

# train loss不斷上升，test loss不斷上升：說明網絡結構設計不當，訓練超參數設置不當，數據集經過清洗等問題。

#%%
from sklearn.metrics import classification_report, confusion_matrix
Y_pred = classifier.predict_generator(validation_set, validation_set.samples)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
print(confusion_matrix(validation_set.classes, y_pred))
print('Classification Report')
target_names = list(training_set.class_indices.keys())
print(classification_report(validation_set.classes, y_pred, target_names=target_names))

#%% 驗證val之表格
import numpy as np
prediction=classifier.predict_generator(training_set,verbose=1)
predict_label=np.argmax(prediction,axis=1)

true_label=training_set.classes

import pandas as pd
pd.crosstab(true_label,predict_label,rownames=['label'],colnames=['predict'])
#https://www.itread01.com/content/1544370907.html


#%%
from keras.preprocessing import image

def pred(img):
       
       global prediction
       global result 
       input_shape = (64,64,3)

       test_image = image.load_img(img, target_size = input_shape)
       test_image = image.img_to_array(test_image)
       test_image /= 255.
       test_image = np.expand_dims(test_image, axis = 0)
#       test_image /= 255.
       
       plt.figure() 
       plt.imshow(test_image[0])
       print(test_image)
       
       result = classifier.predict(test_image)
       classes = training_set.class_indices
       print(classes)
       
       if result[0][0] >= result[0][1]:
              prediction = 'carcinnoma'
              plt.title(result[0][0])
       else   :    
              prediction = 'negative'
              plt.title(result[0][1])
#       if result[0][2] == 1:
#              prediction = '7'
              
       plt.xlabel(str(prediction))
       return print('這張圖屬於 '+str(prediction))

#%%      
pred(img ='350-428,1201-1265.png')
train_trend(history)

#%%
import os 

path = ''
os.chdir(path)
img = os.listdir(path)

for i in range(len(img)):
    pred(img = img[i])


#%%
# from keras.models import load_model
# classifier = load_model('weights-best170-2(目前最好的).hdf5') 
 # classifier = load_model('weights-best170-2.hdf5') 

# classifier.save('cat_and_dog.hdf5')  

#%%
import numpy as np
from matplotlib import pyplot as plt
from keras.models import load_model
from keras.preprocessing import image
import os 
input_shape = (64,64,3)


def scra(path_pred):   
        import os    
        import numpy as np
        allf = []
        global HSIL 
        HSIL = []
        global LSIL 
        LSIL =[]
        global Normal 
        Normal=[]
       
        files = os.listdir(path_pred)
##       global color
#       path_pred = path_pred
       # files = os.listdir(r'C:\Users\Jason\Desktop\CNN\CNN\dataset\training_set\Cervical Cancer_stage2-2\HSIL_細切後')
       
        for num in range(np.size(files)):  
               img = path_pred+'/'+files[num]
              
       #       img = 'C:/Users/Chih-Chieh Huang/Desktop/hos_sobel/'+files[num]
              
               test_image = image.load_img(img, target_size = input_shape)
               test_image = image.img_to_array(test_image)
               test_image /= 255. # normalize
               test_image = np.expand_dims(test_image, axis = 0)     
               classes = training_set.class_indices
              # print(classes)
              
               result = classifier.predict(test_image)


       #       plt.scatter(result[0][0],result[0][1],c='b')
              
#              abnormal.append(result[0][0])
#              normal.append(result[0][1])
              
               if result[0][0] > result[0][1] and result[0][0] > result[0][2]:             
                    prediction = 'HSIL'
                    pred_acc = result[0][0]
                    color ='red'
                    HSIL.append(result[0][0])
               elif result[0][1] > result[0][0] and result[0][1] > result[0][2]:      
                    prediction = 'LSIL'
                    pred_acc = result[0][1]
                    color ='blue'
                    LSIL.append(result[0][1])
               elif result[0][2] > result[0][0] and result[0][2] > result[0][1]:    
                    prediction = 'Normal'
                    pred_acc = result[0][2]
                    color ='green'
                    Normal.append(result[0][2])
                     
               allf.append(files[num]+' '+str(prediction) +'  '+ str(pred_acc))       
               print('檔案名稱 : '+files[num]+'  //  '+str(prediction) +'判斷率 '+ str(pred_acc))
               print(result)
               print('---------------------------------------------------------')

        plt.scatter(range(np.size(HSIL)),HSIL,c='red')  
        plt.scatter(range(np.size(LSIL)),LSIL,c='blue')   
        plt.scatter(range(np.size(Normal)),Normal,c='green')   
#       plt.scatter(range(np.size(files)),result_cell,c='red')  
       # plt.axhline(0.5, color= 'black')
       
       # for i in range(np.size(files)):
       #        plt.figure()
              
#       
#       plt.scatter(result_cell,1-result_cell)
#       plt.axvline(0.5, color= 'green')
#       plt.xlabel('')
#       
#       result_total = [result_cell,result_notcell]
#       result_total = np.array(result_total)           

#%%
        
        labels = ['HSIL', 'LSIL', 'Normal']
        # HSIL_len = [len(HSIL),len(LSIL),len(Normal)]
        # LSIL_len = [len(HSIL),len(LSIL),len(Normal)]
        # Normal_len =[len(HSIL),len(LSIL),len(Normal)]
        
        pic1=[HSIL_len[0],LSIL_len[0], Normal_len[0]]
        pic2=[HSIL_len[1],LSIL_len[1], Normal_len[1]]
        pic3=[HSIL_len[2],LSIL_len[2], Normal_len[2]]
        
        x = np.arange(len(labels))  # the label locations
        width = 0.3  # the width of the bars
        
        fig, ax = plt.subplots()
        rects1 = ax.bar(x - width, pic1, width, label='HSIL',color='r')
        rects2 = ax.bar(x , pic2, width, label='LSIL',color='b')
        rects3 = ax.bar(x +width, pic3, width, label='Normal',color='g')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Numbers')
        ax.set_title('Predict result')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel('Three types of Cervical Cancer ')
        ax.legend()
        
        
        def autolabel(rects):
            """Attach a text label above each bar in *rects*, displaying its height."""
            for rect in rects:
                height = rect.get_height()
                ax.annotate('{}'.format(height),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')
        
        autolabel(rects1)
        autolabel(rects2)
        autolabel(rects3)
        
        fig.tight_layout()
        
        plt.show()
        
        


#%%
p1= r'C:\Users\Jason\Desktop\CNN\CNN\dataset\training_set\Cervical Cancer_stage2-4\HSIL_170x170'
p2= r'C:\Users\Jason\Desktop\CNN\CNN\dataset\training_set\Cervical Cancer_stage2-2\lSIL_細切後'
p3= r'C:\Users\Jason\Desktop\CNN\CNN\dataset\training_set\Cervical Cancer_stage2-4\Normal'

plt.figure()
scra(path_pred = p1)        
plt.figure()
scra(path_pred = p2) 
plt.figure()
scra(path_pred = p3) 
plt.xlabel('Numbers')    
plt.ylabel('Accuracy')
plt.title('Compare the Correct rate about cell')  
#%%
              
# p1= r'C:\Users\Jason\Desktop\CNN\CNN\dataset\training_set\Cervical Cancer_stage2-2\HSIL_細切後'
# p2= r'C:\Users\Jason\Desktop\CNN\CNN\dataset\training_set\Cervical Cancer_stage2-2\lSIL_細切後'
# p3= r'C:\Users\Jason\Desktop\CNN\CNN\dataset\training_set\Cervical Cancer_stage2-2\Normal'

# plt.figure()
# scra(path_pred=p1,c='red')  
# plt.figure()
# plt.scatter(path_pred=p2,c='blue')   
# plt.figure()
# plt.scatter(path_pred=p3,c='green')   
# #       plt.scatter(range(np.size(files)),result_cell,c='red')  

      
              
             #%%
             
# plt.figure()
# scratter(path_pred = r'C:\Users\Chih-Chieh Huang\Desktop\hos_sobel\org\normal',color = 'blue')        
# scratter(path_pred = r'C:\Users\Chih-Chieh Huang\Desktop\hos_sobel\org\abnormal',color = 'red') 
# plt.xlabel('Numbers')    
# plt.ylabel('Accuracy')
# plt.title('Compare the Correct rate about cell')  



