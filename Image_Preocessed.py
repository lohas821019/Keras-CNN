# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 15:48:58 2019

@author: Chih-Chieh Huang
"""

import os
import cv2
from PIL import Image
import numpy as np


def Sobel():
       
       base = input('請輸入所要進行Sobel的資料夾路徑: >>>> ')    
#      base = 'C:/Users/Chih-Chieh Huang/Desktop/hos_label/'
       final_loc = input('請輸入Sobel完成後所要存的資料夾路徑: >>>> ')
#      final_loc = 'C:/Users/Chih-Chieh Huang/Desktop/hos_label/sobel/'

       folders = os.listdir(base)
       for a in range(len(folders)):
              pic = os.listdir(base + '/' + folders[a])

              if not os.path.isdir(final_loc+'/'+folders[a]):
                     os.mkdir(final_loc+'/'+folders[a])
                     os.chdir(final_loc+'/'+folders[a])
              else:
                     os.chdir(final_loc+'/'+folders[a])

              for i in range(len(pic)):
                     try:
                            ori_img = Image.open(base+'/' +folders[a]+ '/' + pic[i])
                            gray_img = ori_img.convert('L')
                                                       
                            matrix= np.array(gray_img)            
                            x = cv2.Sobel(matrix,cv2.CV_16S,1,0)
                            y = cv2.Sobel(matrix,cv2.CV_16S,0,1)
                            
                            absX = cv2.convertScaleAbs(x)   # 转回uint8
                            absY = cv2.convertScaleAbs(y)      
                            dst = cv2.addWeighted(absX,2.5,absY,2.5,0)
#                           img = Image.fromarray(dst).convert('L')
#                           img.show()
                            
#                           img.save(final_loc+'/'+ pic[i])
#                           img.save(final_loc+'/'+folders[a],pic[i])
#                           time.sleep(0.5)
                            cv2.imwrite(pic[i], dst)

                     except:
                            pass
                                        
       print('執行完畢') 