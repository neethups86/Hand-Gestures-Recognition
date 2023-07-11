 
import os
import sys
import xml.etree.ElementTree as ET
import warnings
import keras
warnings.filterwarnings("ignore")
from keras.models import model_from_json
import numpy as np
from PIL import Image
import pandas as pd

if __name__ == "__main__":
    rootDir = "D:\\Neethu\\PHD\\All works\\Work 2 CNN\\CNN\\Delivery_code\\Deep_Learning_Hand_Recognition\\"
        #rootDir = "D:\\Multi_Class_Model\\CNN Finger Gesture Recognition\\Deep_Learning_Hand_Recognition\\"
    Input_Path        =  rootDir + "Input\\"
    tot_number_of_images = len([item for item in os.listdir(Input_Path) if os.path.isfile(os.path.join(Input_Path, item))])
    if tot_number_of_images:  
        lsFile = os.listdir(Input_Path)        
        CNN_Output=list()           
        Input_image_elements =list()
        for j in range(tot_number_of_images):
            img_test       = Image.open(Input_Path+str(lsFile[j]))
            img_test       = img_test.resize((32,32))
            img_test_data  = list(img_test.getdata())
            Input_image_elements.append(img_test_data)
        
        test=Input_image_elements
        num_classes = 7
        img_rows, img_cols = 32, 32
        test_data=list()
        y_test=list()
        for j in range(0,len(Input_image_elements)):
             test_data.append(Input_image_elements[j])
        test_data=np.array(test_data)
        test_data = test_data.astype('float32')
        test_data = test_data / 255.0
        test_data = test_data.reshape(test_data.shape[0], img_rows, img_cols, 3)
        try:            
            jsonModel = "model.json"
            json_file = open(jsonModel, 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            model = model_from_json(loaded_model_json)
        except:
            print("model.json trained model file is missing")
            exit(1)
        try:            
            h5Model = "model.h5"
            model.load_weights(h5Model)
        except:
            print("model.h5 trained weights file is missing")
            exit(1)
        
        predType = list()
        predicted = model.predict(test_data)        
        predTypeList = ['G1','G2','G3','G4','G5','G6','G7']
        for i in range(len(predicted)):
            predType.append(predTypeList[predicted[i].argmax(axis=0)])
                     
        # predicting the labels    
        dfPred    = pd.DataFrame(lsFile,columns=['Img_ImageName'])
        pred      = pd.DataFrame(predType,columns=['Predicted_Label'])
        predType      = pd.DataFrame(predType,columns=['Predicted_Type'])
        
        
        dfPred    = pd.concat([dfPred,pred], axis=1, join='inner')
        dfPred    = pd.concat([dfPred,predType], axis=1, join='inner')
        
        resFile = "Results.csv"
            
        if os.path.isfile(resFile):            
            OldDF = pd.read_csv(resFile,sep=',')
            dfPred = OldDF.append(dfPred,ignore_index=False)
        dfPred.to_csv(resFile , index=False)

        print("Hand Geusture Recognition Completed!")
    else:
        print ("No image data present for prediction!")
