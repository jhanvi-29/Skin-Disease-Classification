import json                                
import logging             
import numpy as np
import cv2
from pathlib import Path
import glob
import time
import skimage.io
import tensorflow as tf
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image as kimg
from flask import Flask, request, jsonify,abort
import boto3
import random

app = Flask(__name__)          
app.logger.setLevel(logging.DEBUG)


def model_architecture():
    model = Sequential()
    model.add(Conv2D(32, (5,5), padding='same', activation='relu',input_shape=(200, 200, 3)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(32, (5,5), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (5,5), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (5,5), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(128, (5,5), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(128, (5,5), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(3, activation='softmax'))
    
    model.load_weights("cp-050.pkl")
    
    model.compile(Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

def enhance_contrast(image_matrix, bins=256):
    image_flattened = image_matrix.flatten()
    image_hist = np.zeros(bins)

    # frequency count of each pixel
    for pix in image_matrix:
        image_hist[pix] += 1

    # cummulative sum
    cum_sum = np.cumsum(image_hist)
    norm = (cum_sum - cum_sum.min()) * 255
    # normalization of the pixel values
    n_ = cum_sum.max() - cum_sum.min()
    uniform_norm = norm / n_
    uniform_norm = uniform_norm.astype(np.uint8)

    # flat histogram
    image_eq = uniform_norm[image_flattened]
    # reshaping the flattened matrix to its original shape
    image_eq = np.reshape(a=image_eq, newshape=image_matrix.shape)

    return image_eq

def equalize_this(image_file, with_plot=False, gray_scale=False, bins=256):
    gray_image = cv2.cvtColor(image_file, cv2.COLOR_BGR2GRAY)
    image_eq = enhance_contrast(image_matrix=gray_image)
    cmap_val = 'gray'
    return image_eq

app = Flask(__name__)
my_dict = {"Image":[],"Prediction":[],"Output":[]};
output_list=[]
bucket_list = []
image_list = []

@app.route("/", methods=["GET"])
def test():
	return 'Hello, this API is working !'


@app.route("/im_size", methods=["GET"])

def process_image():
    s3 = boto3.resource("s3")
    my_bucket=s3.Bucket("pretikaimages")
    bucket_list.clear()
    output_list.clear()
    image_list.clear()
    foldername = request.args['folder']
    print("foldername:",foldername)
    for file in my_bucket.objects.filter(Prefix = foldername):
        file_name=file.key
        if file_name.find(".jpg")!=-1:
            bucket_list.append(file.key)
        elif file_name.find(".jpeg")!=-1:
            bucket_list.append(file.key)
        elif file_name.find(".png")!=-1:
            bucket_list.append(file.key)
    length_bucket_list=len(bucket_list)
    print("***********BUCKETLIST********",bucket_list)
    print("-----------Len_BucketList ----------",length_bucket_list)
    count = 0
    
    model = model_architecture()
    
    for i in bucket_list:
        print("i:",i)
        image_list.append(i)
        s3 = boto3.client("s3")
        bucket_name = "pretikaimages"
        file_obj = s3.get_object(Bucket=bucket_name, Key=str(i))
        file_content = file_obj["Body"].read()
        # creating 1D array from bytes data range between[0,255]
        np_array = np.frombuffer(file_content, np.uint8)
        image_np = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        heatmap2 = cv2.applyColorMap(image_np, cv2.COLORMAP_HOT)
        time.sleep(0.5)
        eq_image = equalize_this(image_file=heatmap2, with_plot=False,gray_scale=True)
        cv2.imwrite('eqimg.jpg',eq_image)
        img = kimg.load_img('eqimg.jpg',target_size=(200,200))
        images = kimg.img_to_array(img)
        gray_img=np.expand_dims(images,axis=0)
        predictions = model.predict(gray_img,workers=-1)
        print("Image :",str(i)," - ","ModelPrediction:",predictions)
        predictions = np.argmax(predictions)
        
        my_dict["Output"].append(int(predictions))
        if predictions==0:
            final_pred = "Mild"
        if predictions==1:
            final_pred = "Normal"
        if predictions==2:
            final_pred = "Severe"
        predictions = {'ans':int(predictions),'Class':str(final_pred)}
        my_dict["Image"].append(str(i))
        my_dict["Prediction"].append(str(final_pred))
        count = count + 1
 
    print("==================== TOTAL COUNT =====================", count)
    for j in my_dict["Output"]:
        print("output",j)
        if(j == 0): #mild images
            random_number = random.randint(8, 10)
            output_list.append(random_number)
        if(j == 1): #normal images
            random_number = random.randint(4,7)
            output_list.append(random_number)
        if(j == 2): #severe images
            random_number = random.randint(1,3)
            output_list.append(random_number)
    print("==========OutputList============:",output_list)
    if(length_bucket_list == 0):
        total_score = 0
    else:
        total_score = (sum(output_list))/length_bucket_list
    print(total_score)
    final_scoreList = output_list
    my_dict["Output"].clear()
    my_dict["Image"].clear()
    my_dict["Prediction"].clear()
    print("***************** MY FINAL DICTIONARY ***************",my_dict)
    return jsonify({'Image':image_list,'OutputScore':final_scoreList,'TotalScore':total_score})
    


if __name__ == "__main__":
    #app.run(debug=True)
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
