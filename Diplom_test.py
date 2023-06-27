import numpy as np  # streamlit run Diplom_test.py  or  streamlit run c:\Users\Professional\anaconda3\Diplom\Diplom_test.py
import PIL
from PIL import Image, ImageOps
import tensorflow as tf
from tensorflow import keras
import keras.api._v2.keras as keras
from keras_preprocessing.image import ImageDataGenerator, img_to_array
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt
import streamlit as st
import os
import pandas as pd
from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.models import load_model
from keras.preprocessing import image
from glob import glob
import zipfile
import tempfile
import graphviz
import pydotplus

st.set_page_config(
    layout="wide",
    page_title="Оценка эффективности алгоритмов",
    page_icon=":mortar_board:",
    initial_sidebar_state="expanded",
)

html_temp = '''
    <div style='background-color:#fab75f ;padding:18px'>
    <h2 style='color:white;text-align:center;'><big><big>Оценка эффективности алгоритмов распознавания объектов на изображениях<big><big>
    </h2>
    </div>
    '''
st.markdown(html_temp, unsafe_allow_html=True)
st.title('_Ввод модели и данных для проверки алгоритма_')
col1, col2 = st.columns([5, 2])
col_1, col_2, col_3, col_4, col_5 = st.columns([15, 15, 5, 1, 5])

def run_model(myzipfile1, myzipfile2, X_size, Y_size):
    with tempfile.TemporaryDirectory() as tmp_dir:
        but_bool[1]=True
        pass1 = tmp_dir+"/cat"
        pass2 = tmp_dir+"/dog"
        os.mkdir(pass1)
        os.mkdir(pass2)
        myzipfile1.extractall(pass1)
        myzipfile2.extractall(pass2)
        img_size=(X_size,Y_size)
        images = sum(len(files) for root, dirs, files in os.walk(tmp_dir))
        for root, dirs, files in os.walk(pass1): 
            for file in files: 
                image = Image.open(pass1+'/'+file)
                file = ImageOps.fit(image, img_size, Image.BICUBIC)
        for root, dirs, files in os.walk(pass2): 
            for file in files: 
                image = Image.open(pass1+'/'+file)
                file = ImageOps.fit(image, img_size, Image.BICUBIC)  
        datagen = ImageDataGenerator(rescale=1./255)
        img_generator=datagen.flow_from_directory(tmp_dir,
                                                target_size=img_size,
                                                batch_size=images,
                                                class_mode='binary')
        imgs,labels=img_generator.next()
        array_imgs=np.transpose(np.asarray([img_to_array(img) for img in imgs]),(0,2,1,3))
        predictions=model.predict(imgs)
        rounded_prediction=np.around(predictions, decimals=0)
        #model.summary()
        plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
        with open("model_plot.png", "rb") as datafile:
            st.download_button(
                label="Загрузка архитектуры",
                data=datafile,
                file_name='Model.png',
               mime='image/png',
            )
        keras.utils.plot_model(model, show_dtype=True, show_layer_names=True, 
                               show_shapes=True, to_file='model.png')
        st.image(Image.open('model.png'), caption='Архитектура модели')
        table = pd.crosstab(labels,rounded_prediction.flatten())
        st.dataframe(table, use_container_width=False)
        st.write('Точность модели = {} %'.format((table.iloc[0][0]+table.iloc[1][1])/images * 100))
        frr = table.iloc[0][1]/images
        far = table.iloc[1][0]/images
        st.write('FRR = {:.3%}'.format(frr))
        st.write('FAR = {:.3%}'.format(far))

        #FPR=[]
        #TPR=[]
        #table = pd.crosstab(labels,rounded_prediction.flatten())
        #for i in range(0, 10000, 500):
        #    ROC_prediction = np.around(predictions, decimals=15)
        #    for j in ROC_prediction:
        #        if j < (i/10000.0): 
        #            j=0.0
        #        else: 
        #            j=1.0
        #    ROC_prediction = np.around(ROC_prediction, decimals=0)
        #    ROC_table = pd.crosstab(labels,ROC_prediction.flatten())
        #    #st.dataframe(ROC_table, use_container_width=False)
        #    FPR.append(ROC_table.iloc[1][0]/(ROC_table.iloc[1][0]+ROC_table.iloc[1][1]))   
        #    TPR.append(ROC_table.iloc[0][0]/(ROC_table.iloc[0][0]+ROC_table.iloc[0][1]))
        #chart_data = pd.DataFrame(
        #    (np.vstack([FPR, TPR])))
        #st.text(np.vstack([FPR, TPR]))
        #st.line_chart(chart_data)#, x='FPR', y='TPR'
        #st.write(predictions)
        
                        
file_1 = None
file_2 = None
but_bool = [True,False]
if "disabled" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False
with col_3:
    X_size = st.number_input('number X', min_value=20, max_value=3500, value=150, format="%d", 
                             disabled=st.session_state.disabled, label_visibility="collapsed")
with col_4:
    st.markdown('**x**')
with col_5:
    Y_size = st.number_input('number Y', min_value=20, max_value=3500, value=150, format="%d", 
                             disabled=st.session_state.disabled, label_visibility="collapsed")
with col2:
    st.markdown('Введите размеры изображения (по умолчанию **_150x150_**)')
    st.checkbox("Сохранение введенных размеров", key="disabled")

with col1:
    stream = st.file_uploader('Пожалуйста, загрузите модель в формате *.h5*.zip', type='zip')
    if stream is not None:
        myzipfile = zipfile.ZipFile(stream)
        with tempfile.TemporaryDirectory() as tmp_dir:
            myzipfile.extractall(tmp_dir)
            root_folder = myzipfile.namelist()[0] # "model.h5py"
            model_dir = os.path.join(tmp_dir, root_folder)
            model = tf.keras.models.load_model(model_dir)

if stream is None:
    col1.text('Пожалуйста, загрузите модель в формате *.h5')
else:
    file_1 = col_1.file_uploader('Пожалуйста, загрузите архив изображений первого класса в формате *.zip', 
                                 type='zip', disabled=not st.session_state.disabled)
    file_2 = col_2.file_uploader('Пожалуйста, загрузите архив изображений второго класса в формате *.zip', 
                                 type='zip', disabled=not st.session_state.disabled)
    if (file_1 is not None and file_2 is not None):
        myzipfile1 = zipfile.ZipFile(file_1)
        ziperror = myzipfile1.testzip()
        if ziperror is not None: 
            col_1.text(f'{ziperror} is fail.')
        else:
            myzipfile2 = zipfile.ZipFile(file_2)
            ziperror = myzipfile2.testzip()
            if ziperror is not None: 
                col_2.text(f'{ziperror} is fail.')
            else:
                but_bool[0]=False
        col_1.button('Запустить модель', disabled= (but_bool[0] or but_bool[1]),on_click=run_model,args=(myzipfile1,myzipfile2,X_size,Y_size))
    else:
        if file_1 is None:
            myzipfile1 = None
        if file_2 is None:
            myzipfile2 = None
        st.text('Пожалуйста, загрузите изображения в формате *.jpg/ *.jpeg/ *.pdf/ *.png в *.zip архиве')
    

st.set_option('deprecation.showfileUploaderEncoding', False)






#if (file_1 is not None and file_2 is not None):
#        myzipfile1 = zipfile.ZipFile(file_1)
#        with myzipfile1 as zf:
#            ziperror = zf.testzip()
#            if ziperror is not None: 
#                col_1.text(f'{ziperror} is fail.')
#            else:
#                myzipfile2 = zipfile.ZipFile(file_2)
#                with myzipfile2 as zf:
#                    ziperror = zf.testzip()
#                if ziperror is not None: 
#                    col_2.text(f'{ziperror} is fail.')
#                else:
#                    but_bool[0]=False
