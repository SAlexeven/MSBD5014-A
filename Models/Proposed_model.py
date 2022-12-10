# -*- coding: utf-8 -*-

from keras.layers import *
from keras.models import Model
from keras import regularizers
import pickle
import pandas as pd
from keras.callbacks import *
from keras.utils import to_categorical
import numpy as np
from keras import regularizers
from keras import models


class EventPrediction(object):
	def __init__(self,window=3,filters=300,bch=16,epoch=100,train_test_split=0.8):
		self.window=window
		self.filters=filters
		self.batch=bch
		self.epc=epoch
		self.mc=ModelCheckpoint(".../Independent Project/Models/Model saving/checkpoints/"+"weights.{epoch:02d}-loss{loss:.2f}-mean_absolute_percentage_error{mean_absolute_percentage_error:.2f}-.hdf5", monitor='loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=100)
		self.es=EarlyStopping(monitor='loss', min_delta=1, patience=15, verbose=1, mode='auto')
		self.tb=TensorBoard(log_dir='./logs',  batch_size=self.batch, write_graph=True, write_grads=True, write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
		self.data=None
		self.split=train_test_split
		self.end=None
		self.model=None

	def load_data(self):
		# Load Data from tuple
		with open(".../Independent Project/Data/Vec training/20221030_100days/close-lms_vec_training.pkl","rb")as f:
			self.data=pickle.load(f)
		self.end=int(len(self.data['Y'])*self.split)


	def load_model(self):
		longt=Input(shape=(None,300),name='long')
		medt=Input(shape=(None,300),name='medium')
		st=Input(shape=(None,300),name="short")
		std=Dropout(0.4)(st)
		lt1=Conv1D(self.filters, self.window, input_shape=(None,300),strides=1, padding='valid', dilation_rate=1, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(longt)
		lt2=Conv1D(self.filters, 2,input_shape=(None,300), strides=1, padding='valid', dilation_rate=1, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(lt1)
		lt2d=Dropout(0.4)(lt2)
		mt1=Conv1D(self.filters, self.window, input_shape=(None,300),strides=1, padding='valid', dilation_rate=1, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(medt)
		mt2=Conv1D(self.filters, 2,strides=1,input_shape=(None,300), padding='valid', dilation_rate=1, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(mt1)
		mt2d=Dropout(0.4)(mt2)
		ltoutput=GlobalMaxPooling1D()(lt2d)
		mtoutput=GlobalMaxPooling1D()(mt2d)
		stoutput=GlobalAveragePooling1D()(st)
		feature=concatenate([ltoutput,mtoutput,stoutput])
		h1=Dense(50,activation='relu')(feature)
		h2=Dense(10,activation='relu')(h1)
		trend=Dense(1,activation='sigmoid',name='price')(h2)
		self.model=Model(inputs=[longt,medt,st],outputs=[trend])
		self.model.compile(
			optimizer='adam',
			loss='binary_crossentropy',
			#loss='mean_squared_error',
			#metrics=['mean_absolute_percentage_error'])
			metrics=['accuracy'])
		self.model.summary()

	def load_saved_model(self):
		files=os.listdir()
		file_name=[i for i  in files if '.hdf5' in i]
		
		print("Select the saved model to use -")
		for i,j in enumerate(file_name):
			print(str(i+1)+"-"+j)
		choice=input()
		print("#"*5+"  Using saved model-"+file_name[int(choice)-1]+"  "+"#"*5)
		model=models.load_model(os.path.join(os.getcwd(),file_name[int(choice)-1]))
		print("#"*5+"  Model Loaded  "+"#"*5)
		self.model=model
		

	def train(self):
		
		self.model.fit({"long":self.data['X']['long'][:self.end],"medium":self.data['X']['medium'][:self.end],"short":self.data['X']['short'][:self.end]},
		{'price':self.data['Y'][:self.end]},epochs=self.epc,batch_size=self.batch,callbacks=[self.mc,self.tb])


	def test(self,data=None):
		print("#"*5+"  Testing  "+"#"*5)
		if(data==None):
			data=self.data
		loss_metrics=self.model.evaluate({"long":data['X']['long'][self.end:],"medium":data['X']['medium'][self.end:],"short":data['X']['short'][self.end:]},
		{'price':data['Y'][self.end:]},batch_size=1)
		for i,j in enumerate(loss_metrics):
			print(self.model.metrics_names[i],j)

	def get_prediction(self):
		self.end=int(len(self.data['Y']))
		outputs = self.model.predict({"long":self.data['X']['long'][:self.end],"medium":self.data['X']['medium'][:self.end],"short":self.data['X']['short'][:self.end]})
		print(outputs)

predict=EventPrediction()
predict.load_data()
predict.load_model()
#predict.load_saved_model()
predict.train()
predict.test()
predict.get_prediction()