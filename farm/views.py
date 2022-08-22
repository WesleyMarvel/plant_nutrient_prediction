from django.shortcuts import render
import serial
import time
from .models import PHL
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow.compat.v2.feature_column as fc
import tensorflow as tf
from csv import writer

# Create your views here.



def index(request):

	
	


	while True:

		if request.method == 'GET':

			return render(request, 'farm/index.html')


		
		try:
			if request.method == 'POST':

				ser = serial.Serial("/dev/cu.usbmodemFD121", 9600)

				ser.timeout = 1

				rea = ser.readline().decode('ascii')

				new_rea = rea.replace("\r\n", "")

				crop_name = request.POST['crop_name']

				crop_cat = request.POST['crop_cat']

				nitr = "none"

				phos = "none"

				kota = "none"

				rec = [crop_name, crop_cat, nitr, phos, kota, new_rea]

				with open('/Users/user/Downloads/ncs_test.csv', 'a') as f_object:

					writer_object = writer(f_object)

					writer_object.writerow(rec)

					f_object.close()

				dftrain = pd.read_csv("/Users/user/Downloads/ncs_train.csv")
				dftest = pd.read_csv("/Users/user/Downloads/ncs_test.csv")
				y_train = dftrain.pop("N")
				y_test = dftest.pop("N")

				bftrain = pd.read_csv("/Users/user/Downloads/ncs_train.csv")
				bftest = pd.read_csv("/Users/user/Downloads/ncs_test.csv")
				z_train = bftrain.pop("K")
				z_test = bftest.pop("K")

				cftrain = pd.read_csv("/Users/user/Downloads/ncs_train.csv")
				cftest = pd.read_csv("/Users/user/Downloads/ncs_test.csv")
				x_train = cftrain.pop("P")
				x_test = cftest.pop("P")

				CATEGORICAL_COLUMNS = ['Crop','CropCategory']
				NUMERIC_COLUMNS = ['Soil Ph level']

				feature_columns = []
				for feature_name in CATEGORICAL_COLUMNS:
					vocabulary = dftrain[feature_name].unique()
					feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

				for feature_name in NUMERIC_COLUMNS:
					feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

				def make_input_fn(data_df, label_df, num_epochs=3, shuffle=True, batch_size=22):
					def input_function():
						ds = tf.data.Dataset.from_tensor_slices((dict(data_df),label_df))
						if shuffle:
							ds = ds.shuffle(100)

						ds = ds.batch(batch_size).repeat(num_epochs)
						return ds
					return input_function
				train_input_fn = make_input_fn(dftrain, y_train)
				eval_input_fn = make_input_fn(dftest, y_test, num_epochs=1, shuffle=False)

				train_input_fn1 = make_input_fn(cftrain, x_train)
				eval_input_fn1 = make_input_fn(cftest, x_test, num_epochs=1, shuffle=False)

				train_input_fn2 = make_input_fn(cftrain, z_train)
				eval_input_fn2 = make_input_fn(cftest, z_test, num_epochs=1, shuffle=False)

				linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns, n_classes=174)
				linear_est.train(train_input_fn)

				linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns, n_classes=174)
				linear_est.train(train_input_fn1)

				linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns, n_classes=174)
				linear_est.train(train_input_fn2)

				result = list(linear_est.predict(eval_input_fn))

				result1 = list(linear_est.predict(eval_input_fn1))

				result2 = list(linear_est.predict(eval_input_fn2))

				nitro = (result[0]['probabilities'][5])*10
				phos = (result2[0]['probabilities'][4])*10
				potassium = (result1[0]['probabilities'][1])*10


				ph_level = PHL(ph_level=new_rea)

				ph_level.save()


				ph_levels = PHL.objects.all()

				ph = request.POST['ph']

				if ph == "on":

					ne = 'on'.strip()

					ser.write(ne.encode())

					time.sleep(0.5)

				elif ph == "off":

					ne = 'off'.strip()

					ser.write(ne.encode())

					time.sleep(0.5)

				#co = ph.strip()

				#ser.write(co.encode())


				

			ser.close()

			print (ph_levels)
			return render(request, 'farm/index.html', {"rea": rea, "ph_levels": ph_levels, "nitro": nitro, "phos": phos, "potassium": potassium})

			

		except UnicodeDecodeError:

			if request.method == 'POST':

				ser = serial.Serial("/dev/cu.usbmodemFD121", 9600)

				ser.timeout = 1

				rea = ser.readline().decode('ascii')

				new_rea = rea.replace("\r\n", "")

				crop_name = request.POST['crop_name']

				crop_cat = request.POST['crop_cat']

				nitr = "none"

				phos = "none"

				kota = "none"

				rec = [crop_name, crop_cat, nitr, phos, kota, new_rea]

				with open('/Users/user/Downloads/ncs_test.csv', 'a') as f_object:

					writer_object = writer(f_object)

					writer_object.writerow(rec)

					f_object.close()

				dftrain = pd.read_csv("/Users/user/Downloads/ncs_train.csv")
				dftest = pd.read_csv("/Users/user/Downloads/ncs_test.csv")
				y_train = dftrain.pop("N")
				y_test = dftest.pop("N")

				bftrain = pd.read_csv("/Users/user/Downloads/ncs_train.csv")
				bftest = pd.read_csv("/Users/user/Downloads/ncs_test.csv")
				z_train = bftrain.pop("K")
				z_test = bftest.pop("K")

				cftrain = pd.read_csv("/Users/user/Downloads/ncs_train.csv")
				cftest = pd.read_csv("/Users/user/Downloads/ncs_test.csv")
				x_train = cftrain.pop("P")
				x_test = cftest.pop("P")

				CATEGORICAL_COLUMNS = ['Crop','CropCategory']
				NUMERIC_COLUMNS = ['Soil Ph level']

				feature_columns = []
				for feature_name in CATEGORICAL_COLUMNS:
					vocabulary = dftrain[feature_name].unique()
					feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

				for feature_name in NUMERIC_COLUMNS:
					feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

				def make_input_fn(data_df, label_df, num_epochs=3, shuffle=True, batch_size=22):
					def input_function():
						ds = tf.data.Dataset.from_tensor_slices((dict(data_df),label_df))
						if shuffle:
							ds = ds.shuffle(100)

						ds = ds.batch(batch_size).repeat(num_epochs)
						return ds
					return input_function
				train_input_fn = make_input_fn(dftrain, y_train)
				eval_input_fn = make_input_fn(dftest, y_test, num_epochs=1, shuffle=False)

				train_input_fn1 = make_input_fn(cftrain, x_train)
				eval_input_fn1 = make_input_fn(cftest, x_test, num_epochs=1, shuffle=False)

				train_input_fn2 = make_input_fn(cftrain, z_train)
				eval_input_fn2 = make_input_fn(cftest, z_test, num_epochs=1, shuffle=False)

				linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns, n_classes=174)
				linear_est.train(train_input_fn)

				linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns, n_classes=174)
				linear_est.train(train_input_fn1)

				linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns, n_classes=174)
				linear_est.train(train_input_fn2)

				result = list(linear_est.predict(eval_input_fn))

				result1 = list(linear_est.predict(eval_input_fn1))

				result2 = list(linear_est.predict(eval_input_fn2))

				nitro = (result[0]['probabilities'][5])*10
				phos = (result2[0]['probabilities'][4])*10
				potassium = (result1[0]['probabilities'][1])*10


				ph_level = PHL(ph_level=new_rea)

				ph_level.save()


				ph_levels = PHL.objects.all()

				ph = request.POST['ph']

				if ph == "on":

					ne = 'on'.strip()

					ser.write(ne.encode())

					time.sleep(0.5)

				elif ph == "off":

					ne = 'off'.strip()

					ser.write(ne.encode())

					time.sleep(0.5)

				#co = ph.strip()

				#ser.write(co.encode())


				

			ser.close()

			print (ph_levels)
			return render(request, 'farm/index.html', {"rea": rea, "ph_levels": ph_levels, "nitro": nitro, "phos": phos, "potassium": potassium})



