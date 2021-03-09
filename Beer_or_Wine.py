import pandas as pd 
import numpy as np
import sklearn
from sklearn import preprocessing
from sklearn.model_selection import cross_validate
from sklearn import neighbors
import matplotlib.pyplot as plt 
from matplotlib import style
style.use('fivethirtyeight')

#transforming values from rgb to hsv and then to wavelength

def rgb_to_hsv(r, g, b):
	max_pixel_value = 255.0
	if r <= max_pixel_value and g <= max_pixel_value and b <= max_pixel_value:
		r = r / max_pixel_value
		g = g / max_pixel_value
		b = b / max_pixel_value

		max_val = max(r, g, b)
		min_val = min(r, g, b)

		v = max_val

		if max_val == 0.0:
			s = 0
			h = 0
		elif max_val - min_val == 0.0:
			s = 0
			h = 0
		else:
			s = (max_val - min_val) / max_val

			if max_val == r:
				h = 60 * ((g - b) / (max_val - min_val)) + 0
			elif max_val == g:
				h = 60 * ((b - r) / (max_val - min_val)) + 120
			else:
				h = 60 * ((r - g) / (max_val - min_val)) + 240
		if h < 0:
			h = h + 360.0

		h = h / 2
		s = s * max_pixel_value
		v = v * max_pixel_value

	return [h, s ,v]


def hue_to_wavelength(h):
	wavelength = 620 - 170 / 270 * h
	return wavelength


#creating the data frame and making necessary adjustments
df = pd.read_csv("t_data2.csv")

df["Beer or Wine"].replace({"Beer": "1", "Wine": "2"}, inplace=True)
df = df.drop(columns=['#'])
red = np.array(df["Red"])
green = np.array(df["Green"])
blue = np.array(df["Blue"])

length = len(red)

wavelengths = []

for i in range(length):
	if (red[i] + green[i] + blue[i]) / 3 > 30: 
		hsv = rgb_to_hsv(red[i], green[i], blue[i])
	h = hsv[0]
	wavelength = hue_to_wavelength(h)
	wavelengths.append(round(wavelength, 1))

df["Wavelength"] = wavelengths

print(df)

#creating the model
data = df[["Wavelength", "Alcohol(%)", "Beer or Wine"]]

x = np.array(data.drop(['Beer or Wine'], 1)) # everything but the class column cause it is the output
y = np.array(data['Beer or Wine'])


x_train, x_test, y_train, y_test= sklearn.model_selection.train_test_split(x, y, test_size=0.2)

clasifier = neighbors.KNeighborsClassifier()
clasifier.fit(x_train, y_train)
 
accuracy = clasifier.score(x_test, y_test)

example = np.asarray([[612, 8]])

prediction = clasifier.predict(example)

print(data.head())
print(accuracy)
print(prediction[0])

rows_count = df.shape[0]

data_dictionary = {"1":[], "2":[]}

x_array = x.tolist()
y_array = y.tolist()

for i in range(rows_count):
	data_dictionary[y_array[i]].append(x_array[i])

point = [612, 8]

for i in data_dictionary:
	for k in data_dictionary[i]:
		if i == "1":
			plt.scatter(k[0], k[1], s=50, color="#E7D11F")
		else:
			plt.scatter(k[0], k[1], s=50, color="#8b0000")
if prediction == "1":
	plt.scatter(point[0], point[1], marker='D', s=100, color="#E7D11F")
elif prediction == "2":
	plt.scatter(point[0], point[1], marker='D', s=100, color="#8b0000")

plt.show()












