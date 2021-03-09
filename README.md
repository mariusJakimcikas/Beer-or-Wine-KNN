# Beer-or-Wine-KNN
Machine Learning project that classifies a beverage as beer or wine given the color and alcohol percentage using K Nearest Neighbors algorithm

# Goal

Create a simple Machine Learning model that can tell whether a beverage is beer or wine based on the color of it and the alcohol percentage. Since this a classification problem
I decided to use K Nearest Neighbors algorithm to solve it. 

#Data Collection

Data was collected from many different samples of beer and wine. Two variables that are used in the model are color of the beverage (HEX and RGB values of the color)
and the alcohol percentage(%). The color was taken using the app called "Color Name AR" to get both HEX and RGB values of that specific color. 

#Data Manipulation

In order to make the model more accurate, color values had to be converted into wavelenght (nm). In order to do that I converted RGB value of the color to a HSV value. 
Then converted Hue value into wavelength. The converted wavelengths were between 450-620nm.

# Deploying Machine Learning

Using the scikit-learn library in python implimented the K Nearest Neighbors algorithm to the data. 

# Visualization of the results

Using Matplotlib visualized the data set and distinguished the classification with a diamond point.

