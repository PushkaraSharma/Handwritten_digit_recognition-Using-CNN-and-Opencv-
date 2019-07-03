# Handwritten_digit_recognition Using CNN and Opencv
This project focuses on recognising string of handwritten digits inputed as an image with the help of deeplearning algorithm of Convolutional Neural Network for training the model and opencv for preprocessing the raw inputted image of digits.
## Install
This project requires Python and the following Python libraries installed:
* NumPy
* Pandas
* Matplotlib
* Keras(deeplearning)
* Opencv
If you do not have Python installed yet, it is highly recommended that you install the python.

## Files Description
* digits_recognition_model.ipynb :- This jupyter notebook file contains code that is used to train the CNN model over the MNIST digits data and save the model for future use.
* preproceesor.py :- This python file contains the functions that are furthur used in digit.py file for doing some preprocessing on image such as resizing pixels,making image sqaure.
* digit.py :- This is the file in which image is being preprocessed(blurring,threshold,canny edge detection and more) using opencv library
and each digit from the image is segmented and feeded to trained model and then prediction is made and this process repeats for all the digits present on the image and the final output is displayed on the console.
* digitsCNN.h5 :- This is the saved model file that can be loaded in any program when ever needed ,so no need ot train the model again.

