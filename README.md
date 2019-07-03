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
## IDE USED -  
* pycharm(for preprocessing image and predictions).
* jupyter notebook(for developing  model).
## Dataset Used
The MNIST database contains 60,000 training images and 10,000 testing images. Half of the training set and half of the test set were taken from NIST's training dataset, while the other half of the training set and the other half of the test set were taken from NIST's testing dataset.
![mnist](https://user-images.githubusercontent.com/46081301/60577821-bdb9af80-9d9d-11e9-9a7c-4cd3099a316a.png)

## Files Description
* digits_recognition_model.ipynb :- This jupyter notebook file contains code that is used to train the CNN model over the MNIST digits data and save the model for future use.
* preproceesor.py :- This python file contains the functions that are furthur used in digit.py file for doing some preprocessing on image such as resizing pixels,making image sqaure.
* digit.py :- This is the file in which image is being preprocessed(blurring,threshold,canny edge detection and more) using opencv library
and each digit from the image is segmented and feeded to trained model and then prediction is made and this process repeats for all the digits present on the image and the final output is displayed on the console.
* digitsCNN.h5 :- This is the saved model file that can be loaded in any program when ever needed ,so no need ot train the model again.

## Result
* batch size :- 32
* Accuracy :- 0.94
* Loss :- 0.044138
#### Actual Image
![Capture](https://user-images.githubusercontent.com/46081301/60577825-be524600-9d9d-11e9-8f91-5af5be32fa86.PNG)
#### Blurred Image
![blurred](https://user-images.githubusercontent.com/46081301/60577822-bdb9af80-9d9d-11e9-9499-7825a88a53ff.PNG)
#### After Threshold
![threshold](https://user-images.githubusercontent.com/46081301/60577827-beeadc80-9d9d-11e9-8a72-699cc44b7746.PNG)
#### Canny Edge detection
![canny](https://user-images.githubusercontent.com/46081301/60577824-be524600-9d9d-11e9-840c-8bb6811e1fe6.PNG)
#### Contours
![boundry](https://user-images.githubusercontent.com/46081301/60577823-bdb9af80-9d9d-11e9-888a-1a2a2984019d.PNG)
#### Final Prediction
![predicted](https://user-images.githubusercontent.com/46081301/60577826-beeadc80-9d9d-11e9-9f44-a910ebe315fb.PNG)
 
 * Actual number :- 504192
 * Predicted number :- 504182
 
## HOW TO RUN:-
       * As the model is being trained already so there is no need to run digits_recognition_model.ipynb file 
       * Just run digitsCNN.py file and upload the required image
       * Install python 3.7
       * pip install all other required libraries  

## Future Work:-
* Right now this model works correctly only with one line string , in future it will read entire page.
* Alphabets recognition will also be added in order to increase its usage.






