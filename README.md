# Markerless Pose and Behavior Prediction for Rat Models

Here we have provided an open source software pipeline to estimate animal pose and detect a limited set of animal behaviors that are important for biomedical animal studies which include locomotion analysis, prevelence of Rearing, and prevelence of stereotopic behavior.



This project was done by Gouthamrajan Nadarajan, Dowlette Alam El Din, and Zachary Smith and was supervised by [Dr. Anirban Dutta](http://medicine.buffalo.edu/faculty/profile.html?ubit=anirband) and was optimized for animal models from [Dr. Jinwoo Park's](http://medicine.buffalo.edu/faculty/profile.html?ubit=jinwoopa) lab.

## Our Process

In this work we leverage the [Deeplabcut toolbox](http://www.mousemotorlab.org/deeplabcut) which is a software pipeline for Markerless pose estimation using deep learning. The documentation can be found [here](https://github.com/AlexEMG/DeepLabCut) , but we provide an brief tutorial on how we levereged this toolbox for our work in Deeplabcut_marker_pred.ipynb. 


<img align="right" width="250" height="250" src=demo.gif caption='Deeplabcut'>

We trained our Model to learn to detect the features:
- Snout
- Eyes(Left and Right)
- Ears(Left and Right)
- Front and Back paws
- Centroid and Right and Left ends of Centorid
- Hips(Left and Right)
- Base of the tail

Once the model was suffeciently trained it can be applied to different animal data to obtain marker coordinate predictions in .csv format. We leverged these marker predictions on our rat models to derive features for behavior classification.


### Rearing Classification

Rearing is defined as when the animal puts its weight on its hind legs and stand with forelimbs above ground. Using the marker prediction data, we derive the following features below:

<img align="right" width="250" height="250" src=rearing_example.png caption='Deeplabcut'>


- Euclidean distance from the Snout to the Base of the tail
- Euclidean distance from the Snout to the Right and left hip
- Deeplabcut's model confidence in prediciton of the right and left centroid locations

These features were then used to train an support vector machine for the binary classification of rearing vs. non-rearing. We found that we can achieve the best results(94% accuracy) on models that are trained and applied on the same rat model. See Rearing_prediction.ipynb for more details.

### Sterotopy behavior Classification


<img align="right" width="250" height="250" src=stereotopy.gif caption='Deeplabcut'>

Stereotopic behaivor in animal models is defined as behaviors that are repetitive and high frequency and serve no obvious purpose. For stereotopy behavior classification, we utilize our deeplabcut marker predictions to train an LSTM model to classify sequences of one minute time sequences for binary classification of stereotopic behavior vs. non stereotopic behavior. Here we use the coordinate predictions for the snout and the centroid as our Features. We achieved an max testing accuracy of 80% on a holdout testing set.

Refer to Stereotopy_Prediction.ipynb for more details


### Locomotion Analysis

Total location is an important metric for many animal studies. Using the marker prediction from deeplabcut for the centroid marker, we have a pipeline to measure the total distance traveled by the rat in an given experiment.

Refer to Locomotion.ipynb for more details



