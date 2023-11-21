<<<<<<< HEAD

Readme file.


LIVE HOST WEB ADDRESS: https://news-classifier-rk6n.onrender.com	

This the News Article Classifier.

Steps involved in creating this model are - 

EDA - 

reading data using Pandas
droping unecessary features
converting dependent variable to numerics(using label encoder)
data spliting to dependent and independent variables
removing html tags
removing urls
making correction of words
removing punctuations
removing stopwords
making tokenisation using spacy
stemming all the words present in the text
removing the single and numerical words
form the sequences using pad sequences..


Model Creation-

data spliting to dependent and independent variables
SVC,Randomforest, Neural Networks.
Use GridsearchCV to hyper tune the models.
By using accuracy matrix techniques we found that the Random forest is with high accuarcy and the size of the model than the other two.
Use the best model as the final model, train and evaluate the model using classification report or accuracy score or confusion matrix.


Prediction:
using the Random Forest perdict the test data
finding the model performance using the accuracy score, confusion matix and classification report.

Creation of User GUI-

Using the flask library we created the use GUI with HTML and CSS.
Deploy this model in local server.
get the values of the feilds selected by the user by flask
Convert the data using text pipelines and process the same to model.
get the predictions from the model.
return that back to user.

Using the logging

We will write back the logs to the logs.log file

Updating the data to mysql.

from the front end user interface get the values of selected feilds and save them back to local database by python mysql connector, cassandra database.
download the data from the database and share....


=======

