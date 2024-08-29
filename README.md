
# GENDER CLASSIFICATION REPORT

# PROBLEM STATEMENT
Developing a machine learning model that can accurately classify an individual's gender based on their physical characteristics, behavior, and/or demographic information, with the goal of achieving a minimum accuracy of 90% on a diverse and representative dataset, while also identifying and mitigating potential biases in the model's predictions.

Key Components of the Dataset:

- Physical Characteristics: (long_hair, forehead_width_cm, forehead_height_cm, nose_wide,nose_long, lips_thin, distance_nose_to_lip_long, gender).
- Target Variable: Gender (Classification of Male and Female).

Deliverables:
- A trained machine learning model that meets the performance metric
- Report detailing the methodology and results.
- Using Estimator and Prediction to display the results

# LIBRARY USED
- Numpy
- Pandas
- Scikit Learn Libraries

# STEPS
Loading our Data (Gender Classification Dataset, a CSV format type)
- Gender=pd.read_csv("C:/Users/HELLO/Desktop/Prediction/gender_classification_v7.csv")
- Gender
Replacing our Gender column with 0 and 1
- Gender.replace("Male",1,inplace=True)
- Gender.replace("Female",0,inplace=True)
- Gender
Checking our Data Type
- Gender.info()

![Screenshot (2)](https://github.com/user-attachments/assets/1fecfdc2-c783-4709-b2e7-cf853860d522)

#### Splitting our Data into X and Y (Features and Target)
- X=Gender.drop(columns="gender")
- Y=Gender["gender"]
Splitting our Data into Test and Train for Splitting
- from sklearn.model_selection import train_test_split
- X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)
Importing our Model for training (Using Support Vector Classifier Model)
- from sklearn.svm import SVC
- SVM_Model=SVC()
Fitting and Checking our Data
- SVM_Model.fit(X_train,Y_train) #Fitting
- SVM_Model.score(X_test,Y_test) #Checking

![Screenshot (3)](https://github.com/user-attachments/assets/c7882b75-67c9-4722-969c-5801cd0b3c65)

Using Random Forest Classifier Model
- from sklearn.ensemble import RandomForestClassifier
- RFC_Model=RandomForestClassifier()
Fitting and Checking our Data
- RFC_Model.fit(X_train,Y_train)
- RFC_Model.score(X_test,Y_test)

![Screenshot (4)](https://github.com/user-attachments/assets/e8309347-9249-465d-90e3-7a878d09fd14)

#### PREDICTING OUR DATA USING RANDOM FOREST CLASSIFIER
- RFC_Model.predict(X_test)
Putting our Y test in a DataFrame
- np.array([Y_test])
Assigning The test result to a variable
- Machine=RFC_Model.predict(X_test)
Creating a Dataframe for the X TEST
- DF=X_test
Creating a column for Our Result
- DF["Our Result"]=Y_test
Creating a column for Machine Result
- DF["Machine"]=Machine
- DF

![Screenshot (10)](https://github.com/user-attachments/assets/60cee7c0-b0a1-41ae-a1cc-a19766486abd)

#### ACCURACY REPORT
Importing the libraries
- from sklearn.metrics import accuracy_score
Getting the prediction and storing it in a variable
- Y_preds=RFC_Model.predict(X_test)
Accuracy Score
- print(accuracy_score(Y_test,Y_preds))

![Screenshot (5)](https://github.com/user-attachments/assets/311d6c28-7cc1-463d-b8b9-eecd1ebf33ac)

#### CLASSIFICATION REPORT
Importing Libraries
- from sklearn.metrics import classification_report
Getting the classfication result
- print(classification_report(Y_test,Y_preds))

![Screenshot (6)](https://github.com/user-attachments/assets/8b974fbf-4e0a-4e39-abc7-48d973556739)

#### CONFUSION MATRIX
Importing Libraries
- from sklearn.metrics import confusion_matrix
- print(confusion_matrix(Y_test,Y_preds))

![Screenshot (7)](https://github.com/user-attachments/assets/b26c029b-6614-405c-9ace-b4fcdbe2d49b)

Importing Library for Confusion Matrix Display
- from sklearn.metrics import ConfusionMatrixDisplay
Getting our Estimator result using Confusion Matrix Display Estimator
- ConfusionMatrixDisplay.from_estimator(RFC_Model,X,Y)

![Screenshot (8)](https://github.com/user-attachments/assets/d85b95fd-be68-4800-ab64-a756d7ff3828)

Getting our predictions using Confusion Matrix Display Prediction
- ConfusionMatrixDisplay.from_predictions(Y_test,Y_preds)

![Screenshot (9)](https://github.com/user-attachments/assets/6e4b12e3-13cf-4a54-8490-73e286497cda)

NOTE: The Estimator Displays the 80% of our Train results, while the Prediction Displays the 20% of our Test.



