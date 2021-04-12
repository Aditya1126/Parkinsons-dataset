import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier,BaggingClassifier,GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn. metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score, recall_score


uploaded_file = st.sidebar.file_uploader("Choose a file")
    
data = pd.read_csv(uploaded_file)      
uploaded_file.seek(0)
top=st.sidebar.selectbox("Select the option for top view",(5,10,20,50))
st.write("First",top,"views of data")
st.write(data.head(top))
bottom=st.sidebar.selectbox("Select the option for bottom view",(5,10,20,50))
st.write("Last",bottom,"views of data")
st.write(data.head(bottom))
st.write("Shape of dataset",data.shape)
        #pre=st.sidebar.selectbox("Select the option",("info","describe"))       
        
        #if pre=="describe":
st.subheader("Describe")
st.write("The describe() method is used for calculating some statistical data like percentile, mean and std of the numerical values of the\
Series or DataFrame.It analyzes both numeric and object series and also the DataFrame column sets of mixed data types.")
st.write(data.describe().transpose())
st.subheader("isNull")
st.write("This function takes a scalar or array-like object and indicates whether values \
        are missing (NaN in numeric arrays, None or NaN in object arrays, NaT in datetimelike).")
st.write(data.isnull().sum())
        #if pre=="info":
        #    data1=data.info()
        #    st.write(data1)

graph=st.sidebar.selectbox("Select the option",("Count plot","Scatter plot","Heat Map"))

if graph=="Count plot":
        x=st.sidebar.selectbox("Select the option",("Gender","Recording","Status"))
        if x=="Gender":
                data['Gender']=data["Gender"].replace(to_replace=0,value="Male")
                data['Gender']=data["Gender"].replace(to_replace=1,value="Female")
                sns.countplot(x=x,hue=x, data=data)
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.pyplot()
        elif x=="Recording":
                sns.countplot(x=x,hue=x, data=data)
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.pyplot()
        elif x=="Status":
                sns.countplot(x=x,hue=x, data=data)
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.pyplot()

if graph=="Heat Map":
        x=st.sidebar.slider("Select the start column",1,48)
        y=st.sidebar.slider("Select the end column",1,48)
        plt.figure(figsize=(20, 10)) 
        sns.heatmap(data[data.columns[x:y]].corr(),annot=True)
        st.pyplot()

if graph=="Scatter plot":
        col=st.sidebar.selectbox("Select the column",("Gender","Recording","Status"))
        hue=st.sidebar.selectbox("Select the hue",("Recording","Gender","Status"))
        x=st.sidebar.selectbox("Select the x-axis",('Jitter_rel', 'Jitter_abs',
       'Jitter_RAP', 'Jitter_PPQ', 'Shim_loc', 'Shim_dB', 'Shim_APQ3',
       'Shim_APQ5', 'Shi_APQ11', 'HNR05', 'HNR15', 'HNR25', 'HNR35', 'HNR38',
       'RPDE', 'DFA', 'PPE', 'GNE', 'MFCC0', 'MFCC1', 'MFCC2', 'MFCC3',
       'MFCC4', 'MFCC5', 'MFCC6', 'MFCC7', 'MFCC8', 'MFCC9', 'MFCC10',
       'MFCC11', 'MFCC12', 'Delta0', 'Delta1', 'Delta2', 'Delta3', 'Delta4',
       'Delta5', 'Delta6', 'Delta7', 'Delta8', 'Delta9', 'Delta10', 'Delta11',
       'Delta12'))
        y=st.sidebar.selectbox("Select the y-axis",('Jitter_abs','Jitter_rel',
       'Jitter_RAP', 'Jitter_PPQ', 'Shim_loc', 'Shim_dB', 'Shim_APQ3',
       'Shim_APQ5', 'Shi_APQ11', 'HNR05', 'HNR15', 'HNR25', 'HNR35', 'HNR38',
       'RPDE', 'DFA', 'PPE', 'GNE', 'MFCC0', 'MFCC1', 'MFCC2', 'MFCC3',
       'MFCC4', 'MFCC5', 'MFCC6', 'MFCC7', 'MFCC8', 'MFCC9', 'MFCC10',
       'MFCC11', 'MFCC12', 'Delta0', 'Delta1', 'Delta2', 'Delta3', 'Delta4',
       'Delta5', 'Delta6', 'Delta7', 'Delta8', 'Delta9', 'Delta10', 'Delta11',
       'Delta12'))
        plt.figure(figsize=(40, 30)) 
        g = sns.FacetGrid(data, col=col,hue=hue)
        g.map(sns.scatterplot, x, y, alpha=.9)
        g.add_legend()
        st.pyplot()

algo=st.sidebar.selectbox("Select the Algorithm",("None","Decision Tree","Support vector machine","Random Forest","Gaussian Naive Bayes"))

data['Gender']=data["Gender"].replace(to_replace="Male",value=0)
data['Gender']=data["Gender"].replace(to_replace="Female",value=1)

Y_data=data['Status']
X_data=data.drop(data[['ID','Status']],axis=1)
x_train,x_test,y_train,y_test=train_test_split(X_data,Y_data,test_size=0.25,random_state=5)
st.write("number of test samples :", x_test.shape[0])
st.write("number of training samples:",x_train.shape[0])



if algo=="Random Forest":
        st.header("Random Forest Classifier")
        rfc=RandomForestClassifier(n_estimators=225,random_state=1)
        rfc.fit(x_train,y_train)
        y_pred=rfc.predict(x_test)
        st.text(classification_report(y_test, y_pred))
        cm = confusion_matrix(y_test, y_pred)
        st.write('Confusion Matrix:\n',cm, '\n')
        st.write('Accuracy:', accuracy_score(y_test, y_pred))
        st.write("Overall Precision:",precision_score(y_test, y_pred))
        st.write("Overall Recall:",recall_score(y_test, y_pred))
        plt.figure(figsize=(10, 5))
        sns.heatmap(cm, annot=True)
        st.pyplot()
        y_scores = rfc.predict_proba(x_test)
        auc = roc_auc_score(y_test,y_scores[:,1])
        st.write('\nAUC: ' + str(auc))
        fpr, tpr, thresholds = roc_curve(y_test, y_scores[:,1])
        fig = plt.figure(figsize=(6, 6))
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr, tpr)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.show()
        st.pyplot()


if algo=="Decision Tree":
        st.header("Decision Tree Classifier")
        dtc=DecisionTreeClassifier()
        dtc.fit(x_train,y_train)
        y_pred=dtc.predict(x_test)
        st.text(classification_report(y_test, y_pred))
        cm = confusion_matrix(y_test, y_pred)
        st.write('Confusion Matrix:\n',cm, '\n')
        st.write('Accuracy:', accuracy_score(y_test, y_pred))
        st.write("Overall Precision:",precision_score(y_test, y_pred))
        st.write("Overall Recall:",recall_score(y_test, y_pred))
        plt.figure(figsize=(10, 5))
        sns.heatmap(cm, annot=True)
        st.pyplot()
        y_scores = dtc.predict_proba(x_test)
        auc = roc_auc_score(y_test,y_scores[:,1])
        st.write('\nAUC: ' + str(auc))
        fpr, tpr, thresholds = roc_curve(y_test, y_scores[:,1])
        fig = plt.figure(figsize=(6, 6))
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr, tpr)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.show()
        st.pyplot()

if algo=="Support vector machine":
        st.header("Support vector machine")
        svc = SVC(kernel='rbf',C=30,gamma=0.01,probability=True)
        svc.fit(x_train, y_train)
        y_pred =svc.predict(x_test)
        st.text(classification_report(y_test, y_pred))
        cm = confusion_matrix(y_test, y_pred)
        st.write('Confusion Matrix:\n',cm, '\n')
        st.write('Accuracy:', accuracy_score(y_test, y_pred))
        st.write("Overall Precision:",precision_score(y_test, y_pred))
        st.write("Overall Recall:",recall_score(y_test, y_pred))
        plt.figure(figsize=(10, 5))
        sns.heatmap(cm, annot=True)
        st.pyplot()
        y_scores = svc.predict_proba(x_test)
        auc = roc_auc_score(y_test,y_scores[:,1])
        st.write('\nAUC: ' + str(auc))
        fpr, tpr, thresholds = roc_curve(y_test, y_scores[:,1])
        fig = plt.figure(figsize=(6, 6))
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr, tpr)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.show()
        st.pyplot()

if algo=="Gaussian Naive Bayes":
        st.header("Gaussian Naive Bayes")
        nb=GaussianNB()
        nb.fit(x_train,y_train)
        y_pred=nb.predict(x_test)
        st.text(classification_report(y_test, y_pred))
        cm = confusion_matrix(y_test, y_pred)
        st.write('Confusion Matrix:\n',cm, '\n')
        st.write('Accuracy:', accuracy_score(y_test, y_pred))
        st.write("Overall Precision:",precision_score(y_test, y_pred))
        st.write("Overall Recall:",recall_score(y_test, y_pred))
        plt.figure(figsize=(10, 5))
        sns.heatmap(cm, annot=True)
        st.pyplot()
        y_scores = nb.predict_proba(x_test)
        auc = roc_auc_score(y_test,y_scores[:,1])
        st.write('\nAUC: ' + str(auc))
        fpr, tpr, thresholds = roc_curve(y_test, y_scores[:,1])
        fig = plt.figure(figsize=(6, 6))
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr, tpr)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.show()
        st.pyplot()