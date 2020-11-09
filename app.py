import streamlit as st
import pandas as pd
import numpy as np
import time
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
import scikitplot as skplt
import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score,accuracy_score

def main():
    st.title("Binary Classification Web App")
    st.sidebar.title("Classification model and Hyper-parameters tuning")
    st.markdown("Let's find out whether your mushroom **Edible** or **Poisonous**? ")
    st.sidebar.markdown("Let's find out whether your mushroom **Edible** or **Poisonous**? ")
    
    #with st.spinner('In progress...'):
     
    my_bar = st.progress(0)
    for percent_complete in range(100):
        time.sleep(0.01)
        my_bar.progress(percent_complete + 1)
    #st.success('Done!')    
    
    with st.beta_expander("Little explanation about the use case"):
        st.subheader('Note:- This dataset has been taken from https://www.kaggle.com/uciml/mushroom-classification')
        st.write('I have taken this dataset just for educational and practice purpose.')
        st.subheader("Content")
        st.write('''This dataset includes descriptions of hypothetical samples corresponding to 23 species of gilled mushrooms in the Agaricus and Lepiota Family Mushroom drawn from The Audubon Society Field Guide to North American Mushrooms (1981). Each species is identified as definitely edible, definitely poisonous, or of unknown edibility and not recommended. This latter class was combined with the poisonous one. The Guide clearly states that there is no simple rule for determining the edibility of a mushroom; no rule like "leaflets three, let it be'' for Poisonous Oak and Ivy.

---- Time period: Donated to UCI ML 27 April 1987''')
        st.subheader('Inspiration')
        st.write('''1.What types of machine learning models perform best on this dataset?
        2.Which features are most indicative of a poisonous mushroom?''')

        st.subheader('Acknowledgements')
        st.write('''This dataset was originally donated to the UCI Machine Learning repository.''')
    
    st.subheader('Choose one of the below option')
    predictions = st.radio("Prediction Option",("Test Data","Real-Time Data"),key = 'predictions')
    
    def load_data():
        mushroom_data=pd.read_csv('mushrooms.csv')
        mushroom_data['stalk_root'] = mushroom_data['stalk_root'].replace('?','r')
        return mushroom_data
        
    def encoding(mushroom_data):
        label = LabelEncoder()
        for col in mushroom_data.columns:
            mushroom_data[col] = label.fit_transform(mushroom_data[col])
        return mushroom_data
        
    def split_scaling(df):
        y = df.type
        X = df.drop(columns = ['type'])
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3, random_state = 42)
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test) 
        return X_train,X_test,y_train,y_test
   
    def plot_metrics(matrics_list):
        if 'Confusion Matrix' in matrics_list:
            st.subheader("Confusion Matrix")
            #plot_confusion_matrix(model,X_test,y_test, display_labels = class_names)
            skplt.metrics.plot_confusion_matrix(y_test, y_pred, normalize=False)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()
        if 'ROC Curve' in matrics_list:
            st.subheader("ROC Curve")
            plot_roc_curve(model, X_test,y_test)
            st.pyplot()
        if 'Precision-Recall Curve' in matrics_list:
            st.subheader("Precision Recall")
            plot_precision_recall_curve(model,X_test,y_test)
            st.pyplot()

    df_load = load_data()
    #st.write('DataFrame_load', df_load)
    df = encoding(df_load)
    #st.write('DataFrame', df)
    X_train,X_test,y_train,y_test = split_scaling(df)
    #st.dataframe(X_train)
    class_names = ['edible','poisonous']
    
    
    st.sidebar.subheader("Choose Classifier")
    classifier = st.sidebar.selectbox("Classifier",("Support Vector Machine (SVM)","Logistic regression","Random Forest",'Naive Bayes','Decision Tree'))
    
    if predictions == 'Real-Time Data':
        st.subheader("Choose Input Features")
    
        df_load = load_data()
        
        col1, col2, col3, col4, col5 = st.beta_columns(5)
        dict = {}
        dict_code = {}
        with col1:
            n = 22
            for col in df_load.columns[1:n-16]:
                value = st.selectbox(label = col, options = tuple(df_load[col].sort_values().unique()))
                dict1 = {col:value}
                dict.update(dict1)
                s = dict.get(col)
                label = LabelEncoder()
                label.fit(df_load[col])
                res = {label.classes_[i]: label.transform(label.classes_)[i] for i in range(len(label.classes_))}
                t = res.get(s)
                dict_code1 = {col : t}
                dict_code.update(dict_code1)
   
        with col2:            
            for col in df_load.columns[n-16:n-11]:
                value = st.selectbox(label = col, options = tuple(df_load[col].sort_values().unique()))
                dict2 = {col:value}
                dict.update(dict2)
                s = dict.get(col)
                label = LabelEncoder()
                label.fit(df_load[col])
                res = {label.classes_[i]: label.transform(label.classes_)[i] for i in range(len(label.classes_))}
                t = res.get(s)
                dict_code2 = {col : t}
                dict_code.update(dict_code2)
              
        with col3:            
            for col in df_load.columns[n-11:n-7]:
                value = st.selectbox(label = col, options = tuple(df_load[col].sort_values().unique()))
                dict3 = {col:value}
                dict.update(dict3)
                s = dict.get(col)
                label = LabelEncoder()
                label.fit(df_load[col])
                res = {label.classes_[i]: label.transform(label.classes_)[i] for i in range(len(label.classes_))}
                t = res.get(s)
                dict_code3 = {col : t}
                dict_code.update(dict_code3)
              
        with col4:            
            for col in df_load.columns[n-7:n-3]:
                value = st.selectbox(label = col, options = tuple(df_load[col].sort_values().unique()))
                dict4 = {col:value}
                dict.update(dict4)
                s = dict.get(col)
                label = LabelEncoder()
                label.fit(df_load[col])
                res = {label.classes_[i]: label.transform(label.classes_)[i] for i in range(len(label.classes_))}
                t = res.get(s)
                dict_code4 = {col : t}
                dict_code.update(dict_code4)
              
        with col5:            
            for col in df_load.columns[n-3:n+1]:
                value = st.selectbox(label = col, options = tuple(df_load[col].sort_values().unique()))
                dict5 = {col:value}
                dict.update(dict5)
                s = dict.get(col)
                label = LabelEncoder()
                label.fit(df_load[col])
                res = {label.classes_[i]: label.transform(label.classes_)[i] for i in range(len(label.classes_))}
                t = res.get(s)
                dict_code5 = {col : t}
                dict_code.update(dict_code5)
                
        features_data = pd.DataFrame(dict, index=[0])
        st.write('DataFrame with selected features',features_data)
        
        features_coded = pd.DataFrame(dict_code, index=[0])
        #st.write('DataFrame_coded',features_coded)
        
        row_value = features_coded.iloc[0:1,:].values
        #st.write('row_value',row_value)
        
        def split_scaling(data):
            y = data.type
            X = data.drop(columns = ['type'])
            X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3, random_state = 42)
            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            row_value_scaled = sc.transform(row_value)
            return row_value_scaled
         
        row_value_scaled_2 = split_scaling(df)    
        #st.write('scaled_row_value',row_value_scaled_2)
        
        if classifier == 'Support Vector Machine (SVM)':
            st.sidebar.subheader("Choose Model")
            model = st.sidebar.radio("Model",("LinearSVC","SVC"),key = 'model')
            if model == 'LinearSVC':
                st.sidebar.subheader("Model Hyperparameters")
                C1 = st.sidebar.number_input("C (Regularization parameter)", 0.01,10.0,step=0.01,key='C')
                max_iter = st.sidebar.slider("Maximum number of iterations", 1000,15000,key='max_iter')
            else:    
                st.sidebar.subheader("Model Hyperparameters")
                C = st.sidebar.number_input("C (Regularization parameter)", 0.01,10.0,step=0.01,key='C')
                kernel = st.sidebar.radio("Kernel",("rbf","poly"),key = 'kernel')
                gamma = st.sidebar.radio("Gamma (Kernel Cofficient)",("scale","auto"),key='gamma')

            #metrics = st.sidebar.multiselect("What metrices to plot?", ('Confusion Matrix','ROC Curve','Precision-Recall Curve'))

            if st.sidebar.button("Classify", key='classify'):
                st.subheader("Support Vector Machine (SVM) Results")
                if model == 'LinearSVC':
                    model = LinearSVC(C = C1, max_iter = max_iter, dual = True, loss='hinge')
                    model.fit(X_train,y_train)
                    y_pred = model.predict(row_value_scaled_2)
                    #st.write('Prediction',y_pred)
                    if y_pred == 1:
                        st.markdown("According to our prediction this Mushroom is **Poisonous**")
                    else:
                        st.markdown("According to our prediction this Mushroom is **Edible**")
                    #st.write("Accuracy: ", accuracy_score(y_test,y_pred))
                    #st.write("Precission: ",precision_score(y_test,y_pred,labels=class_names).round(2))
                    #st.write("Recall: ",recall_score(y_test,y_pred,labels=class_names).round(2))
                    #plot_metrics(metrics)
                else:
                    model = SVC(C=C,kernel=kernel,gamma=gamma)
                    model.fit(X_train,y_train)
                    accuracy = model.score(X_test,y_test)
                    y_pred = model.predict(row_value_scaled_2)
                    #st.write('Prediction',y_pred)
                    if y_pred == 1:
                        st.markdown("According to our prediction this Mushroom is **Poisonous**")
                    else:
                        st.markdown("According to our prediction this Mushroom is **Edible**")
                    #st.write("Accuracy: ", accuracy.round(2))
                    #st.write("Precission: ",precision_score(y_test,y_pred,labels=class_names).round(2))
                    #st.write("Recall: ",recall_score(y_test,y_pred,labels=class_names).round(2))
                    #plot_metrics(metrics)
        if classifier == 'Logistic regression':
            st.sidebar.subheader("Model Hyperparameters")
            C = st.sidebar.number_input("C (Regularization parameter)", 0.01,10.0,step=0.01,key='C_LR')
            max_iter = st.sidebar.slider("Maximum number of iterations", 100,500,key='max_iter')

            #metrics = st.sidebar.multiselect("What metrices to plot?", ('Confusion Matrix','ROC Curve','Precision-Recall Curve'))

            if st.sidebar.button("Classify", key='classify'):
                st.subheader("Logistic regression Results")
                model = LogisticRegression(C=C,max_iter=max_iter)
                model.fit(X_train,y_train)
                accuracy = model.score(X_test,y_test)
                y_pred = model.predict(row_value_scaled_2)
                #st.write('Prediction',y_pred)
                if y_pred == 1:
                    st.markdown("According to our prediction this Mushroom is **Poisonous**")
                else:
                    st.markdown("According to our prediction this Mushroom is **Edible**")
                

        if classifier == 'Random Forest':
            st.sidebar.subheader("Model Hyperparameters")
            n_estimators=st.sidebar.number_input("The number of trees in the forest",10,200,step = 10,key='n_estimator')
            max_depth = st.sidebar.number_input("The maximum depth",1,20,step=1,key="max_depth")
            bootstrap = st.sidebar.radio("Bootstrap samples ",('True','False'),key='bootstrap')

            #metrics = st.sidebar.multiselect("What metrices to plot?", ('Confusion Matrix','ROC Curve','Precision-Recall Curve'))

            if st.sidebar.button("Classify", key='classify'):
                st.subheader("Random Forest Results")
                model = RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth,bootstrap=bootstrap,n_jobs=-1)
                model.fit(X_train,y_train)
                accuracy = model.score(X_test,y_test)
                y_pred = model.predict(row_value_scaled_2)
                #st.write('Prediction',y_pred)
                if y_pred == 1:
                    st.markdown("According to our prediction this Mushroom is **Poisonous**")
                else:
                    st.markdown("According to our prediction this Mushroom is **Edible**")
                
     
        if classifier == 'Decision Tree':
            st.sidebar.subheader("Model Hyperparameters")
            max_depth = st.sidebar.number_input("The maximum depth",1,20,step=1,key="max_depth")
            criterion = st.sidebar.radio("Criterion ",('entropy','gini'),key='criterion')

            #metrics = st.sidebar.multiselect("What metrices to plot?", ('Confusion Matrix','ROC Curve','Precision-Recall Curve'))

            if st.sidebar.button("Classify", key='classify'):
                st.subheader('Decision Tree Results')
                model = DecisionTreeClassifier(max_depth = max_depth, criterion = criterion)
                model.fit(X_train,y_train)
                y_pred = model.predict(row_value_scaled_2)
                #st.write('Prediction',y_pred)
                if y_pred == 1:
                    st.markdown("According to our prediction this Mushroom is **Poisonous**")
                else:
                    st.markdown("According to our prediction this Mushroom is **Edible**")
                
            
        if classifier == 'Naive Bayes':
            #st.sidebar.subheader("Model Hyperparameters") 
            #metrics = st.sidebar.multiselect("What metrices to plot?", ('Confusion Matrix','ROC Curve','Precision-Recall Curve'))

            if st.sidebar.button("Classify", key='classify'):
                st.subheader('Naive Bayes Results')
                model = GaussianNB()
                model.fit(X_train, y_train)
                y_pred = model.predict(row_value_scaled_2)
                #st.write('Prediction',y_pred)
                if y_pred == 1:
                    st.markdown("According to our prediction this Mushroom is **Poisonous**")
                else:
                    st.markdown("According to our prediction this Mushroom is **Edible**")
                           
    
        with st.beta_expander("See explanation of input features"):
            st.subheader("About the Input features")
            st.write('''Attribute Information: (classes: edible=e, poisonous=p)

cap-shape: bell=b,conical=c,convex=x,flat=f, knobbed=k,sunken=s

cap-surface: fibrous=f,grooves=g,scaly=y,smooth=s

cap-color: brown=n,buff=b,cinnamon=c,gray=g,green=r,pink=p,purple=u,red=e,white=w,yellow=y

bruises: bruises=t,no=f

odor: almond=a,anise=l,creosote=c,fishy=y,foul=f,musty=m,none=n,pungent=p,spicy=s

gill-attachment: attached=a,descending=d,free=f,notched=n

gill-spacing: close=c,crowded=w,distant=d

gill-size: broad=b,narrow=n

gill-color: black=k,brown=n,buff=b,chocolate=h,gray=g, green=r,orange=o,pink=p,purple=u,red=e,white=w,yellow=y

stalk-shape: enlarging=e,tapering=t

stalk-root: bulbous=b,club=c,cup=u,equal=e,rhizomorphs=z,rooted=r

stalk-surface-above-ring: fibrous=f,scaly=y,silky=k,smooth=s

stalk-surface-below-ring: fibrous=f,scaly=y,silky=k,smooth=s

stalk-color-above-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y

stalk-color-below-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y

veil-type: partial=p,universal=u

veil-color: brown=n,orange=o,white=w,yellow=y

ring-number: none=n,one=o,two=t

ring-type: cobwebby=c,evanescent=e,flaring=f,large=l,none=n,pendant=p,sheathing=s,zone=z

spore-print-color: black=k,brown=n,buff=b,chocolate=h,green=r,orange=o,purple=u,white=w,yellow=y

population: abundant=a,clustered=c,numerous=n,scattered=s,several=v,solitary=y

habitat: grasses=g,leaves=l,meadows=m,paths=p,urban=u,waste=w,woods=d''')
    
    if predictions == 'Test Data':
        if classifier == 'Support Vector Machine (SVM)':
            st.sidebar.subheader("Choose Model")
            model = st.sidebar.radio("Model",("LinearSVC","SVC"),key = 'model')
            if model == 'LinearSVC':
                st.sidebar.subheader("Model Hyperparameters")
                C1 = st.sidebar.number_input("C (Regularization parameter)", 0.01,10.0,step=0.01,key='C')
                max_iter = st.sidebar.slider("Maximum number of iterations", 1000,15000,key='max_iter')
            else:    
                st.sidebar.subheader("Model Hyperparameters")
                C = st.sidebar.number_input("C (Regularization parameter)", 0.01,10.0,step=0.01,key='C')
                kernel = st.sidebar.radio("Kernel",("rbf","poly"),key = 'kernel')
                gamma = st.sidebar.radio("Gamma (Kernel Cofficient)",("scale","auto"),key='gamma')

            metrics = st.sidebar.multiselect("What metrices to plot?", ('Confusion Matrix','ROC Curve','Precision-Recall Curve'))

            if st.sidebar.button("Classify", key='classify'):
                st.subheader("Support Vector Machine (SVM) Results")
                if model == 'LinearSVC':
                    model = LinearSVC(C = C1, max_iter = max_iter, dual = True, loss='hinge')
                    model.fit(X_train,y_train)
                    accuracy = model.score(X_test,y_test)
                    y_pred = model.predict(X_test)
                    st.write("Accuracy: ", accuracy.round(2))
                    st.write("Precission: ",precision_score(y_test,y_pred,labels=class_names).round(2))
                    st.write("Recall: ",recall_score(y_test,y_pred,labels=class_names).round(2))
                    plot_metrics(metrics)
                else:
                    model = SVC(C=C,kernel=kernel,gamma=gamma)
                    model.fit(X_train,y_train)
                    accuracy = model.score(X_test,y_test)
                    y_pred = model.predict(X_test)
                    st.write("Accuracy: ", accuracy.round(2))
                    st.write("Precission: ",precision_score(y_test,y_pred,labels=class_names).round(2))
                    st.write("Recall: ",recall_score(y_test,y_pred,labels=class_names).round(2))
                    plot_metrics(metrics)
           
        if classifier == 'Logistic regression':
            st.sidebar.subheader("Model Hyperparameters")
            C = st.sidebar.number_input("C (Regularization parameter)", 0.01,10.0,step=0.01,key='C_LR')
            max_iter = st.sidebar.slider("Maximum number of iterations", 100,500,key='max_iter')

            metrics = st.sidebar.multiselect("What metrices to plot?", ('Confusion Matrix','ROC Curve','Precision-Recall Curve'))

            if st.sidebar.button("Classify", key='classify'):
                st.subheader("Logistic regression Results")
                model = LogisticRegression(C=C,max_iter=max_iter)
                model.fit(X_train,y_train)
                accuracy = model.score(X_test,y_test)
                y_pred = model.predict(X_test)
                st.write("Accuracy: ", accuracy.round(2))
                st.write("Precission: ",precision_score(y_test,y_pred,labels=class_names).round(2))
                st.write("Recall: ",recall_score(y_test,y_pred,labels=class_names).round(2))
                plot_metrics(metrics)

        if classifier == 'Random Forest':
            st.sidebar.subheader("Model Hyperparameters")
            n_estimators=st.sidebar.number_input("The number of trees in the forest",10,200,step = 10,key='n_estimator')
            max_depth = st.sidebar.number_input("The maximum depth",1,20,step=1,key="max_depth")
            bootstrap = st.sidebar.radio("Bootstrap samples ",('True','False'),key='bootstrap')

            metrics = st.sidebar.multiselect("What metrices to plot?", ('Confusion Matrix','ROC Curve','Precision-Recall Curve'))

            if st.sidebar.button("Classify", key='classify'):
                st.subheader("Random Forest Results")
                model = RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth,bootstrap=bootstrap,n_jobs=-1)
                model.fit(X_train,y_train)
                accuracy = model.score(X_test,y_test)
                y_pred = model.predict(X_test)
                st.write("Accuracy: ", accuracy.round(2))
                st.write("Precission: ",precision_score(y_test,y_pred,labels=class_names).round(2))
                st.write("Recall: ",recall_score(y_test,y_pred,labels=class_names).round(2))
                plot_metrics(metrics)
     
        if classifier == 'Decision Tree':
            st.sidebar.subheader("Model Hyperparameters")
            max_depth = st.sidebar.number_input("The maximum depth",1,20,step=1,key="max_depth")
            criterion = st.sidebar.radio("Criterion ",('entropy','gini'),key='criterion')

            metrics = st.sidebar.multiselect("What metrices to plot?", ('Confusion Matrix','ROC Curve','Precision-Recall Curve'))

            if st.sidebar.button("Classify", key='classify'):
                st.subheader('Decision Tree Results')
                model = DecisionTreeClassifier(max_depth = max_depth, criterion = criterion)
                model.fit(X_train,y_train)
                accuracy = model.score(X_test,y_test)
                y_pred = model.predict(X_test)
                st.write("Accuracy: ", accuracy.round(2))
                st.write("Precission: ",precision_score(y_test,y_pred,labels=class_names).round(2))
                st.write("Recall: ",recall_score(y_test,y_pred,labels=class_names).round(2))
                plot_metrics(metrics)
            
        if classifier == 'Naive Bayes':
            #st.sidebar.subheader("Model Hyperparameters") 
            metrics = st.sidebar.multiselect("What metrices to plot?", ('Confusion Matrix','ROC Curve','Precision-Recall Curve'))

            if st.sidebar.button("Classify", key='classify'):
                st.subheader('Naive Bayes Results')
                model = GaussianNB()
                model.fit(X_train, y_train)
                accuracy = model.score(X_test,y_test)
                y_pred = model.predict(X_test)
                st.write("Accuracy: ", accuracy.round(2))
                st.write("Precission: ",precision_score(y_test,y_pred,labels=class_names).round(2))
                st.write("Recall: ",recall_score(y_test,y_pred,labels=class_names).round(2))
                plot_metrics(metrics)
     
    df_load = load_data()
    if st.sidebar.checkbox("Show raw data ",False):
        st.subheader("Mushroom Dataset")
        st.write(df_load)
         
                
if __name__ == '__main__':
    main()
