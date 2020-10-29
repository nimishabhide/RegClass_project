import streamlit as st
import pandas as pd
from gtts import gTTS
import os
st.set_option('deprecation.showfileUploaderEncoding', False)

html_temp = """
    <div style="background-color:black ;padding:10px">
    <h1 style="color:white;text-align:center;">REGRESSION AND CLASSIFICATION ON THE GO!!</h1>
    </div>
    """
st.markdown(html_temp, unsafe_allow_html=True)
html_temp69 = """
    <div style="background-color:white ;padding:10px">
    <h3 style="color:black;text-align:center;">PLEASE HAVE A LOOK AT THE SIDEBAR TO MAKE THE BEST USE OF this WEBAPP</h3>
    </div>
    """
st.markdown(html_temp69, unsafe_allow_html=True)
st.sidebar.header("RegClass")
st.sidebar.markdown('<b>This is an one of a kind website where you just need to upload a clean(One which contains just numerical data) csv file and it would assist you in finding the best regressor or classifier which best suits your dataset</b>', unsafe_allow_html=True)
st.sidebar.markdown('<b>This helps you in meeting your fast approaching deadlines by being highly efficient.</b>', unsafe_allow_html=True)
st.sidebar.markdown('<b>Created by:Nimisha Bhide</b>', unsafe_allow_html=True)
st.sidebar.markdown('<b>Email id:nbhide.nb@gmail.com</b>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"],value=0,key=0)
if st.button("Regression"):
    dc=pd.read_csv(uploaded_file)
    df1=dc.copy()
    df2=dc.copy()
    df3=dc.copy()
    df4=dc.copy()
    print(df1)
    X1 = df1.iloc[:, :-1].values
    y1 = df1.iloc[:, -1].values
    from sklearn.model_selection import train_test_split
    X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size = 0.2, random_state = 0)
    from sklearn.linear_model import LinearRegression
    regressor1 = LinearRegression()
    regressor1.fit(X1_train, y1_train)
    y1_pred = regressor1.predict(X1_test)
    from sklearn.metrics import r2_score
    r1=r2_score(y1_test, y1_pred)
    print(r1)
    X2 = df2.iloc[:, :-1].values
    y2 = df2.iloc[:, -1].values
    from sklearn.model_selection import train_test_split
    X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size = 0.2, random_state = 0)
    from sklearn.preprocessing import PolynomialFeatures
    poly_reg = PolynomialFeatures(degree = 4)
    X_poly = poly_reg.fit_transform(X2_train)
    lin_reg_2 = LinearRegression()
    lin_reg_2.fit(X_poly, y2_train)
    y2_pred=lin_reg_2.predict(poly_reg.fit_transform(X2_test))
    from sklearn.metrics import r2_score
    r2=r2_score(y1_test, y1_pred)
    print(r2)
    X3 = df3.iloc[:, :-1].values
    y3 = df3.iloc[:, -1].values
    from sklearn.model_selection import train_test_split
    X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size = 0.2, random_state = 0)
    from sklearn.tree import DecisionTreeRegressor
    regressor3 = DecisionTreeRegressor(random_state = 0)
    regressor3.fit(X3_train, y3_train)
    y3_pred = regressor3.predict(X3_test)
    from sklearn.metrics import r2_score
    r3=r2_score(y3_test, y3_pred)
    print(r3)
    X4 = df4.iloc[:, :-1].values
    y4 = df4.iloc[:, -1].values
    from sklearn.model_selection import train_test_split
    X4_train, X4_test, y4_train, y4_test = train_test_split(X4, y4, test_size = 0.2, random_state = 0)
    from sklearn.ensemble import RandomForestRegressor
    regressor4 = RandomForestRegressor(n_estimators = 10, random_state = 0)
    regressor4.fit(X4_train, y4_train)
    y4_pred = regressor4.predict(X4_test)
    from sklearn.metrics import r2_score
    r4=r2_score(y4_test, y4_pred)
    print(r4)
    if(((r1>r2) and (r1>r3)) and (r1>r4)):
        p="Please use multiple regression for best results"
        st.write("Multiple regression")
        st.write(r1)
    elif(((r2>r1) and (r2>r3)) and (r2>r4)):
        p="Please use plolynomial regression for best results"
        st.write("Polynomial regression")
        st.write(r2)
    elif(((r3>r2) and (r3>r1)) and (r3>r4)):
        p="Please use decision tree regression for best results"
        st.write("Decision Tree regression")
        st.write(r3)
    elif(((r4>r2) and (r4>r3)) and (r4>r1)):
        p="Please use random forest regression for best results"
        st.write("Random Forest regression")
        st.write(r4)
    else:
        p="Please use XGBoost as all other regression techniques are not good enough and XGBoost yeilds the best results for your data"
        st.write('Use XGBoost for regression')
    output=gTTS(text=p,lang=language,slow=False)
    output.save("voice.ogg")
    audio_file = open('voice.ogg', 'rb')
    audio_bytes = audio_file.read()
    st.audio(audio_bytes, format='audio/ogg')
if st.button("Classification"):
    d=pd.read_csv(uploaded_file)
    d1=d.copy()
    d2=d.copy()
    d3=d.copy()
    d4=d.copy()
    d5=d.copy()
    d6=d.copy()
    d7=d.copy()
    print(d1)
    print(d2)
    print(d3)
    a1 = d1.iloc[:, :-1].values
    b1 = d1.iloc[:, -1].values
    from sklearn.model_selection import train_test_split
    a1_train, a1_test, b1_train, b1_test = train_test_split(a1, b1, test_size = 0.25, random_state = 0)
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    a1_train = sc.fit_transform(a1_train)
    a1_test = sc.transform(a1_test)
    from sklearn.linear_model import LogisticRegression
    classifier = LogisticRegression(random_state = 0)
    classifier.fit(a1_train, b1_train)
    b1_pred = classifier.predict(a1_test)
    from sklearn.metrics import accuracy_score
    acc1=accuracy_score(b1_test, b1_pred)
    print(acc1)
    a2 = d2.iloc[:, :-1].values
    b2 = d2.iloc[:, -1].values
    from sklearn.model_selection import train_test_split
    a2_train, a2_test, b2_train, b2_test = train_test_split(a2, b2, test_size = 0.25, random_state = 0)
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    a2_train = sc.fit_transform(a2_train)
    a2_test = sc.transform(a2_test)
    from sklearn.neighbors import KNeighborsClassifier
    classifier2 = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
    classifier2.fit(a2_train, b2_train)
    b2_pred = classifier2.predict(a2_test)
    from sklearn.metrics import accuracy_score
    acc2=accuracy_score(b2_test, b2_pred)
    a3 = d3.iloc[:, :-1].values
    b3 = d3.iloc[:, -1].values
    from sklearn.model_selection import train_test_split
    a3_train, a3_test, b3_train, b3_test = train_test_split(a3, b3, test_size = 0.25, random_state = 0)
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    a3_train = sc.fit_transform(a3_train)
    a3_test = sc.transform(a3_test)
    from sklearn.svm import SVC
    classifier3 = SVC(kernel = 'linear', random_state = 0)
    classifier3.fit(a3_train, b3_train)
    b3_pred = classifier3.predict(a3_test)
    from sklearn.metrics import accuracy_score
    acc3=accuracy_score(b3_test, b3_pred)
    a4 = d4.iloc[:, :-1].values
    b4 = d4.iloc[:, -1].values
    from sklearn.model_selection import train_test_split
    a4_train, a4_test, b4_train, b4_test = train_test_split(a4, b4, test_size = 0.25, random_state = 0)
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    a4_train = sc.fit_transform(a4_train)
    a4_test = sc.transform(a4_test)
    from sklearn.svm import SVC
    classifier4 = SVC(kernel = 'rbf', random_state = 0)
    classifier4.fit(a4_train, b4_train)
    b4_pred = classifier4.predict(a4_test)
    from sklearn.metrics import accuracy_score
    acc4=accuracy_score(b4_test, b4_pred)
    a5 = d5.iloc[:, :-1].values
    b5 = d5.iloc[:, -1].values
    from sklearn.model_selection import train_test_split
    a5_train, a5_test, b5_train, b5_test = train_test_split(a5, b5, test_size = 0.25, random_state = 0)
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    a5_train = sc.fit_transform(a5_train)
    a5_test = sc.transform(a5_test)
    from sklearn.naive_bayes import GaussianNB
    classifier5 = GaussianNB()
    classifier5.fit(a5_train, b5_train)
    b5_pred = classifier5.predict(a5_test)
    from sklearn.metrics import accuracy_score
    acc5=accuracy_score(b5_test, b5_pred)
    a6 = d6.iloc[:, :-1].values
    b6 = d6.iloc[:, -1].values
    from sklearn.model_selection import train_test_split
    a6_train, a6_test, b6_train, b6_test = train_test_split(a6, b6, test_size = 0.25, random_state = 0)
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    a6_train = sc.fit_transform(a6_train)
    a6_test = sc.transform(a6_test)
    from sklearn.tree import DecisionTreeClassifier
    classifier6 = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
    classifier6.fit(a6_train, b6_train)
    b6_pred = classifier6.predict(a6_test)
    from sklearn.metrics import accuracy_score
    acc6=accuracy_score(b6_test, b6_pred)
    a7 = d7.iloc[:, :-1].values
    b7 = d7.iloc[:, -1].values
    from sklearn.model_selection import train_test_split
    a7_train, a7_test, b7_train, b7_test = train_test_split(a7, b7, test_size = 0.25, random_state = 0)
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    a7_train = sc.fit_transform(a7_train)
    a7_test = sc.transform(a7_test)
    from sklearn.ensemble import RandomForestClassifier
    classifier7 = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
    classifier7.fit(a7_train, b7_train)
    b7_pred = classifier7.predict(a7_test)
    from sklearn.metrics import accuracy_score
    acc7=accuracy_score(b7_test, b7_pred)
    if(((acc1>acc2) and (acc1>acc3)) and ((acc1>acc4) and (acc1>acc5)) and ((acc1>acc6) and (acc1>acc7))):
        q="Please use logistic regression for best results"
        st.write('Logistic Regression')
        st.write(acc1)
    elif(((acc2>acc1) and (acc2>acc3)) and ((acc2>acc4) and (acc2>acc5)) and ((acc2>acc6) and (acc2>acc7))):
        q="Please use KNN classification for best results"
        st.write('KNN')
        st.write(acc2)
    elif(((acc3>acc2) and (acc3>acc1)) and ((acc3>acc4) and (acc3>acc5)) and ((acc3>acc6) and (acc3>acc7)) ):
        q="Please use SVM(Linear) classification for best results"
        st.write('SVM(Linear)')
        st.write(acc3)
    elif(((acc4>acc2) and (acc4>acc3)) and ((acc4>acc1) and (acc4>acc5)) and ((acc4>acc6) and(acc4>acc7)) ):
        q="Please use SVM(Kernel) classification for best results"
        st.write('SVM(Kernel)')
        st.write(acc4)
    elif(((acc5>acc2) and (acc5>acc3)) and ((acc5>acc4) and (acc5>acc1)) and ((acc5>acc6) and (acc5>acc7)) ):
        q="Please use Naive Bayes classification for best results"
        st.write('Naive Bayes')
        st.write(acc5)
    elif(((acc6>acc2) and (acc6>acc3)) and ((acc6>acc4) and (acc6>acc5)) and ((acc6>acc1) and (acc6>acc7)) ):
        q="Please use Decision Tree classification for best results"
        st.write('Decision Tree')
        st.write(acc6)
    elif(((acc7>acc2) and (acc7>acc3)) and ((acc7>acc4) and (acc7>acc5)) and ((acc7>acc1) and (acc7>acc7)) ):
        q="Please use Random Forest classification for best results"
        st.write('Random Forest')
        st.write(acc6)
    else:
        q="Please use XGBoost as all other classification techniques are not good enough and XGBoost yeilds the best results for your data"
        st.write('Use XGBoost for Classification')
    output=gTTS(text=q,lang=language,slow=False)
    output.save("voice.ogg")
    audio_file = open('voice.ogg', 'rb')
    audio_bytes = audio_file.read()



    

