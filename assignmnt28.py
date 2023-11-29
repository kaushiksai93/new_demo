import numpy as np 
import pandas as pd 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import streamlit as st


df = pd.read_csv('boston.csv')
print(df)

df_num = df.select_dtypes("number")
Q1 = df_num.quantile(0.25)
Q3 = df_num.quantile(0.75)
IQR = Q3-Q1
UT = Q3 + 1.5*IQR
LT = Q1 - 1.5*IQR
df_trim = df[~((df_num > UT) | (df_num < LT)).any(axis='columns')]
df_features = df_trim.iloc[:,:-1]
df_X_num = df_features.select_dtypes("number")
df_X_cat = df_features.select_dtypes("object")
df_X_concat = pd.concat([df_X_num,df_X_cat],axis='columns')
X = df_X_concat

y = df_trim.MEDV


df_X_num_rescaled = (df_X_num - df_X_num.mean())/df_X_num.std(ddof=1)
df_X_concat = pd.concat([df_X_num,df_X_cat],axis='columns')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

lm = LinearRegression()
model = lm.fit(X_train[['RM']],y_train)

yhat_train = model.predict(X_train[['RM']])
yhat_train_calc = model.intercept_ + model.coef_ * X_train['RM']

model.score(X_train[['RM']],y_train)

SST = sum((y_train - np.mean(y_train))**2)
SSE = sum((y_train - yhat_train)**2)
SSR = sum((yhat_train - np.mean(y_train))**2)

model_slr_r_square = SSR/SST

yhat_test = model.predict(X_test[['RM']])

model_slr_MSE_train = np.mean((y_train - yhat_train)**2)
model_slr_MSE_test = np.mean((y_test - yhat_test)**2)
model_slr_MSE_train, model_slr_MSE_test

model_slr_MAPE_train = np.mean((abs(y_train - yhat_train)/y_train)*100)
model_slr_MAPE_test = np.mean((abs(y_test - yhat_test)/y_test)*100)

model_slr_performance = {'model':"model_slr",'R Sq':model_slr_r_square,'train_MSE':model_slr_MSE_train,'test_MSE':model_slr_MSE_test,'train_MAPE':model_slr_MAPE_train,'test_MAPE':model_slr_MAPE_test}

df_snapshot = pd.DataFrame(columns=['model','R Sq','train_MSE','test_MSE','train_MAPE','test_MAPE'])

st.header("Streamlit demo")

st.sidebar.header("This is a web app")

X_test = st.sidebar.slider("Select X to get yhat", 0, 10, 5)

st.write("X test is:", X_test)

yhat_test = model.predict([[X_test]])

st.write("b0 is", round(model.intercept_, 3))
st.write("b1 is", round(model.coef_[0], 3))
st.write("yhat test is", yhat_test)








