from matplotlib import pyplot as plt
import numpy as np
import streamlit as st
import pandas as pd

np.set_printoptions(precision=3)

st.title("Eigen Values and Eigen Vectors")
st.write(
    "This app shows the effect of linear transformation with respect to eigen values and eigen vectors"
)

def getSquareY(x):
    if x==-1 or x == 1:
        return 0
    else:
        return 1

getSquareYVectorised = np.vectorize(getSquareY)

def getCircle(x):
    return np.sqrt(1-np.square(x))

with st.sidebar:
    data = st.selectbox('Select type of dataset', ['Square', 'Circle'])
    st.write("---")
    st.text("Enter transformation matrix elements")
    a_00 = st.slider(label = '$A_{0,0}$', min_value = -5, max_value=5, value=1)
    a_01 = st.slider(label = '$A_{0,1}$', min_value = -5, max_value=5, value=0)
    a_10 = st.slider(label = '$A_{1,0}$', min_value = -5, max_value=5, value=0)
    a_11 = st.slider(label = '$A_{1,1}$', min_value = -5, max_value=5, value=1)

def transform(x,y):
    return a_00 * x + a_01 * y, a_10 * x + a_11 * y
    

x = np.linspace(-1,1,1000)
y = getSquareYVectorised(x) if data == 'Square' else getCircle(x)

x_dash_up, y_dash_up = transform(x,y)
x_dash_down, y_dash_down = transform(x,-y)

t = np.array([[a_00,a_01], [a_10, a_11]], dtype=np.float64)

try:
    evl, evec = np.linalg.eig(t)
    fig, ax = plt.subplots()
    ax.plot(x_dash_up,y_dash_up,'r')
    ax.plot(x_dash_down,y_dash_down, 'g')
    ax.quiver(0,0,evec[0][0],evec[0][1],scale=1,scale_units ='xy',angles='xy', facecolor='yellow', label='$\lambda_0$')
    ax.quiver(0,0,evec[1][0],evec[1][1],scale=1,scale_units ='xy',angles='xy', facecolor='blue',label='$\lambda_1$')
    ax.set_xlim(-5,5)
    ax.set_ylim(-5,5)
    ax.set_aspect('equal', adjustable='box')
    fig.legend()
    st.pyplot(fig)

    df = pd.DataFrame({'Eigen Values': evl, 'Eigen Vectors': [str(evec[0]), str(evec[1])]})
    st.table(df)
except:
    st.write("Given matrix has eigen vectors in complex space")