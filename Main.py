from statistics import mode
import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
from PIL import Image
regre_lineal = linear_model.LinearRegression()


df = ''
data = ''
c_x = ''
c_y = ''


def main():
    global df, data, x, y, c_x, c_y, regre_lineal
    st.title('Modelamiento de Datos')
    st.sidebar.header('Ingresar Parametros')

    Algoritmo = ['Regresion Lineal', 'Regresion Polinomial', 'Clasificador Gaussiano', 'Arboles de Desicion', 'Redes Neuronales']
    model = st.sidebar.selectbox('Seleccione el tipo de Algoritmo', Algoritmo)

    Graphs = ['Graficar puntos', 'Definir función de tendencia', 'Realizar predicción de tendencia', 'Clasificar por Gauss',  'Clasificar por árboles de decisión',  'Clasificador por redes neuronales']
    ne = st.sidebar.selectbox('Seleccione la Operacion a Realizar', Graphs)

    grado = st.sidebar.text_input('Ingrese en grado del polinomio')

    pred = st.sidebar.text_input('Ingrese el valor a predecir :')

    data = st.file_uploader("Seleccione el Archivo", type=["csv", "xls", "xlsx", "json"])
    if data is not None:
        spli = os.path.splitext(data.name)
        if spli[1] == '.csv':
            df = pd.read_csv(data)
            st.dataframe(df)
            x  = df.head()
            y  = df.head()
            c_x = st.selectbox('seleccione X: ', x.columns)
            c_y = st.selectbox('seleccione Y: ', y.columns) 
            if st.sidebar.button('Realizar accion'):
                if ((model == 'Regresion Lineal') or (model == 'Regresion Polinomial')) and (ne == 'Graficar puntos'):
                    fil = df[c_x].tolist()
                    col = df[c_y].tolist()
                    plt.scatter(fil, col, color='blue')
                    plt.ylabel(c_y)
                    plt.xlabel(c_x)
                    if (model == 'Regresion Lineal'):
                        cx = np.array(df[c_x]).reshape(-1,1)
                        cy = df[c_y]
                        regre_lineal.fit(cx, cy)
                        b0 = regre_lineal.intercept_
                        b1 = regre_lineal.coef_
                        if (b0 < 0):
                            st.write(str(b1[0]) + 'x ' + str(b0))
                        elif (b0 > 0):
                            st.write(str(b1[0]) + 'x + ' + str(b0))
                        y_p = b1[0]*cx+b0
                        plt.plot(cx, y_p, color='red')
                    if (model == 'Regresion Polinomial'):
                        if grado is not None: 
                            cx = np.asarray(df[c_x]).reshape(-1, 1)
                            cy = df[c_y]
                            polgra = PolynomialFeatures(degree= int(grado))
                            x_t = polgra.fit_transform(cx)
                            regre_lineal.fit(x_t, cy)
                            #plt.plot(x_t, cy)
                            y_pred = regre_lineal.predict(x_t)
                            plt.plot(cx, y_pred, color='red')
                    plt.savefig('Dispersion.png')
                    plt.close()
                    image = Image.open('Dispersion.png')
                    st.image(image, caption="Grafica de Dispersion")
                elif ((model == 'Regresion Lineal') or (model == 'Regresion Polinomial')) and (ne == 'Definir función de tendencia'):
                    if (model == 'Regresion Lineal'):
                        cx = np.array(df[c_x]).reshape(-1,1)
                        cy = df[c_y]
                        regre_lineal.fit(cx, cy)
                        R = regre_lineal.score(cx,cy)
                        st.write('Pendiente : ')
                        b1 = regre_lineal.coef_
                        st.write(b1)
                        st.write('Intercepto : ')
                        b0 = regre_lineal.intercept_
                        st.write(b0)
                        st.write('Funcion de tendencia central')
                        if (b0 < 0):
                            st.write(str(b1[0]) + 'x ' + str(b0))
                        elif (b0 > 0):
                            st.write(str(b1[0]) + 'x + ' + str(b0))
                        st.write('Coeficiente de Correlacion')
                        st.write(R)
                    elif (model == 'Regresion Polinomial'):
                        if grado is not None: 
                            cx = np.asarray(df[c_x]).reshape(-1, 1)
                            cy = df[c_y]
                            polgra = PolynomialFeatures(degree= int(grado))
                            x_t = polgra.fit_transform(cx)
                            regre_lineal.fit(x_t, cy)
                            st.write('Valor de los Coeficientes :')
                            coer = regre_lineal.coef_
                            st.write(coer)
                            st.write('Valor del Intercepto :')
                            st.write(regre_lineal.intercept_)
                            st.write('Coeficiente de Correlacion :')
                            y_pred = regre_lineal.predict(x_t)
                            st.write(regre_lineal.score(x_t, cy))
                            st.write('Funcion de Tendencia Central :')
                            concatenacion = ""
                            da = len(coer) - 1
                            while da>=0:
                                if da != 0:
                                    concatenacion = concatenacion + str(round(coer[da], 0)) + 'X^' + str(da) + '+'
                                else:
                                    concatenacion = concatenacion + str(round(coer[da], 0))
                                da = da - 1
                            st.write(concatenacion)
                elif ((model == 'Regresion Lineal') or (model == 'Regresion Polinomial')) and (ne == 'Realizar predicción de tendencia'):
                    if (model == 'Regresion Lineal'):
                        cx = np.array(df[c_x]).reshape(-1,1)
                        cy = df[c_y]
                        pred1 = int(pred)
                        regre_lineal.fit(cx, cy)
                        b0 = regre_lineal.intercept_
                        b1 = regre_lineal.coef_
                        y_p = b1[0]*pred1+b0
                        st.write('El valor que se predijo es :')
                        st.write(y_p)
                    elif(model == 'Regresion Polinomial'):
                        if grado is not None:
                            cx = np.asarray(df[c_x]).reshape(-1,1)
                            cy = df[c_y]
                            polgra = PolynomialFeatures(degree=int(grado))
                            x_t = polgra.fit_transform(cx)
                            regre_lineal.fit(x_t, cy)
                            y_pred = regre_lineal.predict(x_t)
                            aux = 0
                            coer = regre_lineal.coef_
                            tam = len(regre_lineal.coef_)-1
                            while tam >= 0:
                                aux = aux + round((coer[tam]),0)*int(pred)**(int(tam))
                                tam = tam -1
                            st.write('La prediccion es de : ')
                            st.write(aux)
        elif spli[1] == '.xls':
            df = pd.read_excel(data)
            x = df.head()
            y = df.head()
            c_x = st.selectbox('Seleccione el eje X: ', x.columns)
            c_y = st.selectbox('Seleccione el eje Y: ', y.columns)
        elif spli[1] == '.xlsx':
            df = pd.read_excel(data)
            x = df.head()
            y = df.head()
            c_x = st.selectbox('Seleccione el eje X: ', x.columns)
            c_y = st.selectbox('Seleccione el eje Y: ', y.columns)
        elif spli[1] == '.json':
            df = pd.read_json(data)
            x = df.head()
            y = df.head()
            c_x = st.selectbox('Seleccione el eje X: ', x.columns)
            c_y = st.selectbox('Seleccione el eje Y: ', y.columns)

if __name__ == '__main__':
    main()