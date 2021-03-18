## @knitr Item1
import numpy as np
from numpy.core.fromnumeric import transpose
import pandas as pd
import prince
import matplotlib.pyplot as plt
import seaborn as sns
from   sklearn.cluster import KMeans

class ProceseArchivo():
  def CargarArchivo(self,nombrearchivo):
    self.__df=pd.read_csv(nombrearchivo, delimiter=',',decimal=".")    
    return True  
  
  def ReducirDataFrame(self,columnas):
    df_reducido=self.__df.drop(  self.__df.columns[columnas],axis=1)
    return df_reducido
  
  def VisualizarDatos(self):
    print("Contenido de archivo leido")
    print(self.__df.iloc[0:8,0:5])
    print("\nNombres de las columnas (variables)")
    print(self.__df.columns)
    print("\nNombres de los primeros 10 países")
    nombres=self.__df.iloc[:,1]
    print(nombres[0:10])
  
  def ProcesarDatos(self,NombreArchivo,Indice,ColumnasBorrar):
    if not self.CargarArchivo(NombreArchivo):
      return False
    self.VisualizarDatos()
    df_reducido=self.ReducirDataFrame(ColumnasBorrar)  
    df_reducido=df_reducido.set_index(Indice)
    #
    df_reducido=df_reducido.diff(axis=1)
    df_reducido=df_reducido.drop(df_reducido.columns[0],axis=1)
    #
    df_reducido=df_reducido.groupby(df_reducido.index).agg(lambda m:sum(m))
    return df_reducido

Procesar=ProceseArchivo()
xpath="./Data/"
filename_eval = 'time_series_covid19_confirmed_global.csv'
indice="Country/Region"
ColumnsToDelete=[0]+ list(range(2,4))    # Se eliminarán en todos los archivos las columnas 0,2,3
DF=Procesar.ProcesarDatos(xpath+filename_eval,indice,ColumnsToDelete)
## @knitr Item2

print('Archivo reducido para resolver el examen: ')
print(DF)
## @knitr Item 3

# Aqui inicia el examen

## @knitr Item4

def split_data(df):
    countries = ['Costa Rica', 'Guatemala', 'Honduras', 'El Salvador', 'Panama', 'Mexico', 'Belize', 'Canada', 'Cuba', 'Haiti', 'Trinidad and Tobago']
    df_dot = df.loc[countries]
    return df_dot
print(split_data(DF))

## @knitr Item5

def transpose_df(dataf):
    df = split_data(dataf)
    return df.T
dat_tr= transpose_df(DF)
print(dat_tr)
## @knitr Item6

def k_means(dataframe):
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(dataframe)
    centros = np.array(kmeans.cluster_centers_)
    vert = centros[:1,:]
    y  = vert.tolist()[0]
    N = len(y)
    x = range(N)
    width = 1/1.5
    colores = ['#CF8EF8', '#523C81', '#D2D132', '#59C951', '#E5826C']
    null=plt.bar(x, y, width, color=colores)
    null=plt.xticks(range(dataframe.shape[1]), dataframe.columns, rotation=45, fontsize=10)
    plt.title("Centro 1")
    plt.show()
    plt.close()
    
    #centro 2
    vert1 = centros[1:2,:]
    y1  = vert1.tolist()[0]
    N1 = len(y1)
    x1 = range(N1)
    width = 1/1.5
    colores = ['#CF8EF8', '#523C81', '#D2D132', '#59C951', '#E5826C']
    null=plt.bar(x1, y1, width, color=colores)
    null=plt.xticks(range(dataframe.shape[1]), dataframe.columns, rotation=45, fontsize=10)
    plt.title("Centro 2")
    plt.show()
    plt.close()

    # centro 3
    vert2 = centros[2:3,:]
    y2  = vert2.tolist()[0]
    N2 = len(y2)
    x2 = range(N2)
    width = 1/1.5
    colores = ['#CF8EF8', '#523C81', '#D2D132', '#59C951', '#E5826C']
    null=plt.bar(x2, y2, width, color=colores)
    null=plt.xticks(range(dataframe.shape[1]), dataframe.columns, rotation=45, fontsize=10)
    plt.title("Centro 3")
    plt.show()
    plt.close()

k_means(dat_tr)
## @knitr Item7

def groups(dataframe):
    '''Se agrupa por mes y año el dataframe
    Se retorna el valor medio diario de casos
    para cada combinacion de mes y annio'''

    tdf = transpose_df(dataframe)
    months = pd.DatetimeIndex(tdf.index).month
    tdm =tdf.assign(mes_x = months)
    annos = pd.DatetimeIndex(tdf.index).year
    tdm = tdm.assign(year_x = annos)
    covid = tdm.groupby(['mes_x','year_x']).mean()
    return covid
## @knitr Item8
new_DF = groups(DF)
k_means(new_DF)

## @knitr Item9

def plot_heatmap(dataframe):
    sum_casos = dataframe.apply(lambda x : x/max(x))
    heat = sns.heatmap(sum_casos.corr(),
    mask=np.zeros_like(sum_casos.corr()), 
    cmap=sns.diverging_palette(220, 10, as_cmap=True))
    plt.show()
    plt.close()

def box_plot(dataframe):
    box = dataframe.boxplot(column=['Costa Rica', 'Guatemala', 'Honduras', 'El Salvador', 'Panama', 'Mexico', 'Belize', 'Canada', 'Cuba', 'Haiti', 'Trinidad and Tobago'])  
    plt.xticks(rotation=45)
    plt.show()
    plt.close()

## @knitr Item10
plot_heatmap(new_DF)
box_plot(new_DF)