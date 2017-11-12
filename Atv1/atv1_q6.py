
# coding: utf-8

# In[128]:

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from scipy.stats import pearsonr
get_ipython().magic('matplotlib inline')


# In[129]:

dataframe = pd.read_csv('http://dados.ufrn.br/dataset/a8b897f9-4659-44d4-842e-ac70ae21eb83/resource/067e7cad-934c-4134-a5d5-807915c074b4/download/obras.csv', sep=';', error_bad_lines=False)


# In[130]:

# preparação dos dados
dfValor = dataframe['valor'].str.split(expand=True)[1]
dfValor = dfValor.str.split(",",expand=True)[0]
dfValor = dfValor.str.replace(".", "")
dfValor = pd.to_numeric(dfValor, downcast ="integer")
dataframe['valor'] = dfValor


# ## Obras
# Foi aplicado a função **pearsonr** para calcular o coeficiente de correlação do valor do projeto em relação a quantidade de dias em que a obra é feita.<br>
# O coeficiente foi elevado ao quadrado para extrair o coeficiente de determinação e multiplicado por 100 para ter uma melhor visualização.
# 
# ### A escala do coeficiente vai de 0 a 100, este ficou com 16.7, então mostra que a relação entre o valor de uma obra com a quantidade de dias em que ela fica pronta é muito baixa ou inexistente.
# 
# base de dados: http://dados.ufrn.br/dataset/obras

# In[131]:

cof_det = (pearsonr(dataframe['valor'], dataframe['qtd_dias'])[0]**2)*100
print("coeficiente de determinação =", cof_det)

x_values = dataframe[['valor']]
y_values = dataframe[['qtd_dias']]

#train model on data
body_reg = linear_model.LinearRegression()
body_reg.fit(x_values, y_values)

#visualize results
plt.scatter(x_values, y_values)
plt.plot(x_values, body_reg.predict(x_values))
plt.show()


# ## PRODUTOS DE EXTENSÃO
# Foi aplicado a função **pearsonr** para calcular o coeficiente de correlação do valor do público estimado com a quantidade de discientes no projeto.<br>
# 
# ### A escala do coeficiente vai de 0 a 100, este ficou com 0.02, então mostra que a correlação entre a quantidade de discentes com o público que eles querem atigir é inexistente.
# Base de dados: http://dados.ufrn.br/dataset/produtos-de-extensao

# In[132]:

dataframe = pd.read_csv('http://dados.ufrn.br/dataset/1898890f-b500-4bc3-afa0-2c4742aa4acf/resource/33aa79cf-52fb-4b29-897d-9f196731f8c0/download/produtos-de-extensao.csv', sep=';', error_bad_lines=False)
cof_det = (pearsonr(dataframe['publico_estimado'], dataframe['quantidade_discente'])[0]**2)*100
print("coeficiente de determinação =", cof_det)

x_values = dataframe[['publico_estimado']]
y_values = dataframe[['quantidade_discente']]

#train model on data
body_reg = linear_model.LinearRegression()
body_reg.fit(x_values, y_values)

#visualize results
plt.scatter(x_values, y_values)
plt.plot(x_values, body_reg.predict(x_values))
plt.show()

