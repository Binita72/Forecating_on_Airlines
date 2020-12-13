#!/usr/bin/env python
# coding: utf-8

# # Importing the required dataset

# In[1]:


import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
import seaborn as sns


# # Loading the dataset

# In[2]:


Airlines = pd.read_excel(r"C:\Users\Binita Mandal\Desktop\finity\forecasting\Airlines+Data.xlsx")


# In[3]:


Airlines


# In[4]:


# First rows 
Airlines.head()


# In[5]:


# Tail
Airlines.tail()


# In[6]:


# Plotting 
Airlines.Passengers.plot()


# In[7]:


Airlines.describe()


# In[8]:


Airlines["Date"] = pd.to_datetime(Airlines.Month,format="%b-%y")
Airlines["Months"] = Airlines.Date.dt.strftime("%b")
Airlines["Year"] = Airlines.Date.dt.strftime("%Y")


# In[9]:


# Heatmap
plt.figure(figsize=(12,8))
heatmap_y_month = pd.pivot_table(data=Airlines,values="Passengers",index="Year",columns="Month",aggfunc="mean",fill_value=0)
sns.heatmap(heatmap_y_month,annot=True,fmt="g") #fmt is format of the grid values


# #### Box plot

# In[10]:


sns.boxplot(x="Months",y="Passengers",data=Airlines)
sns.boxplot(x="Year",y="Passengers",data=Airlines)


# ### Preparing for the dummies

# In[11]:


Month_Dummies = pd.DataFrame(pd.get_dummies(Airlines['Months']))
Airlines1 = pd.concat([Airlines,Month_Dummies],axis = 1)


# In[12]:


Airlines1["t"] = np.arange(1,97)
Airlines1["t_squared"] = Airlines1["t"]*Airlines1["t"]
Airlines1["Log_Passengers"] = np.log(Airlines1["Passengers"])


# # Lineplot

# In[13]:


plt.figure(figsize=(12,3))
sns.lineplot(x="Year",y="Passengers",data=Airlines)


# In[14]:


decompose_ts_add = seasonal_decompose(Airlines.Passengers,period=12)
decompose_ts_add.plot()
plt.show()


# ## Splitting the dataset

# In[15]:


Train = Airlines1.head(84)
Test = Airlines1.tail(12)


# In[16]:


Airlines1.Passengers.plot()


# In[17]:


# Linear Model
import statsmodels.formula.api as smf 
linear_model = smf.ols('Passengers~t',data=Train).fit()
pred_linear =  pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))
rmse_linear = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_linear))**2))
rmse_linear


# In[18]:


# Exponential Model
Exp = smf.ols('Log_Passengers~t',data=Train).fit()
pred_Exp = pd.Series(Exp.predict(pd.DataFrame(Test['t'])))
rmse_Exp = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(np.exp(pred_Exp)))**2))
rmse_Exp


# In[19]:


# Quadratic Model
Quad = smf.ols('Passengers~t+t_squared',data=Train).fit()
pred_Quad = pd.Series(Quad.predict(Test[["t","t_squared"]]))
rmse_Quad = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_Quad))**2))
rmse_Quad


# In[20]:


# Additive seasonality
add_sea = smf.ols('Passengers~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()
pred_add_sea = pd.Series(add_sea.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov']]))
rmse_add_sea = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_add_sea))**2))
rmse_add_sea


# In[21]:


# Additive Seasonality Quadratic
add_sea_Quad = smf.ols('Passengers~t+t_squared+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()
pred_add_sea_quad = pd.Series(add_sea_Quad.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','t','t_squared']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_add_sea_quad))**2))
rmse_add_sea_quad 


# In[22]:


# Multiplicative Seasonality
Mul_sea = smf.ols('Log_Passengers~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data = Train).fit()
pred_Mult_sea = pd.Series(Mul_sea.predict(Test))
rmse_Mult_sea = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(np.exp(pred_Mult_sea)))**2))
rmse_Mult_sea


# In[23]:


# Multiplicative Additive Seasonality
Mul_Add_sea = smf.ols('Log_Passengers~t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data = Train).fit()
pred_Mult_add_sea = pd.Series(Mul_Add_sea.predict(Test))
rmse_Mult_add_sea = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(np.exp(pred_Mult_add_sea)))**2))
rmse_Mult_add_sea 


# In[24]:


# Testing
data = {"MODEL":pd.Series(["rmse_linear","rmse_Exp","rmse_Quad","rmse_add_sea","rmse_add_sea_quad","rmse_Mult_sea","rmse_Mult_add_sea"]),"RMSE_Values":pd.Series([rmse_linear,rmse_Exp,rmse_Quad,rmse_add_sea,rmse_add_sea_quad,rmse_Mult_sea,rmse_Mult_add_sea])}
table_rmse=pd.DataFrame(data)
table_rmse


# ### Predict for new time period

# In[25]:


predict_data = pd.read_excel(r"C:\Users\Binita Mandal\Desktop\finity\forecasting\Airlines+Data_New.xlsx")


# In[26]:


predict_data


# In[27]:


#Build the model on entire data set
model_full = smf.ols('Passengers~t',data=Airlines1).fit()
pred_new  = pd.Series(model_full.predict(predict_data))
pred_new


# In[28]:


predict_data["forecasted_Passengers"] = pd.Series(pred_new)
predict_data


# ### Conclusion:- for this method we can say that Multiplicative Additive Seasonality is the best fit model.

# In[ ]:




