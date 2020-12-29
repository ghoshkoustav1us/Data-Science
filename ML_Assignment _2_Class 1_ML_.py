import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math as m
import statsmodels.api as sm
from statsmodels.formula.api import ols

df=pd.read_csv('C:\\Users\\ghosh\\.PyCharmEdu2018.2\\config\\scratches\\cereal.csv',header='infer')
print (df.columns);
sugar_log_val=[]
vitamin_log_val=[]
j=0
for i in str(df['sugars'].values.tolist()).replace('[','').replace(']','').strip().split(','):

    try:
        j=m.pow(int(i),2);
        sugar_log_val.append(j);
    except:
        print("too big value ")


for index,rows in df.iterrows():
    vitamin_log_val.append((rows[11]*2))


fig,ax=plt.subplots(nrows=2,ncols=1,sharex=True,figsize=(7,4))

sns.distplot(a=sugar_log_val,kde=False,hist=True,bins=50,ax=ax[0])
sns.distplot(a=vitamin_log_val,kde=False,hist=True,bins=50,ax=ax[1])
plt.show()


decode_mfg_nm={'N': 'Nabisco',
'Q': 'Quaker Oats',
'K': 'Kelloggs',
'R': 'Raslston Purina',
'G': 'General Mills' ,
'P' :'Post' ,
'A':'American Home Foods Products'}
df["mfg_full_nm"]='NA'

# for index,rows in df.iterrows():
#
#     for k,v in decode_mfg_nm.items():
#         if rows[1]==k:
#             df.iloc[index,16]=v
#             break;

df["mfg_full_nm"]=df['mfr'].map(decode_mfg_nm)
print(df)
# cmp_Cnt_Ser=df.groupby('mfg_full_nm').size()
# print(cmp_Cnt_Ser)
# plt.bar(height=cmp_Cnt_Ser.values,x=cmp_Cnt_Ser.index.values,alpha=0.6,color='black')
# plt.show()

sns.countplot(data=df,y='mfg_full_nm',hue='mfg_full_nm' ,orient='v')
plt.grid(True)
plt.show()



# test_data=df.sample(frac=0.25)
# sns.lmplot(data=df,y='rating',x='calories',hue='mfg_full_nm', fit_reg=True,ci=95,scatter=True,col='mfg_full_nm')
#
# plt.show()

pricing_model=ols("rating ~ calories+protein+fat+sodium+fiber+carbo+sugars+potass+vitamins+shelf+weight+cups", data=df).fit()
# summarize our model
pricing_model_summary = pricing_model.summary()
print(pricing_model_summary)

fig = plt.figure(figsize=(15,8));

# pass in the model as the first parameter, then specify the
# predictor variable we want to analyze
fig = sm.graphics.plot_regress_exog(pricing_model, "calories", fig=fig)
plt.show()

x = df[['calories','protein','fat','sodium','fiber','carbo','sugars','potass','vitamins','shelf','weight','cups']]
y = df[['rating']]
fig, ax = plt.subplots(figsize=(10,7))
ax.plot(x.iloc[:,0], pricing_model.fittedvalues, 'g--.', label="OLS")
plt.show()
