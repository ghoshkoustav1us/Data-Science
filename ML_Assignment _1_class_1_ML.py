import pandas as pd
import matplotlib.pyplot  as plt

df=pd.read_csv('C:\\Users\\ghosh\\.PyCharmEdu2018.2\\config\\scratches\\prisoners.csv',names=
['STATE/UT','YEAR','NOIBB_Elementary Education','NOIBB_Adult Education',''
'NOIBB_Higher Education','NOIBB_Computer Course'],header=None)

print(df.head());
print(df.tail());
print(df.describe())

print(df.columns);


no_inmate_df=df.loc[(df['NOIBB_Elementary Education']==0) & (df['NOIBB_Adult Education']==0) & (df['NOIBB_Higher Education']==0) &(df['NOIBB_Computer Course']==0) ]
print(no_inmate_df)

sum_totals_rowwise=df.iloc[::,2:6:1].sum(axis=1)
print(sum_totals_rowwise)
sum_totals_colwise_dict=df.iloc[::,2:6:1].sum(axis=0).to_dict();

print(sum_totals_colwise_dict)

df2=pd.DataFrame(data=sum_totals_colwise_dict,columns=sum_totals_colwise_dict.keys(),index=[1])

df['total_benefitted']=sum_totals_rowwise;
df3=pd.concat([df,df2],ignore_index=True,join='outer',sort=True,axis=0,keys=sum_totals_colwise_dict.keys())
#df3=pd.merge(df,df2,how='outer',left_index=True,right_index=True,sort=True)
df3.reset_index(drop=True)

print(df3.iloc[35,::]);

df[['STATE/UT','total_benefitted']].plot(x='STATE/UT',y='total_benefitted',kind='bar',legend=True,color='Red');
plt.show();
#list1=[]
#for index,rows in df3.iterrows():
 #   if index==len(df.index):
 #       for j in rows:
 #           list1.append(j)

ser4= str(df3.iloc[-1::,0:4].values.tolist()).replace('[','').replace(']','').split(',');
print(ser4)

plt.pie(ser4,
        explode=[0.3,0,0,0],autopct='%1.1f%%',labels=['NOIBB_Adult Education','NOIBB_Computer Course','NOIBB_Elementary Education','NOIBB_Higher Education'],
        shadow=True, startangle=45)
plt.show()




