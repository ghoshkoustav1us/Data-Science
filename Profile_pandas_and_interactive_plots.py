import pandas as pd
import pandas_profiling as pp
import pprint
import seaborn as sns
from IPython.display import display,HTML
import matplotlib.pyplot as plt
import plotly
plotly.tools.set_credentials_file(username='Koustav_py', api_key='dt2NjDfjQK2Ql4Uw3bW8')
import cufflinks as cf
df=pd.read_csv('C:\\Users\\ghosh\\.PyCharmEdu2018.2\\config\\scratches\\prisoners.csv' ,header='infer',warn_bad_lines=False,low_memory=True,encoding='utf-8')
pprint.pprint(df.shape[0])
df.describe()
profile1 = df.profile_report(title='Prisoner data Profiling Report')
profile1.to_file(output_file="C:\\Users\\ghosh\\Downloads\\Prisoner_data_profiling.html")
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)
df.iplot(kind='hist' , filename='1_plt.html')
URL='https://plot.ly/~Koustav_py/0'
def plot(URL1):
    from IPython.display import IFrame
    return IFrame(URL1, width='100%', height=500)
i=plot(URL)

