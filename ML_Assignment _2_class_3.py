import io
import os
import re
import string as str1
import pandas as pd
x=os.path.dirname(os.path.abspath('.'));
y=os.getcwd();
print(x);
print(y);
##################################################################
df=pd.read_csv('C:\\Users\\ghosh\\.PyCharmEdu2018.2\\config\\scratches\\bank-data.csv',index_col=False);
#print(df);
list1=list(df['age']);
Min_Age=min(list1);
Max_age=max(list1);
print('Max age of Loan is : %d and Min age of Loan is : %d ',(Max_age,Min_Age));
jobset=set(df.iloc[:,1]);
print(jobset);

###################################################################
INP_COUNTER='START';
initial_list=[];

while(INP_COUNTER!='END'):
        match_found='N';
        print("Enter Data ");
        INP_COUNTER=input();
        if str(INP_COUNTER).upper()=='END':
            break;
        print("Enter the Profession of the Customer");
        input1=str(input()).upper();
        print("Enter the Age of the Customer");
        input2=str(input());
        print("Enter the Marital status of the Customer");
        input3=str(input());
        if bool(input1.isalpha())==False  or bool(input2.isnumeric())==False :
            print('''Invalid Argument Either in Profession or Age of Customer .Profession Would be String (e.g: 'Doctor','Engineer' etc), and Age would be Integer (e.g:20,30,45 etc)''');
            continue;

        for job in jobset:
            if  input1==job.upper() and int(input2)>= Min_Age and int(input2)<=Max_age:
                match_found='Y';
                print('Eligible job');
                initial_list.append([input2,input1,input3.upper(),'yes']);
                break;
            else :
                continue;
        if match_found=='N':
            print("Not Eligible job or Age ");
            ###########optional line - I may or may not put the reject data in output file.In this case I have chosen to insert rejected customer record also.
            initial_list.append([input2,input1,input3.upper(),'No']);


df=df.append(pd.DataFrame(data=initial_list,columns=['age','job','marital','y']),ignore_index=False);
for index,row in  df.iterrows():
    pass
    #print( df['job'].str.contains('entrepreneur',case=True, regex=False));
        #print('job is :'+df.iloc[index,1]);
    #print(df['job'].str.replace('entrepreneur','sponsorE'));
    #print(row);


#print(df);
df.to_csv(path_or_buf='C:\\Users\\ghosh\\.PyCharmEdu2018.2\\config\\scratches\\bank-data_output.csv',index=False);
