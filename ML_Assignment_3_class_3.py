import os
import io
import re
import pandas as pd
import numpy as np
from pandas import DataFrame


f1=open('C:\\Users\\ghosh\\.PyCharmEdu2018.2\\config\\scratches\\FairDealCustomerData_1.csv', "r+");
class BlackListedException(Exception):
    def __init__(self, arg):
        self.arg = arg
    def PrintError(self,arg1):
        print("Blacklisted Customer");
        return arg1;
class Customer:

    isblacklisted = 0;

    def __init__(self):
        self.title = ""

    def __str__(self):
        return "Title:" + self.title + " First Name:" + self.fname + " Last Name:" + self.lname + " Blacklisted:" + self.isblacklisted

    def setIsblacklisted(self,isblacklisted):
        self.isblacklisted = isblacklisted

    def isblacklisted(self):
        return self.isblacklisted

    def StoreData(self,fileinput):
        self.fileinput=fileinput;
        list1=[];
        for line in fileinput.readlines():
            list1.append(line.split(','));
            #print(list1);
        Full_list=[];
        for listitem in list1:
            ll=listitem[0];
            #print(ll);
            Black_list_ind=re.sub('\n','',listitem[-1]);
            #print(Black_list_ind);
            Title_First_name=listitem[1];
            Title_First_name_list=Title_First_name.split();
            fn=Title_First_name_list[1];
            #print(fn);
            title=Title_First_name_list[0];
            #print(title);
            Full_list.append([title,fn,ll,Black_list_ind])

            #print(Full_list);
        df=pd.DataFrame(data=Full_list,columns=['Title','FirstName','LastName','isblacklisted']);
        return(df);
    def Check_Blacklist(self,First_name,Last_name,Title,dataframe1):
        self.First_name=First_name;
        self.Last_name=Last_name;
        self.Title=Title;
        self.dataframe1=dataframe1;
        Bl_ind=0;
        for key,value in dataframe1.iterrows():
            try:
                if str(value[1]).strip().upper()==First_name and str(value[2]).strip().upper()==Last_name and str(value[0]).strip().upper()==Title and str(value[3]).strip()=='1':
                        Bl_ind=1;
                        raise BlackListedException("OMG")
                else :
                    continue;
            except BlackListedException as e:
                print (e.arg);
                print( e.PrintError('Customer '+First_name+'  '+Last_name+ ' is a blacklisted customer'));
                return(dataframe1);
                break;
        if Bl_ind==0:
            #df2=;
            dataframe1=dataframe1.append(pd.DataFrame(data={'Title':Title,'FirstName':First_name,'LastName':Last_name,'isblacklisted':'0'},index=np.arange(1,2,1)),ignore_index=True,sort=False);
            return(dataframe1);

 ################################################################Class and Functions are above######################

c=Customer();

DataFrame1=c.StoreData(f1);
print(DataFrame1);
step='Start';
print("Starts Record Insertion");

while str(step).upper()!='END':
    print("Enter Record number ,TYPE 'END' to end the Record Insertion");
    step=input();

    if str(step).upper()=='END':
        break;
    else:
        print("Enter Title,Permissible values are Mr.,Miss.,Mrs. ")
        Title=str(input()).strip().upper();
        print("Enter First_name: Only Alpha Characters please ")
        First_name =str(input()).strip().upper();
        print("Enter Last_name: Only Alpha Characters please  ")
        Last_name =str(input()).strip().upper();
        DataFrame1=c.Check_Blacklist(Title=Title,First_name=First_name,Last_name=Last_name,dataframe1=DataFrame1);
        print(DataFrame1);

