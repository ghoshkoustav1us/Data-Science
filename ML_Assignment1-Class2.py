#!/usr/bin/python3
import re
import random
import numpy as np
#exercise 1
nums =set([1,1,2,3,3,3,4,4]);
print(len(nums));
#exercise2
d ={"john":40, "peter":45};
print(list(d.keys()))
#exercise 3
res1=input();

if (len(re.findall('[A-Z]+',res1)) ==0 or len( re.findall('[$#@]+',res1))==0 or len(re.findall('[a-z]+',res1))==0 or  len(re.findall('[0-9]+',res1))==0):
    print ("wrong pwd");
else:
    if len(res1)<6 or len(res1)>12:
        print("wrong pwd");
    else:
        print("right pwd");
#Exercise 4:
a = [4,7,3,2,5,9] ;
for  i in a:
    print("Element is %d , Position is %d"%(i,a.index(i)))        ;

#exercise 5
s1=input();#put H1e2l3l4o5w6o7r8l9d as input here#
j=0;
word='';
for i in s1:

    if j%2==0:
        word=word+i;
        j=j+1;
    else:
        j=j+1;
        continue;
print(word);

##Exercise 6
s2=input();
print (s2[::-1]);

##Exercise 7
s4='abcdabcdabcd';
listcontain1=set(list(s4));

for i in listcontain1:
    print("%s,%d"%(i,s4.count(i,0,len(s4))));


#Exercise 8
L10=[1,3,6,78,35,55]  ;
L11=[12,24,35,24,88,120,155];
print(set(L10).intersection(L11));

#Exercise 9
L23=[12,24,35,24,88,120,155,88,120,155];
print(np.unique(np.array(L23)));


#Exercise 10
inp_list=[12,24,35,24,88,120,155,120];
j10=int(input());
try:
    for i in inp_list:
        inp_list.remove(j10);
except ValueError:
    print (" Element no longer in  array")

print(inp_list);

#Exercise 11
L22=[0,4,5];
L24=[12,24,35,70,88,120,155];
L25=[]

for i in L22:
        L25.append(L24[i]);
print(set(L24).difference(L25));


#Exercise12
inp_list=[12,24,35,70,88,120,155];
Match_list=[];
for i in inp_list:
        if (i%5==0 or i%7==0 ):
            Match_list.append(i);
inp_list=set(inp_list).difference(Match_list);
print(inp_list);

#Exercise 13
List26=np.array(np.random.randint(1,28,size=5)*35);
print(List26);

#Exercise14
n1=int(input());
x=0.0;
while n1>= 1:
    x+=n1/(n1+1);
    n1-=1;
print(x);


