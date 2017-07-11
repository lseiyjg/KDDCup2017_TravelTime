
# coding: utf-8

# In[1]:

#import pandas,numpy,xgboost,datetime
import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime,timedelta


# In[2]:

def AddDatetime(df):
    df["dt"]=dt=pd.to_datetime(str(2016)+"-"+df["month"].astype(str)+"-"+df["date"].astype(str)            +" "+df["hour"].astype(str)+":"+df["minute"].astype(str))
    df["dt"]=df["dt"].dt.dayofyear*72+df["time_number"]
    return df


# In[3]:

def mape_nparray(y,d):
    c=d
    rec=np.sum(np.abs(y-c)/c)/len(c)
    return rec

def mape_nparray_ln(y,d):
    c=d
    result=np.sum(np.abs(np.power(np.e,y)-np.abs(np.power(np.e,c)))/np.abs(np.power(np.e,c)))/len(c)
    return result

def mape(y,d):
    c=d.get_label()
    result=np.sum(np.abs(y-c)/c)/len(c)
    return "mape",result

def mape_ln(y,d):
    c=d.get_label()
    result=np.sum(np.abs(np.power(np.e,y)-np.abs(np.power(np.e,c)))/np.abs(np.power(np.e,c)))/len(c)
    return "mape",result

def mape_object(y,d):
    g=1.0*np.sign(y-d)/d
    h=1.0/d
    return -g,h


# In[4]:

def DataAverage(df):
    q=df.groupby(["month","date","hour","minute","intersection_id","tollgate_id"])["travel_time"].mean()
    p=df.drop(["travel_time","datetime","starting_time"],axis=1).drop_duplicates().sort(columns=["month","date","hour","minute","intersection_id","tollgate_id"])
    p["travel_time"]=q.values
    cols=list(p)
    cols.insert(2, cols.pop(cols.index('travel_time')))
    p= p.ix[:, cols]
    return p


# In[5]:

#Use KNN Algorithm to select validation data(10-13,10-13,9-29,9-26,10-2,9-25,-10-6)

def SelectTrainAndValidation(data):
    select_time2=((data["hour"]<10)&(data["hour"]>7))|((data["hour"]<19)&(data["hour"]>16))
    select_time1=((data["hour"]<8)&(data["hour"]>5))|((data["hour"]<17)&(data["hour"]>14))
    
    t1=(data["month"]==10)&(data["date"]==13)&select_time2
    t2=(data["month"]==10)&(data["date"]==13)&select_time2
    t3=(data["month"]==9)&(data["date"]==29)&select_time2
    t4=(data["month"]==9)&(data["date"]==26)&select_time2
    t5=(data["month"]==10)&(data["date"]==2)&select_time2
    t6=(data["month"]==9)&(data["date"]==25)&select_time2
    t7=(data["month"]==10)&(data["date"]==6)&select_time2
    
    e1=(data["month"]==10)&(data["date"]==13)&select_time1
    e2=(data["month"]==10)&(data["date"]==13)&select_time1
    e3=(data["month"]==9)&(data["date"]==29)&select_time1
    e4=(data["month"]==9)&(data["date"]==26)&select_time1
    e5=(data["month"]==10)&(data["date"]==22)&select_time1
    e6=(data["month"]==9)&(data["date"]==25)&select_time1
    e7=(data["month"]==10)&(data["date"]==6)&select_time1
    
    dt1=data[t1].copy()
    dt2=data[t2].copy()
    dt3=data[t3].copy()
    dt4=data[t4].copy()
    dt5=data[t5].copy()
    dt6=data[t6].copy()
    dt7=data[t7].copy()
    
    
    de1=data[e1].copy()
    de2=data[e2].copy()
    de3=data[e3].copy()
    de4=data[e4].copy()
    de5=data[e5].copy()
    de6=data[e6].copy()
    de7=data[e7].copy()

    
    dtrain=data[~(t1|t2|t3|t4|t5|t6|t7|e1|e2|e3|e4|e5|e6|e7)].copy()
    dpred=pd.concat([dt1,dt2,dt3,dt4,dt5,dt6,dt7])
    dtest=pd.concat([de1,de2,de3,de4,de5,de6,de7])
    return dtrain,dtest,dpred,data


# In[6]:

def AddFeature_target(data): 
    
    #month,date,hour,minute
    data_starting_time=data["starting_time"].str.split(' ',expand=True)
    data_starting_time_date=data_starting_time[0].str.split('-',expand=True)
    data_starting_time_time=data_starting_time[1].str.split(':',expand=True)
    data["month"]=data_starting_time_date[1].apply(int)
    data["date"]=data_starting_time_date[2].apply(int)
    data["hour"]=data_starting_time_time[0].apply(int)
    data["minute"]=data_starting_time_time[1].apply(int)
    data["minute"]=(data["minute"]/20).apply(int)*20
    data["number_intersection_id"]=data["intersection_id"].replace(["A","B","C"],[1,2,3])
    
    #holiday
    h1=data[data["starting_time"]<"2016-09-15"]
    h2=data[(data["starting_time"]>"2016-09-15")&(data["starting_time"]<"2016-09-18")]
    h3=data[(data["starting_time"]>"2016-09-18")&(data["starting_time"]<"2016-09-30")]
    h4=data[(data["starting_time"]>"2016-09-30")&(data["starting_time"]<"2016-10-08")]
    h5=data[data["starting_time"]>"2016-10-08"]
    h1["holiday"]=0
    h2["holiday"]=1
    h3["holiday"]=0
    h4["holiday"]=1
    h5["holiday"]=0
    data=pd.concat([h1,h2,h3,h4,h5])
    
    #datetime,weekday
    data["datetime"]=pd.to_datetime(data["starting_time"])
    data["weekday"]=data["datetime"].dt.weekday
    
    #one_hot A,B,C
    data["ifA"]=data["intersection_id"].replace(["A","B","C"],[1,0,0])
    data["ifB"]=data["intersection_id"].replace(["A","B","C"],[0,1,0])
    data["ifC"]=data["intersection_id"].replace(["A","B","C"],[0,0,1])

    #one_hot 1,2,3
    data["if1"]=data["tollgate_id"].replace([1,2,3],[1,0,0])
    data["if2"]=data["tollgate_id"].replace([1,2,3],[0,1,0])
    data["if3"]=data["tollgate_id"].replace([1,2,3],[0,0,1])
    
    #one_hot weekday
    data["wk1"]=data["weekday"].replace(range(0,7),[1,0,0,0,0,0,0])
    data["wk2"]=data["weekday"].replace(range(0,7),[0,1,0,0,0,0,0])
    data["wk3"]=data["weekday"].replace(range(0,7),[0,0,1,0,0,0,0])
    data["wk4"]=data["weekday"].replace(range(0,7),[0,0,0,1,0,0,0])
    data["wk5"]=data["weekday"].replace(range(0,7),[0,0,0,0,1,0,0])
    data["wk6"]=data["weekday"].replace(range(0,7),[0,0,0,0,0,1,0])
    data["wk7"]=data["weekday"].replace(range(0,7),[0,0,0,0,0,0,1])
    
    #time_number,date_number
    data["time_number"]=((data["hour"]*3+data["minute"]/20)).astype(int)
    data["date_number"]=data["month"]*31+data["date"]
    data=data.iloc[:,[0,1,2]+range(3,len(data.T))]
    return data


# In[7]:

def AddFeature_rawdata(data):
      
    #month,date,hour,minute
    data_starting_time=data["starting_time"].str.split(' ',expand=True)
    data_starting_time_date=data_starting_time[0].str.split('-',expand=True)
    data_starting_time_time=data_starting_time[1].str.split(':',expand=True)
    data["month"]=data_starting_time_date[1].apply(int)
    data["date"]=data_starting_time_date[2].apply(int)
    data["hour"]=data_starting_time_time[0].apply(int)
    data["minute"]=data_starting_time_time[1].apply(int)
    data["minute"]=(data["minute"]/20).apply(int)*20
    data["number_intersection_id"]=data["intersection_id"].replace(["A","B","C"],[1,2,3])
    
    #holiday
    h1=data[(data["starting_time"]<"2016-09-15")]
    h2=data[(data["starting_time"]>"2016-09-15")&(data["starting_time"]<"2016-09-18")]
    h3=data[(data["starting_time"]>"2016-09-18")&(data["starting_time"]<"2016-09-30")]
    h4=data[(data["starting_time"]>"2016-09-30")&(data["starting_time"]<"2016-10-08")]
    h5=data[data["starting_time"]>"2016-10-08"]
    h1["holiday"]=0
    h2["holiday"]=1
    h3["holiday"]=0
    h4["holiday"]=1
    h5["holiday"]=0
    data=pd.concat([h1,h2,h3,h4,h5])
    
    #datetime,weekday
    data["datetime"]=pd.to_datetime(data["starting_time"])
    data["weekday"]=data["datetime"].dt.weekday
    
    #one_hot A,B,C
    data["ifA"]=data["intersection_id"].replace(["A","B","C"],[1,0,0])
    data["ifB"]=data["intersection_id"].replace(["A","B","C"],[0,1,0])
    data["ifC"]=data["intersection_id"].replace(["A","B","C"],[0,0,1])

    #one_hot 1,2,3
    data["if1"]=data["tollgate_id"].replace([1,2,3],[1,0,0])
    data["if2"]=data["tollgate_id"].replace([1,2,3],[0,1,0])
    data["if3"]=data["tollgate_id"].replace([1,2,3],[0,0,1])
    
    #one_hot weekday
    data["wk1"]=data["weekday"].replace(range(0,7),[1,0,0,0,0,0,0])
    data["wk2"]=data["weekday"].replace(range(0,7),[0,1,0,0,0,0,0])
    data["wk3"]=data["weekday"].replace(range(0,7),[0,0,1,0,0,0,0])
    data["wk4"]=data["weekday"].replace(range(0,7),[0,0,0,1,0,0,0])
    data["wk5"]=data["weekday"].replace(range(0,7),[0,0,0,0,1,0,0])
    data["wk6"]=data["weekday"].replace(range(0,7),[0,0,0,0,0,1,0])
    data["wk7"]=data["weekday"].replace(range(0,7),[0,0,0,0,0,0,1])
    data["time_number"]=((data["hour"]*3+data["minute"]/20)).astype(int)
    data["date_number"]=data["month"]*31+data["date"]
    data=data.iloc[:,[0,1,2]+range(3,len(data.T))]
        
    datatrain,datatest,datapredict,datatransforemed=SelectTrainAndValidation(data)
    datavalidation=datapredict
    q=datavalidation.groupby(["month","date","hour","minute","intersection_id","tollgate_id"])["travel_time"].mean()
    p=datavalidation.drop(["travel_time","datetime","starting_time"],axis=1).drop_duplicates().sort(columns=["month","date","hour","minute","intersection_id","tollgate_id"])
    p["travel_time"]=q.values
    cols=list(p)
    cols.insert(2, cols.pop(cols.index('travel_time')))
    p= p.ix[:, cols]
    datavalidation=p    

    return datatrain,datatest,datavalidation,datatransforemed


# In[8]:

#read data
data1 = pd.read_csv('../RawData/trajectories(table_5)_training2.csv')
data2 = pd.read_csv('../RawData/trajectories(table 5)_test2.csv')
data3 = pd.read_csv('../RawData/trajectories(table 5)_training.csv')

#get training data,validation data
data=pd.concat([data1,data3])
data=data.iloc[:,[0,1,3,5]]
datatrain,datatest,datavalidation,datatransforemed=AddFeature_rawdata(data)


# In[9]:

datavalidation=pd.concat([datavalidation,datavalidation[datavalidation["date"]==13]])


# In[10]:

train=pd.concat([datatrain,datatest])
t=DataAverage(train.copy())
train=pd.concat([t,DataAverage(AddFeature_target(data2).ix[:,list(datatrain)])])

HIGH_TRAIN=500
LOW_TRAIN=10
train=train[(train["travel_time"]>LOW_TRAIN)&(train["travel_time"]<HIGH_TRAIN)]


# In[14]:

# Select Feature
feature=[2]+range(10,16)+[8,9]+[3,-2]+range(16,23)

x=train.copy().iloc[:,feature]
xx=x.copy().drop(['travel_time'],axis=1)
xy=np.log(x['travel_time'])

z=datavalidation.copy().iloc[:,feature]
zx=z.drop(['travel_time'],axis=1)
zy=np.log(z['travel_time'])
z


# In[12]:

#Train in Xgboost
xlf = xgb.XGBRegressor(max_depth=11, 
                        learning_rate=0.1,
                        n_estimators=4000, 
                        silent=True, 
                        objective="reg:linear", 
    
                        gamma=0,
                        min_child_weight=5, 
                        max_delta_step=0, 
                        subsample=0.8, 
                        colsample_bytree=0.9, 
                        colsample_bylevel=1, 
                        reg_alpha=1e0, 
                        reg_lambda=0, 
                        scale_pos_weight=1, 
                        seed=1, 
                        missing=None)
xlf.fit(xx,xy, eval_metric=mape_ln,verbose = True, eval_set = [(zx, zy)],early_stopping_rounds=20)
limit=xlf.best_iteration+1


# # OUT

# In[13]:

# Predict
df_id=pd.DataFrame({"intersection_id":["A","A","B","B","C","C"],             "tollgate_id":[2,3,1,3,1,3]})
tmptime=datetime(2016,10,25)
timeseries=[]
while(tmptime<datetime(2016,11,1)):
    if((tmptime.hour==8)|(tmptime.hour==9)|(tmptime.hour==17)|(tmptime.hour==18)):
        timeseries=timeseries+[str(tmptime)]
    tmptime=tmptime+timedelta(minutes=20)
df_tw=pd.DataFrame({"tw":timeseries})

df_tw["merge"]=1
df_id["merge"]=1
target=pd.merge(df_tw,df_id,on=["merge"])
target["time_window"]="["+target["tw"].astype(str)                        +","+(pd.to_datetime(target["tw"])+np.timedelta64(20,"m")).astype(str)+")"
target=target.iloc[:,[2,3,4]]

target=target.sort(columns=["intersection_id","tollgate_id","time_window"])
target["avg_travel_time"]=1

time_window=target["time_window"].copy()
target["time_window"]=target["time_window"].str.split(',',expand=True)[0]
target["time_window"]=target["time_window"].str.split('[',expand=True)[1]
target=target.rename(columns={"time_window":"starting_time","avg_travel_time":"travel_time"})
target=AddFeature_target(target)
target=target.sort(columns=["intersection_id","tollgate_id","starting_time"])

target.index=range(0,len(target))
target["time_window"]=time_window.values
targetx=target.ix[:,list(datavalidation)].copy().iloc[:,feature].drop(['travel_time'],axis=1)
target["travel_time"]=np.power(np.e,xlf.predict(targetx,ntree_limit=limit))
target=target.rename(columns={"travel_time":"avg_travel_time"})

target1=target[(target["hour"]>0)&(target["hour"]<12)].                sort(columns=["intersection_id","tollgate_id","date","hour","minute"])
target2=target[(target["hour"]>12)&(target["hour"]<24)].                sort(columns=["intersection_id","tollgate_id","date","hour","minute"])
predict=pd.concat([target1,target2])
predict.index=range(0,len(predict))
predict=predict.iloc[:,[0,1,-1,3]]
predict.to_csv("../Submisssion/submission.csv",index=False)
predict


# In[ ]:




# In[ ]:



