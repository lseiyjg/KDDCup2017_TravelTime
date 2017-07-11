
# coding: utf-8

# In[43]:

import numpy as np
import pandas as pd
import xgboost as xgb
from datetime import datetime,timedelta
from numpy.matlib import repmat
def mape_nparray(y,d):
    c=d
    rec=np.sum(np.abs(y-c)/c)/len(c)
    return rec

def mape(y,d):
    c=d.get_label()
    result=np.sum(np.abs(y-c)/c)/len(c)
    return "mape",result


# In[44]:

tmptime=datetime(2016,7,19)
timeseries=[]
while(tmptime<datetime(2016,11,1)):
    timeseries=timeseries+[str(tmptime)]
    tmptime=tmptime+timedelta(minutes=20)
    
idframe=pd.DataFrame({"intersection_id":["A","A","B","B","C","C"],
              "tollgate_id":[2,3,1,3,1,3],
             })
idframe["hhh"]=1
timeframe=pd.DataFrame(timeseries)
timeframe["hhh"]=1
twframe=pd.merge(idframe,timeframe,how="outer")
twframe=twframe.iloc[:,[0,1,3]]
twframe.rename(columns={0:'datetime'}, inplace = True)
twframe["datetime"]=pd.to_datetime(twframe["datetime"])


# In[45]:

def data_avg(train):
    q=train.groupby(["month","date","hour","minute","intersection_id","tollgate_id"])["travel_time"].mean()
    p=train.drop(["travel_time","datetime","starting_time"],axis=1).drop_duplicates().sort(columns=["month","date","hour","minute","intersection_id","tollgate_id"])
    p["travel_time"]=q.values
    cols=list(p)
    cols.insert(2, cols.pop(cols.index('travel_time')))
    p= p.ix[:, cols]
    return p


# In[46]:

def select_val_knn(data):
    select_time2=((data["hour"]<10)&(data["hour"]>7))|((data["hour"]<19)&(data["hour"]>16))
    select_time1=((data["hour"]<8)&(data["hour"]>5))|((data["hour"]<17)&(data["hour"]>14))
    
    t1=(data["month"]==10)&(data["date"]==25)&select_time2
    t2=(data["month"]==10)&(data["date"]==26)&select_time2
    t3=(data["month"]==10)&(data["date"]==27)&select_time2
    t4=(data["month"]==10)&(data["date"]==28)&select_time2
    t5=(data["month"]==10)&(data["date"]==29)&select_time2
    t6=(data["month"]==10)&(data["date"]==30)&select_time2
    t7=(data["month"]==10)&(data["date"]==31)&select_time2
    
    e1=(data["month"]==10)&(data["date"]==25)&select_time1
    e2=(data["month"]==10)&(data["date"]==26)&select_time1
    e3=(data["month"]==10)&(data["date"]==27)&select_time1
    e4=(data["month"]==10)&(data["date"]==28)&select_time1
    e5=(data["month"]==10)&(data["date"]==29)&select_time1
    e6=(data["month"]==10)&(data["date"]==30)&select_time1
    e7=(data["month"]==10)&(data["date"]==31)&select_time1
    
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


    
    dtrain=data[(data["month"]*31+data["date"])<335].copy()
    dpred=pd.concat([dt1,dt2,dt3,dt4,dt5,dt6,dt7])
    dtest=pd.concat([de1,de2,de3,de4,de5,de6,de7])
    return dtrain,dtest,dpred,data


# In[47]:

def select_val(data):
    select_time2=((data["hour"]<10)&(data["hour"]>7))|((data["hour"]<19)&(data["hour"]>16))
    select_time1=((data["hour"]<8)&(data["hour"]>5))|((data["hour"]<17)&(data["hour"]>14))
    
    t1=(data["month"]==10)&(data["date"]==11)&select_time2
    t2=(data["month"]==10)&(data["date"]==12)&select_time2
    t3=(data["month"]==10)&(data["date"]==13)&select_time2
    t4=(data["month"]==10)&(data["date"]==14)&select_time2
    t5=(data["month"]==10)&(data["date"]==15)&select_time2
    t6=(data["month"]==10)&(data["date"]==16)&select_time2
    t7=(data["month"]==10)&(data["date"]==17)&select_time2
    
    e1=(data["month"]==10)&(data["date"]==11)&select_time1
    e2=(data["month"]==10)&(data["date"]==12)&select_time1
    e3=(data["month"]==10)&(data["date"]==13)&select_time1
    e4=(data["month"]==10)&(data["date"]==14)&select_time1
    e5=(data["month"]==10)&(data["date"]==15)&select_time1
    e6=(data["month"]==10)&(data["date"]==16)&select_time1
    e7=(data["month"]==10)&(data["date"]==17)&select_time1
    
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


    
    dtrain=data[(data["month"]*31+data["date"])<321].copy()
    dpred=pd.concat([dt1,dt2,dt3,dt4,dt5,dt6,dt7])
    dtest=pd.concat([de1,de2,de3,de4,de5,de6,de7])
    return dtrain,dtest,dpred,data


# In[48]:

def data_to_TrainAndVal(data,datao):
      
    data_starting_time=data["starting_time"].str.split(' ',expand=True)
    data_starting_time_date=data_starting_time[0].str.split('-',expand=True)
    data_starting_time_time=data_starting_time[1].str.split(':',expand=True)
    data["month"]=data_starting_time_date[1].apply(int)
    data["date"]=data_starting_time_date[2].apply(int)
    data["hour"]=data_starting_time_time[0].apply(int)
    data["minute"]=data_starting_time_time[1].apply(int)
    data["minute"]=(data["minute"]/20).apply(int)*20
    data["number_intersection_id"]=data["intersection_id"].replace(["A","B","C"],[1,2,3])
    data_std=datao.groupby(["intersection_id","tollgate_id","hour"])["travel_time"].std().reset_index().rename(columns={"travel_time":"travel_time_std"})
    data_mean=datao.groupby(["intersection_id","tollgate_id","hour"])["travel_time"].mean().reset_index().rename(columns={"travel_time":"travel_time_mean"})
    data=pd.merge(data,data_std,on=["intersection_id","tollgate_id","hour"])
    data=pd.merge(data,data_mean,on=["intersection_id","tollgate_id","hour"])
    
    data["holiday_before1"]=1*(((data["month"]==9)&(data["date"]==14))                            |((data["month"]==9)&(data["date"]==30)))
    data["holiday_before2"]=1*(((data["month"]==9)&(data["date"]==15))                            |((data["month"]==10)&(data["date"]==1)))
    data["holiday_after1"]=1*(((data["month"]==9)&(data["date"]==17))                            |((data["month"]==10)&(data["date"]==7)))
    data["holiday_after2"]=1*(((data["month"]==9)&(data["date"]==18))                            |((data["month"]==10)&(data["date"]==8)))
    data["holiday1"]=1*((data["month"]==9)&(data["date"]>15)&(data["date"]<19))
    data["holiday2"]=1*((data["month"]==10)&(data["date"]>0)&(data["date"]<8))
    data["workholiday"]=1*(((data["month"]==9)&(data["date"]==19))                        |((data["month"]==10)&(data["date"]==9))                        |((data["month"]==10)&(data["date"]==10)))
    data["holiday"]=data["holiday1"]+data["holiday2"]
   
    
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
    
    data["time_number"]=data["hour"]*60+data["minute"]
    data["date_number"]=(data["datetime"]-datetime(2016,7,19)).dt.days
    

    dtrain,dtest,dpred,ddata=select_val(data)
    data_validation=dpred
    q=data_validation.groupby(["month","date","hour","minute","intersection_id","tollgate_id"])["travel_time"].mean()
    p=data_validation.drop(["travel_time","datetime","starting_time","time_number"],axis=1).drop_duplicates().sort(columns=["month","date","hour","minute","intersection_id","tollgate_id"])
    p["travel_time"]=q.values
    cols=list(p)
    cols.insert(2, cols.pop(cols.index('travel_time')))
    p= p.ix[:, cols]
    data_validation=p

    

    return dtrain,dtest,data_validation,ddata


# In[50]:

data1 = pd.read_csv('../RawData/trajectories(table 5)_training.csv')
data2 = pd.read_csv('../RawData/trajectories(table 5)_test1.csv')
data=pd.concat([data1,data2])
data=data.iloc[:,[0,1,3,5]]
dtrain,dtest,val,ddata=data_to_TrainAndVal(data,data)
datao=ddata.copy()


# In[51]:

train=pd.concat([dtrain,dtest]).ix[:,list(val)]


# In[52]:

feature=[2]+range(17,19)+[3]+range(19,25)+[4,5]
x=train.copy().iloc[:,feature]
xx=x.copy().drop(['travel_time'],axis=1)
xy=(x['travel_time'])

z=val.copy().iloc[:,feature]
zx=z.drop(['travel_time'],axis=1)
zy=(z['travel_time'])
z


# In[53]:

xlf = xgb.XGBRegressor(max_depth=6, 
                        learning_rate=0.1,
                        n_estimators=4000, 
                        silent=True, 
                        objective="reg:linear", 
    
                        gamma=0,
                        min_child_weight=5, 
                        max_delta_step=0, 
                        subsample=0.8, 
                        colsample_bytree=0.8, 
                        colsample_bylevel=1, 
                        reg_alpha=2, 
                        reg_lambda=0, 
                        scale_pos_weight=1, 
                        seed=1, 
                        missing=None)
xlf.fit(xx,xy, eval_metric=mape,verbose = True, eval_set = [(zx, zy)],early_stopping_rounds=20)
limit=xlf.best_iteration+1


# In[54]:

def adjust(target,a):
    l=target.copy()
    l["travel_time"]=l["travel_time"]                    +a[0]*((l["ifC"]==1)&(l["if3"]==1)&((l["hour"]==8)|(l["hour"]==9)))*l["travel_time"]
    l["travel_time"]=l["travel_time"]                    +a[1]*((l["ifC"]==1)&(l["if3"]==1)&((l["hour"]==17)|(l["hour"]==18)))*l["travel_time"]

    l["travel_time"]=l["travel_time"]                    +a[2]*((l["ifC"]==1)&(l["if1"]==1)&((l["hour"]==8)|(l["hour"]==9)))*l["travel_time"]
    l["travel_time"]=l["travel_time"]                    +a[3]*((l["ifC"]==1)&(l["if1"]==1)&((l["hour"]==17)|(l["hour"]==18)))*l["travel_time"]

    l["travel_time"]=l["travel_time"]                    +a[4]*((l["ifA"]==1)&(l["if2"]==1)&((l["hour"]==8)|(l["hour"]==9)))*l["travel_time"]
    l["travel_time"]=l["travel_time"]                    +a[5]*((l["ifA"]==1)&(l["if2"]==1)&((l["hour"]==17)|(l["hour"]==18)))*l["travel_time"]

    l["travel_time"]=l["travel_time"]                    +a[6]*((l["ifA"]==1)&(l["if3"]==1)&((l["hour"]==8)|(l["hour"]==9)))*l["travel_time"]
    l["travel_time"]=l["travel_time"]                    +a[7]*((l["ifA"]==1)&(l["if3"]==1)&((l["hour"]==17)|(l["hour"]==18)))*l["travel_time"]

    l["travel_time"]=l["travel_time"]                    +a[8]*((l["ifB"]==1)&(l["if1"]==1)&((l["hour"]==8)|(l["hour"]==9)))*l["travel_time"]
    l["travel_time"]=l["travel_time"]                    +a[9]*((l["ifB"]==1)&(l["if1"]==1)&((l["hour"]==17)|(l["hour"]==18)))*l["travel_time"]

    l["travel_time"]=l["travel_time"]                    +a[10]*((l["ifB"]==1)&(l["if3"]==1)&((l["hour"]==8)|(l["hour"]==9)))*l["travel_time"]
    l["travel_time"]=l["travel_time"]                    +a[11]*((l["ifB"]==1)&(l["if3"]==1)&((l["hour"]==17)|(l["hour"]==18)))*l["travel_time"]
    return l


# In[55]:

pred=xlf.predict(zx,ntree_limit=limit)
l=val.copy()
adjust_para=np.array(range(0,12))*0.0
best_para=np.array(range(0,12))*0.0

bestscore=999


for i in range(0,12):
    tmp=list((np.array(range(0,61))-30)/200.0)
    for j in range(0,len(tmp)):
        tmp_para=best_para.copy()
        tmp_para[i]=tmp[j]
        l["travel_time"]=pred
        l=adjust(l,tmp_para)
        score=mape_nparray(l["travel_time"],zy)
        if (score<bestscore):
            bestscore=score
            best_para=tmp_para.copy()

    print(" i= ",i," best= ",bestscore)
    
pred=xlf.predict(zx,ntree_limit=limit)
l=val.copy()
l["travel_time"]=pred
l=adjust(l,best_para)


# In[56]:

def data_to_TrainAndVal_knn(data,datao):
      
    data_starting_time=data["starting_time"].str.split(' ',expand=True)
    data_starting_time_date=data_starting_time[0].str.split('-',expand=True)
    data_starting_time_time=data_starting_time[1].str.split(':',expand=True)
    data["month"]=data_starting_time_date[1].apply(int)
    data["date"]=data_starting_time_date[2].apply(int)
    data["hour"]=data_starting_time_time[0].apply(int)
    data["minute"]=data_starting_time_time[1].apply(int)
    data["minute"]=(data["minute"]/20).apply(int)*20
    data["number_intersection_id"]=data["intersection_id"].replace(["A","B","C"],[1,2,3])
    data_std=datao.groupby(["intersection_id","tollgate_id","hour"])["travel_time"].std().reset_index().rename(columns={"travel_time":"travel_time_std"})
    data_mean=datao.groupby(["intersection_id","tollgate_id","hour"])["travel_time"].mean().reset_index().rename(columns={"travel_time":"travel_time_mean"})
    data=pd.merge(data,data_std,on=["intersection_id","tollgate_id","hour"])
    data=pd.merge(data,data_mean,on=["intersection_id","tollgate_id","hour"])
    
    data["holiday_before1"]=1*(((data["month"]==9)&(data["date"]==14))                            |((data["month"]==9)&(data["date"]==30)))
    data["holiday_before2"]=1*(((data["month"]==9)&(data["date"]==15))                            |((data["month"]==10)&(data["date"]==1)))
    data["holiday_after1"]=1*(((data["month"]==9)&(data["date"]==17))                            |((data["month"]==10)&(data["date"]==7)))
    data["holiday_after2"]=1*(((data["month"]==9)&(data["date"]==18))                            |((data["month"]==10)&(data["date"]==8)))
    data["holiday1"]=1*((data["month"]==9)&(data["date"]>15)&(data["date"]<19))
    data["holiday2"]=1*((data["month"]==10)&(data["date"]>0)&(data["date"]<8))
    data["workholiday"]=1*(((data["month"]==9)&(data["date"]==19))                        |((data["month"]==10)&(data["date"]==9))                        |((data["month"]==10)&(data["date"]==10)))
    data["holiday"]=data["holiday1"]+data["holiday2"]
   
    
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
    
    data["time_number"]=data["hour"]*60+data["minute"]
    data["date_number"]=(data["datetime"]-datetime(2016,7,19)).dt.days
    
#     data=data.iloc[:,[0,1,2]+range(3,len(data.T))]


    dtrain,dtest,dpred,ddata=select_val_knn(data)
    data_validation=dpred
    q=data_validation.groupby(["month","date","hour","minute","intersection_id","tollgate_id"])["travel_time"].mean()
    p=data_validation.drop(["travel_time","datetime","starting_time","time_number"],axis=1).drop_duplicates().sort(columns=["month","date","hour","minute","intersection_id","tollgate_id"])
    p["travel_time"]=q.values
    cols=list(p)
    cols.insert(2, cols.pop(cols.index('travel_time')))
    p= p.ix[:, cols]
    data_validation=p

    

    return dtrain,dtest,data_validation,ddata


# In[57]:

data1 = pd.read_csv('../RawData/trajectories(table 5)_training.csv')
data2 = pd.read_csv('../RawData/trajectories(table_5)_training2.csv')
data3 = pd.read_csv('../RawData/trajectories(table 5)_test2.csv')
data=pd.concat([data1,data2,data3])
data=data.iloc[:,[0,1,3,5]]


dtrain,dtest,val,ddata=data_to_TrainAndVal_knn(data,data)
datao=ddata.copy()


# In[58]:

train=pd.concat([dtrain,dtest])
origin=train.copy()
t=data_avg(origin.copy())
train=t
train["datetime"]=pd.to_datetime(str(2016)+"-"+train["month"].astype(str)+'-'+train["date"].astype(str)            +' '+train["hour"].astype(str)+":"+train["minute"].astype(str))


# In[59]:

tmptw=twframe[twframe["datetime"]<pd.to_datetime(np.array([datetime(2016,10,31,19)]*len(twframe)).astype(str))]


# In[60]:

complete_train=pd.merge(train,tmptw,how="right",on=["intersection_id","tollgate_id","datetime"])


# In[61]:

nan_train=complete_train[complete_train["travel_time"].isnull()]
Nnan_train=complete_train[~(complete_train["travel_time"].isnull())]


# In[62]:

nan_train


# In[63]:

def data_to_TrainAndVal_nan(data,datao):
    

    data=data.copy().drop(["travel_time_std","travel_time_mean"],axis=1)
    data_starting_time=data["datetime"].astype(str).str.split(' ',expand=True)
    data_starting_time_date=data_starting_time[0].str.split('-',expand=True)
    data_starting_time_time=data_starting_time[1].str.split(':',expand=True)
    data["month"]=data_starting_time_date[1].apply(int)
    data["date"]=data_starting_time_date[2].apply(int)
    data["hour"]=data_starting_time_time[0].apply(int)
    data["minute"]=data_starting_time_time[1].apply(int)
    data["minute"]=(data["minute"]/20).apply(int)*20
    data["number_intersection_id"]=data["intersection_id"].replace(["A","B","C"],[1,2,3])
    data_std=datao.groupby(["intersection_id","tollgate_id","hour"])["travel_time"].std().reset_index().rename(columns={"travel_time":"travel_time_std"})
    data_mean=datao.groupby(["intersection_id","tollgate_id","hour"])["travel_time"].mean().reset_index().rename(columns={"travel_time":"travel_time_mean"})
    data=pd.merge(data,data_std,on=["intersection_id","tollgate_id","hour"])
    data=pd.merge(data,data_mean,on=["intersection_id","tollgate_id","hour"])
    
    data["holiday_before1"]=1*(((data["month"]==9)&(data["date"]==14))                            |((data["month"]==9)&(data["date"]==30)))
    data["holiday_before2"]=1*(((data["month"]==9)&(data["date"]==15))                            |((data["month"]==10)&(data["date"]==1)))
    data["holiday_after1"]=1*(((data["month"]==9)&(data["date"]==17))                            |((data["month"]==10)&(data["date"]==7)))
    data["holiday_after2"]=1*(((data["month"]==9)&(data["date"]==18))                            |((data["month"]==10)&(data["date"]==8)))
    data["holiday1"]=1*((data["month"]==9)&(data["date"]>15)&(data["date"]<19))
    data["holiday2"]=1*((data["month"]==10)&(data["date"]>0)&(data["date"]<8))
    data["workholiday"]=1*(((data["month"]==9)&(data["date"]==19))                        |((data["month"]==10)&(data["date"]==9))                        |((data["month"]==10)&(data["date"]==10)))
    data["holiday"]=data["holiday1"]+data["holiday2"]
    
    
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
    
    data["time_number"]=data["hour"]*60+data["minute"]
    data["date_number"]=(data["datetime"]-datetime(2016,7,19)).dt.days
    data=data.iloc[:,[0,1,2]+range(3,len(data.T))]



    

    return data


# In[64]:

nanfill_train=data_to_TrainAndVal_nan(nan_train,datao)


# In[65]:

nanfill_train["travel_time"]=xlf.predict(nanfill_train.ix[:,list(z)].drop(['travel_time'],axis=1),
                                         ntree_limit=limit)


# In[66]:

nanfill_train=adjust(nanfill_train,best_para)


# In[67]:

nanfill_train


# In[68]:

full_train=pd.concat([nanfill_train,Nnan_train]).ix[:,list(val)]
full_train.index=range(0,len(full_train))


# In[69]:

full_train["datetime"]=pd.to_datetime(str(2016)+"-"+full_train["month"].astype(int).astype(str)+'-'+full_train["date"].astype(int).astype(str)            +' '+full_train["hour"].astype(int).astype(str)+":"+full_train["minute"].astype(int).astype(str))


# In[70]:

full_train


# In[71]:

full_train=full_train.sort(columns=["intersection_id","tollgate_id","month","date","hour","minute"])


# In[72]:

full_train


# In[73]:

train_knn=full_train[(full_train["hour"]==6)|(full_train["hour"]==7)|(full_train["hour"]==15)|(full_train["hour"]==16)].copy()
train_knn


# In[74]:

val_knn=data_avg(dtest)
val_knn


# In[75]:

listmonth=[]
listdate=[]


# In[76]:

tmpval=val_knn[val_knn["date"]==25].copy()
tmpval.index=range(0,len(tmpval))
start_time=datetime(2016,7,19)
bestmonth=0
bestday=0
bestscore=999
while(start_time<datetime(2016,10,25)):
    tmpscore=0
    scorelen=0
    for j in range(0,len(tmpval)):
        tmp=tmpval.iloc[j,:]
        tmptrain=train_knn[(train_knn["month"]==start_time.month)&                           (train_knn["date"]==start_time.day)&                           (train_knn["intersection_id"]==tmp["intersection_id"])&                           (train_knn["tollgate_id"]==tmp["tollgate_id"])&                           (train_knn["hour"]==tmp["hour"])&                           (train_knn["minute"]==tmp["minute"])]
        if(len(tmptrain)):
            tmpscore=tmpscore*scorelen/(scorelen+1)+abs(tmptrain["travel_time"].values-tmp["travel_time"])/(scorelen+1)
            scorelen=scorelen+1
            
    if(bestscore>tmpscore):
        bestscore=tmpscore
        bestday=tmptrain["date"]
        bestmonth=tmptrain["month"]
    start_time=start_time+timedelta(days=1)
print(bestmonth.values,"-",bestday.values)
listmonth=listmonth+[int(bestmonth.values[0])]
listdate=listdate+[int(bestday.values[0])]


# In[77]:

tmpval=val_knn[val_knn["date"]==26].copy()
tmpval.index=range(0,len(tmpval))
start_time=datetime(2016,7,19)
bestmonth=0
bestday=0
bestscore=999
while(start_time<datetime(2016,10,25)):
    tmpscore=0
    scorelen=0
    for j in range(0,len(tmpval)):
        tmp=tmpval.iloc[j,:]
        tmptrain=train_knn[(train_knn["month"]==start_time.month)&                           (train_knn["date"]==start_time.day)&                           (train_knn["intersection_id"]==tmp["intersection_id"])&                           (train_knn["tollgate_id"]==tmp["tollgate_id"])&                           (train_knn["hour"]==tmp["hour"])&                           (train_knn["minute"]==tmp["minute"])]
        if(len(tmptrain)):
            tmpscore=tmpscore*scorelen/(scorelen+1)+abs(tmptrain["travel_time"].values-tmp["travel_time"])/(scorelen+1)
            scorelen=scorelen+1
            
    if(bestscore>tmpscore):
        bestscore=tmpscore
        bestday=tmptrain["date"]
        bestmonth=tmptrain["month"]
    start_time=start_time+timedelta(days=1)
print(bestmonth.values,"-",bestday.values)
listmonth=listmonth+[int(bestmonth.values[0])]
listdate=listdate+[int(bestday.values[0])]


# In[78]:

tmpval=val_knn[val_knn["date"]==27].copy()
tmpval.index=range(0,len(tmpval))
start_time=datetime(2016,7,19)
bestmonth=0
bestday=0
bestscore=999
while(start_time<datetime(2016,10,25)):
    tmpscore=0
    scorelen=0
    for j in range(0,len(tmpval)):
        tmp=tmpval.iloc[j,:]
        tmptrain=train_knn[(train_knn["month"]==start_time.month)&                           (train_knn["date"]==start_time.day)&                           (train_knn["intersection_id"]==tmp["intersection_id"])&                           (train_knn["tollgate_id"]==tmp["tollgate_id"])&                           (train_knn["hour"]==tmp["hour"])&                           (train_knn["minute"]==tmp["minute"])]
        if(len(tmptrain)):
            tmpscore=tmpscore*scorelen/(scorelen+1)+abs(tmptrain["travel_time"].values-tmp["travel_time"])/(scorelen+1)
            scorelen=scorelen+1
            
    if(bestscore>tmpscore):
        bestscore=tmpscore
        bestday=tmptrain["date"]
        bestmonth=tmptrain["month"]
    start_time=start_time+timedelta(days=1)
print(bestmonth.values,"-",bestday.values)
listmonth=listmonth+[int(bestmonth.values[0])]
listdate=listdate+[int(bestday.values[0])]


# In[79]:

tmpval=val_knn[val_knn["date"]==28].copy()
tmpval.index=range(0,len(tmpval))
start_time=datetime(2016,7,19)
bestmonth=0
bestday=0
bestscore=999
while(start_time<datetime(2016,10,25)):
    tmpscore=0
    scorelen=0
    for j in range(0,len(tmpval)):
        tmp=tmpval.iloc[j,:]
        tmptrain=train_knn[(train_knn["month"]==start_time.month)&                           (train_knn["date"]==start_time.day)&                           (train_knn["intersection_id"]==tmp["intersection_id"])&                           (train_knn["tollgate_id"]==tmp["tollgate_id"])&                           (train_knn["hour"]==tmp["hour"])&                           (train_knn["minute"]==tmp["minute"])]
        if(len(tmptrain)):
            tmpscore=tmpscore*scorelen/(scorelen+1)+abs(tmptrain["travel_time"].values-tmp["travel_time"])/(scorelen+1)
            scorelen=scorelen+1
            
    if(bestscore>tmpscore):
        bestscore=tmpscore
        bestday=tmptrain["date"]
        bestmonth=tmptrain["month"]
    start_time=start_time+timedelta(days=1)
print(bestmonth.values,"-",bestday.values)
listmonth=listmonth+[int(bestmonth.values[0])]
listdate=listdate+[int(bestday.values[0])]


# In[80]:

tmpval=val_knn[val_knn["date"]==29].copy()
tmpval.index=range(0,len(tmpval))
start_time=datetime(2016,7,19)
bestmonth=0
bestday=0
bestscore=999
while(start_time<datetime(2016,10,25)):
    tmpscore=0
    scorelen=0
    for j in range(0,len(tmpval)):
        tmp=tmpval.iloc[j,:]
        tmptrain=train_knn[(train_knn["month"]==start_time.month)&                           (train_knn["date"]==start_time.day)&                           (train_knn["intersection_id"]==tmp["intersection_id"])&                           (train_knn["tollgate_id"]==tmp["tollgate_id"])&                           (train_knn["hour"]==tmp["hour"])&                           (train_knn["minute"]==tmp["minute"])]
        if(len(tmptrain)):
            tmpscore=tmpscore*scorelen/(scorelen+1)+abs(tmptrain["travel_time"].values-tmp["travel_time"])/(scorelen+1)
            scorelen=scorelen+1
            
    if(bestscore>tmpscore):
        bestscore=tmpscore
        bestday=tmptrain["date"]
        bestmonth=tmptrain["month"]
    start_time=start_time+timedelta(days=1)
print(bestmonth.values,"-",bestday.values)
listmonth=listmonth+[int(bestmonth.values[0])]
listdate=listdate+[int(bestday.values[0])]


# In[81]:

tmpval=val_knn[val_knn["date"]==30].copy()
tmpval.index=range(0,len(tmpval))
start_time=datetime(2016,7,19)
bestmonth=0
bestday=0
bestscore=999
while(start_time<datetime(2016,10,25)):
    tmpscore=0
    scorelen=0
    for j in range(0,len(tmpval)):
        tmp=tmpval.iloc[j,:]
        tmptrain=train_knn[(train_knn["month"]==start_time.month)&                           (train_knn["date"]==start_time.day)&                           (train_knn["intersection_id"]==tmp["intersection_id"])&                           (train_knn["tollgate_id"]==tmp["tollgate_id"])&                           (train_knn["hour"]==tmp["hour"])&                           (train_knn["minute"]==tmp["minute"])]
        if(len(tmptrain)):
            tmpscore=tmpscore*scorelen/(scorelen+1)+abs(tmptrain["travel_time"].values-tmp["travel_time"])/(scorelen+1)
            scorelen=scorelen+1
            
    if(bestscore>tmpscore):
        bestscore=tmpscore
        bestday=tmptrain["date"]
        bestmonth=tmptrain["month"]
    start_time=start_time+timedelta(days=1)
print(bestmonth.values,"-",bestday.values)
listmonth=listmonth+[int(bestmonth.values[0])]
listdate=listdate+[int(bestday.values[0])]


# In[82]:

tmpval=val_knn[val_knn["date"]==31].copy()
tmpval.index=range(0,len(tmpval))
start_time=datetime(2016,7,19)
bestmonth=0
bestday=0
bestscore=999
while(start_time<datetime(2016,10,25)):
    tmpscore=0
    scorelen=0
    for j in range(0,len(tmpval)):
        tmp=tmpval.iloc[j,:]
        tmptrain=train_knn[(train_knn["month"]==start_time.month)&                           (train_knn["date"]==start_time.day)&                           (train_knn["intersection_id"]==tmp["intersection_id"])&                           (train_knn["tollgate_id"]==tmp["tollgate_id"])&                           (train_knn["hour"]==tmp["hour"])&                           (train_knn["minute"]==tmp["minute"])]
        if(len(tmptrain)):
            tmpscore=tmpscore*scorelen/(scorelen+1)+abs(tmptrain["travel_time"].values-tmp["travel_time"])/(scorelen+1)
            scorelen=scorelen+1
            
    if(bestscore>tmpscore):
        bestscore=tmpscore
        bestday=tmptrain["date"]
        bestmonth=tmptrain["month"]
    start_time=start_time+timedelta(days=1)
print(bestmonth.values,"-",bestday.values)
listmonth=listmonth+[int(bestmonth.values[0])]
listdate=listdate+[int(bestday.values[0])]


# In[83]:

pd.DataFrame({"date":listdate,"month":listmonth}).iloc[:,[1,0]].to_csv("../Other/ValidationDay.csv",index=False)


# In[ ]:



