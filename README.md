# KDDCup2017_TravelTime
第一个python项目，KDDCup2017,举办于阿里天池，TravelTime部分代码，复赛11名，但未按时间段分模型，混入未来数据进训练集中，最后没有名次。

使用指南

1.运行../../Code/KDD_cup_2017.py
2.生成的../../Submission/submission.csv即为最后一天上传的结果


文件清单

RawData: 所需的原始数据
	trajectories(table 5)_test2.csv 10月25日到10月31日6：00～8：00,15：00～17：00数据
	trajectories(table 5)_training.csv 7月19日到10月17日所有数据
	trajectories(table_5)_training2.csv 10月18日到10月24日所有数据
Document: 使用说明+预测思路
Code: 代码
	KDD_cup_2017.py: 通过原始数据生成预测数据的python版代码，跑10秒钟左右
	KDD_cup_2017.ipynb: 通过原始数据生成预测数据的ipython notebook版代码
        SelectValidation.py 通过原始数据生成KDD_cup_2017.py中作为验证集的7天，生成文件在../../Other，跑1分钟左右
	SelectValidation.ipynb: ipython notebook版代码
Submission: 提交文件
Other： 生成的其他文件
        ValidationDay.csv: 通过KNN选出的最像预测目标日的7天作为验证集


所需库
pandas
numpy
xgboost
datetime



预测思路

1.读取数据


2.选择训练集，验证集

验证集：
根据预测7天预测时段前两小时的数据,即10月25日至10月31日的6：00～8：00,15：00～17：00，选择与之最像的7天的预测时段作为验证集。
“最像”的数学表达即同样时间段的误差绝对值和最小（缺失数据经过第一阶段最好的模型进行补全），即K=1时的KNN算法。
可由SelectValidation.py得到最像10月25日至10月31日的7天
训练集：
抛去验证集其他天的所有数据+验证集天的6：00～8：00,15：00～17：00数据



3.异常数据去除

本地调试得到，删去平均Travel_time大于500,小于10的时间段数据，验证集的分数有明显提高，于是根据这个标准清除异常数据。



4.预测模型
用Xgboost模型，特征为：
	收费口tollgate_id的one_hot
	出入口intersection的one_hot
	节假日holiday
	星期几weekday（0～6）
        星期几的one_hot（wk1～wk7）
        月month
        时间段time_number（以20分钟为一段，一天分为0～71一共72段）
并进行调参



5.输出预测结果



English version is in ../Document
