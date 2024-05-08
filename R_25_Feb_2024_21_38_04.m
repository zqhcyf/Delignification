clc;clear;close all;	
load('R_25_Feb_2024_21_38_04.mat')	
	
data_str=G_out_data.data_path_str ;  %读取数据的路径 	
dataO=readtable(data_str,'VariableNamingRule','preserve'); %读取数据 	
data1=dataO(:,2:end);test_data=table2cell(dataO(1,2:end));	
for i=1:length(test_data)	
      if ischar(test_data{1,i})==1	
          index_la(i)=1;     %char类型	
      elseif isnumeric(test_data{1,i})==1	
          index_la(i)=2;     %double类型	
      else	
        index_la(i)=0;     %其他类型	
    end 	
end	
index_char=find(index_la==1);index_double=find(index_la==2);	
 %% 数值类型数据处理	
 if length(index_double)>=1	
    data_numshuju=table2array(data1(:,index_double));	
    index_double1=index_double;	
	
    index_double1_index=1:size(data_numshuju,2);	
    data_NAN=(isnan(data_numshuju));    %找列的缺失值	
    num_NAN_ROW=sum(data_NAN);	
    index_NAN=num_NAN_ROW>round(0.2*size(data1,1));	
    index_double1(index_NAN==1)=[]; index_double1_index(index_NAN==1)=[];	
    data_numshuju1=data_numshuju(:,index_double1_index);	
    data_NAN1=(isnan(data_numshuju1));  %找行的缺失值	
     num_NAN__COL=sum(data_NAN1');	
     index_NAN1=num_NAN__COL>0;	
     index_double2_index=1:size(data_numshuju,1);	
     index_double2_index(index_NAN1==1)=[];	
     data_numshuju2=data_numshuju1(index_double2_index,:);	
     index_need_last=index_double1;	
 else	
   index_need_last=[];	
  data_numshuju2=[];	
end	
%% 文本类型数据处理	
	
data_shuju=[];	
 if length(index_char)>=1	
  for j=1:length(index_char)	
    data_get=table2array(data1(index_double2_index,index_char(j)));	
    data_label=unique(data_get);	
if j==length(index_char)	
  data_label_str=data_label ;	
 end    	
	
    for NN=1:length(data_label)	
            idx = find(ismember(data_get,data_label{NN,1}));  	
            data_shuju(idx,j)=NN; 	
     end	
  end	
 end	
label_all_last=[index_char,index_need_last];	
[~,label_max]=max(label_all_last);	
 if(label_max==length(label_all_last))	
     str_label=0; %标记输出是否字符类型	
     data_all_last=[data_shuju,data_numshuju2];	
     label_all_last=[index_char,index_need_last];	
 else	
    str_label=1;	
    data_all_last=[data_numshuju2,data_shuju];	
    label_all_last=[index_need_last,index_char];     	
 end	
   data=data_all_last;	
   data_biao_all=data1.Properties.VariableNames;	
   for j=1:length(label_all_last)	
     data_biao{1,j}=data_biao_all{1,label_all_last(j)};	
   end	
	
% 异常值检测	
data=data;	
	
%%  特征处理 特征选择或者降维	
	
  A_data1=data;	
 data_biao1=data_biao;	
 select_feature_num=G_out_data.select_feature_num1;   %特征选择的个数	
index_name=data_biao1;	
print_index_name=[]; 	
 RF_Model = TreeBagger(50,A_data1(:,1:end-1),A_data1(:,end),'Method','regression','OOBPredictorImportance','on'); 	
 imp = RF_Model.OOBPermutedPredictorDeltaError;	
 y_index = index_name; x_index=index_name;  	
 figure;bar(imp);   %  重要性衡量	
 title('Curvature Test'); ylabel('Predictor importance estimates');  xlabel('Predictors');	
 h = gca;h.TickLabelInterpreter = 'none';	
 [~,sort_feature]=sort(imp,'descend'); 	
 index_name=data_biao1; 	
 feature_need_last=sort_feature(1:select_feature_num);	
 for NN=1:length(feature_need_last) 	
   print_index_name{1,NN}=index_name{1,feature_need_last(NN)};	
end 	
disp('选择特征');disp(print_index_name)  	
data_select=[A_data1(:,feature_need_last),A_data1(:,end)];  %经过特征选择后的数据 	
	
	
%% 数据划分	
  x_feature_label=data_select(:,1:end-1);    %x特征	
  y_feature_label=data_select(:,end);          %y标签	
 index_label1=1:(size(x_feature_label,1));	
 index_label=G_out_data.spilt_label_data;  % 数据索引	
 if isempty(index_label)	
     index_label=index_label1;	
 end	
spilt_ri=G_out_data.spilt_rio;  %划分比例 训练集:验证集:测试集	
train_num=round(spilt_ri(1)/(sum(spilt_ri))*size(x_feature_label,1));          %训练集个数	
vaild_num=round((spilt_ri(1)+spilt_ri(2))/(sum(spilt_ri))*size(x_feature_label,1)); %验证集个数	
 %训练集，验证集，测试集	
 train_x_feature_label=x_feature_label(index_label(1:train_num),:);	
 train_y_feature_label=y_feature_label(index_label(1:train_num),:);	
 vaild_x_feature_label=x_feature_label(index_label(train_num+1:vaild_num),:);	
vaild_y_feature_label=y_feature_label(index_label(train_num+1:vaild_num),:);	
 test_x_feature_label=x_feature_label(index_label(vaild_num+1:end),:);	
 test_y_feature_label=y_feature_label(index_label(vaild_num+1:end),:);	
%Zscore 标准化	
%训练集	
 x_mu = mean(train_x_feature_label);  x_sig = std(train_x_feature_label); 	
 train_x_feature_label_norm = (train_x_feature_label - x_mu) ./ x_sig;    % 训练数据标准化	
 y_mu = mean(train_y_feature_label);  y_sig = std(train_y_feature_label); 	
train_y_feature_label_norm = (train_y_feature_label - y_mu) ./ y_sig;    % 训练数据标准化  	
%验证集	
 vaild_x_feature_label_norm = (vaild_x_feature_label - x_mu) ./ x_sig;    %验证数据标准化	
 vaild_y_feature_label_norm=(vaild_y_feature_label - y_mu) ./ y_sig;  %验证数据标准化	
%测试集	
test_x_feature_label_norm = (test_x_feature_label - x_mu) ./ x_sig;    % 测试数据标准化	
 test_y_feature_label_norm = (test_y_feature_label - y_mu) ./ y_sig;    % 训练数据标准化  	
	
%% 参数设置	
num_pop=G_out_data.num_pop1;   %种群数量	
num_iter=G_out_data.num_iter1;   %种群迭代数	
method_mti=G_out_data.method_mti1;   %优化方法	
BO_iter=G_out_data.BO_iter;   %贝叶斯迭代次数	
min_batchsize=G_out_data.min_batchsize;   %batchsize	
max_epoch=G_out_data.max_epoch1;   %maxepoch	
hidden_size=G_out_data.hidden_size1;   %hidden_size	
	
	
%% 算法处理块	
	
	
	
	
	
	
	
t1=clock;	
 disp('MLP回归') 	
 hidden_size=G_out_data.hidden_size1;    %神经网络隐藏层	
  Mdl=newff(train_x_feature_label_norm',train_y_feature_label_norm',hidden_size,{'logsig','tansig','purelin'}); %神经元和激活函数	
  [Mdl,~]=train(Mdl,train_x_feature_label_norm',train_y_feature_label_norm');  %view(Mdl)  查看网络	
 y_train_predict_norm= sim(Mdl,train_x_feature_label_norm'); y_train_predict_norm=y_train_predict_norm';	
 y_vaild_predict_norm = sim(Mdl,vaild_x_feature_label_norm'); y_vaild_predict_norm=y_vaild_predict_norm';	
y_test_predict_norm = sim(Mdl,test_x_feature_label_norm'); y_test_predict_norm=y_test_predict_norm';	
t2=clock;	
 Time=t2(3)*3600*24+t2(4)*3600+t2(5)*60+t2(6)-(t1(3)*3600*24+t1(4)*3600+t1(5)*60+t1(6));       	
	
	
	
 y_train_predict=y_train_predict_norm*y_sig+y_mu;  %反标准化操作 	
 y_vaild_predict=y_vaild_predict_norm*y_sig+y_mu; 	
 y_test_predict=y_test_predict_norm*y_sig+y_mu; 	
 train_y=train_y_feature_label; disp('***************************************************************************************************************')   	
train_MAE=sum(abs(y_train_predict-train_y))/length(train_y) ; disp(['训练集平均绝对误差MAE：',num2str(train_MAE)])	
train_MAPE=sum(abs((y_train_predict-train_y)./train_y))/length(train_y); disp(['训练集平均相对误差MAPE：',num2str(train_MAPE)])	
train_MSE=(sum(((y_train_predict-train_y)).^2)/length(train_y)); disp(['训练集均方根误差MSE：',num2str(train_MSE)]) 	
 train_RMSE=sqrt(sum(((y_train_predict-train_y)).^2)/length(train_y)); disp(['训练集均方根误差RMSE：',num2str(train_RMSE)]) 	
train_R2= 1 - (norm(train_y - y_train_predict)^2 / norm(train_y - mean(train_y))^2);   disp(['训练集均方根误差R2：',num2str(train_R2)]) 	
vaild_y=vaild_y_feature_label;disp('***************************************************************************************************************')	
vaild_MAE=sum(abs(y_vaild_predict-vaild_y))/length(vaild_y) ; disp(['验证集平均绝对误差MAE：',num2str(vaild_MAE)])	
vaild_MAPE=sum(abs((y_vaild_predict-vaild_y)./vaild_y))/length(vaild_y); disp(['验证集平均相对误差MAPE：',num2str(vaild_MAPE)])	
vaild_MSE=(sum(((y_vaild_predict-vaild_y)).^2)/length(vaild_y)); disp(['验证集均方根误差MSE：',num2str(vaild_MSE)])     	
 vaild_RMSE=sqrt(sum(((y_vaild_predict-vaild_y)).^2)/length(vaild_y)); disp(['验证集均方根误差RMSE：',num2str(vaild_RMSE)]) 	
vaild_R2= 1 - (norm(vaild_y - y_vaild_predict)^2 / norm(vaild_y - mean(vaild_y))^2);    disp(['验证集均方根误差R2:  ',num2str(vaild_R2)])			
 test_y=test_y_feature_label;disp('***************************************************************************************************************');   	
test_MAE=sum(abs(y_test_predict-test_y))/length(test_y) ; disp(['测试集平均绝对误差MAE：',num2str(test_MAE)])        	
test_MAPE=sum(abs((y_test_predict-test_y)./test_y))/length(test_y); disp(['测试集平均相对误差MAPE：',num2str(test_MAPE)])	
 test_MSE=(sum(((y_test_predict-test_y)).^2)/length(test_y)); disp(['测试集均方根误差MSE：',num2str(test_MSE)]) 	
 test_RMSE=sqrt(sum(((y_test_predict-test_y)).^2)/length(test_y)); disp(['测试集均方根误差RMSE：',num2str(test_RMSE)]) 	
 test_R2= 1 - (norm(test_y - y_test_predict)^2 / norm(test_y - mean(test_y))^2);   disp(['测试集均方根误差R2：',num2str(test_R2)]) 	
disp(['算法运行时间Time: ',num2str(Time)])	
