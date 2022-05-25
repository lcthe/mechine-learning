clear 
close all

%色泽：青绿0 乌黑1 浅白2
%根蒂：蜷缩0 稍蜷1 硬挺2
%敲声：沉闷0 浊响1 清脆2
%纹理：模糊0 稍糊1 清晰2
%脐部：凹陷0 稍凹1 平坦2
%触感：软粘0 硬滑1        考虑拉普拉斯修正
%好瓜：坏瓜0 好瓜1

%样本
trainSet = [0,0,1,2,0,1,0.697,0.460,1;
            1,0,0,2,0,1,0.774,0.376,1;
            1,0,1,2,0,1,0.634,0.264,1;
            0,0,0,2,0,1,0.608,0.318,1;
            2,0,1,2,0,1,0.556,0.215,1;
            0,1,1,2,1,0,0.403,0.237,1;
            1,1,1,0,1,0,0.481,0.149,1;
            1,1,1,2,1,1,0.437,0.211,1;
            
            1,1,0,1,1,1,0.666,0.091,0;
            0,2,2,2,2,0,0.243,0.267,0;
            2,2,2,0,2,1,0.245,0.057,0;
            2,0,1,0,2,0,0.343,0.099,0;
            0,1,1,1,0,1,0.639,0.161,0;
            2,1,0,1,0,1,0.657,0.198,0;
            1,1,1,2,1,0,0.360,0.370,0;
            2,0,1,0,2,1,0.593,0.042,0;
            0,0,0,1,1,1,0.719,0.103,0;
            ];
        
% 输入测试集
testSet = [0,0,1,2,0,1,0.597,0.160];

%样本个数和属性个数
[rows,cols]=size(trainSet);

%总样本个数
trainNum = rows;
classNum=3;

%% 求正负样本的先验概率
posSampleNum=0;
for i = 1:trainNum
  if( trainSet(i,cols) == 1 )
     posSampleNum = posSampleNum + 1;
  end
end

%正样本个数posSampleNum 负样本个数negSampleNum
%正样本的先验概率posRioClass 负样本的先验概率regRioClass 
%考虑拉普拉斯修正
negSampleNum=trainNum-posSampleNum;
posRioProbability=(posSampleNum+1)/(trainNum+2);
negRioProbability=1-posRioProbability;


%% 求各个属性的条件概率

%先计算正样本的离散属性的条件概率

%正样本的条件概率 预分配内存
posConditionProbability=zeros(classNum,cols-3);

%正样本中各属性值的个数 预分配内存
posClassNum=zeros(classNum,cols-3);

%i cols-3个属性，j 正样本个数 k每个属性的特征个数
for i = 1:cols-3
    for j = 1:posSampleNum
        for k = 1:classNum
            if(trainSet(j,i) == k-1)
                posClassNum(k,i)= posClassNum(k,i) + 1;
            end
            %计算正样本的条件概率 需要拉普拉斯修正
            if(i <=5)
                posConditionProbability(k,i)=(posClassNum(k,i)+1)/(posSampleNum+3);
            else
                posConditionProbability(k,i)=(posClassNum(k,i)+1)/(posSampleNum+2);
            end
        end
    end
end


%计算负样本的离散属性的条件概率

%负样本的条件概率 预分配内存
negConditionProbability=zeros(classNum,cols-3);

%负样本中各属性值的个数 预分配内存
negClassNum=zeros(classNum,cols-3);

%外循环j cols-3个属性，内循环k 负样本个数
for i = 1:cols-3
    for j = posSampleNum+1:trainNum
        for k = 1:classNum
            if(trainSet(j,i) == k-1)
                negClassNum(k,i)= negClassNum(k,i) + 1;
            end
            %计算负样本的条件概率 需要拉普拉斯修正
            if(i<=5)
                negConditionProbability(k,i)=(negClassNum(k,i)+1)/(negSampleNum+3);
            else 
                negConditionProbability(k,i)=(negClassNum(k,i)+1)/(negSampleNum+2);
            end
        end
    end
end

%计算连续属性的条件概率，用概率密度函数（高斯函数）计算

%计算正负样本密度和含糖率的均值和方差
posDensitySum=0;
posSugerSum=0;
negDensitySum=0;
negSugerSum=0;

for i= cols-2 : cols-1
    for j = 1 :  posSampleNum 
        if(i == 7)
     posDensitySum = posDensitySum + trainSet(j,i);
        end
        if(i == 8)
     posSugerSum = posSugerSum + trainSet(j,i);
        end
    end
    for j = posSampleNum+1 :  trainNum 
        if(i == 7)
     negDensitySum = negDensitySum + trainSet(j,i);
        end
        if(i == 8)
     negSugerSum = negSugerSum + trainSet(j,i);
        end
    end
end

%均值
posDensityMean = posDensitySum / posSampleNum;
posSugerMean = posSugerSum / posSampleNum;

negDensityMean = negDensitySum / negSampleNum;
negSugerMean = negSugerSum / negSampleNum;

%样本方差
temp1=0;
temp2=0;
for i = cols-2 : cols-1
    for j = 1 : posSampleNum
         if(i == 7)
        temp1 = temp1+(posDensityMean-trainSet(j,i))^2;
        posDensityVar = temp1 / (posSampleNum-1);
         end
         if(i==8)
        temp2 = temp2+(posSugerMean-trainSet(j,i))^2;
        posSugerVar = temp2 / (posSampleNum-1);
         end
    end
end

temp1=0;
temp2=0;
for i = cols-2 : cols-1
    for j = posSampleNum+1 : trainNum
         if(i == 7)
        temp1 = temp1+(negDensityMean-trainSet(j,i))^2;
        negDensityVar = temp1 / (negSampleNum-1);
         end
         if(i==8)
        temp2 = temp2+(negSugerMean-trainSet(j,i))^2;
        negSugerVar = temp2 / (negSampleNum-1);
         end
    end
end

%概率密度函数（高斯函数）计算连续属性的条件概率
posDensityConditionProbability = (1/(sqrt(2*pi*posDensityVar)))*exp((-1/2)*((testSet(7)-posDensityMean)^2/posDensityVar));
% posDensityConditionProbability = normpdf(testSet(7),posDensityMean,posDensityVar);
posSugerConditionProbability = (1/(sqrt(2*pi*posSugerVar)))*exp((-1/2)*((testSet(8)-posSugerMean)^2/posSugerVar));
negDensityConditionProbability = (1/(sqrt(2*pi*negDensityVar)))*exp((-1/2)*((testSet(7)-negDensityMean)^2/negDensityVar));
negSugerConditionProbability = (1/(sqrt(2*pi*negSugerVar)))*exp((-1/2)*((testSet(8)-negSugerMean)^2/negSugerVar));




%% 计算后验概率
posAfterProbability=1;
negAfterProbability=1;
%离散属性的正负条件概率相乘
for i = 1:6
    posAfterProbability = posAfterProbability * posConditionProbability(testSet(i)+1,i);
    negAfterProbability = negAfterProbability * negConditionProbability(testSet(i)+1,i);
end
%后验概率=先验概率*离散属性的条件概率*连续属性的条件概率
%最好取log将连乘转换为加，防止数值下溢
posAfterProbability = posAfterProbability * posRioProbability * posDensityConditionProbability * posSugerConditionProbability;
negAfterProbability = negAfterProbability * negRioProbability * negDensityConditionProbability * negSugerConditionProbability;

fprintf("正后验概率=%f,负后验概率=%f\t",posAfterProbability,negAfterProbability);

 %% 判断 比较正样本后验概率和负样本后验概率大小
if(posAfterProbability > negAfterProbability)
    fprintf("它系个好瓜");
else
    fprintf("它系个坏瓜")
end
