clear 
close all

%ɫ������0 �ں�1 ǳ��2
%���٣�����0 ����1 Ӳͦ2
%����������0 ����1 ���2
%����ģ��0 �Ժ�1 ����2
%�겿������0 �԰�1 ƽ̹2
%���У���ճ0 Ӳ��1        ����������˹����
%�ùϣ�����0 �ù�1

%����
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
        
% ������Լ�
testSet = [0,0,1,2,0,1,0.597,0.160];

%�������������Ը���
[rows,cols]=size(trainSet);

%����������
trainNum = rows;
classNum=3;

%% �������������������
posSampleNum=0;
for i = 1:trainNum
  if( trainSet(i,cols) == 1 )
     posSampleNum = posSampleNum + 1;
  end
end

%����������posSampleNum ����������negSampleNum
%���������������posRioClass ���������������regRioClass 
%����������˹����
negSampleNum=trainNum-posSampleNum;
posRioProbability=(posSampleNum+1)/(trainNum+2);
negRioProbability=1-posRioProbability;


%% ��������Ե���������

%�ȼ�������������ɢ���Ե���������

%���������������� Ԥ�����ڴ�
posConditionProbability=zeros(classNum,cols-3);

%�������и�����ֵ�ĸ��� Ԥ�����ڴ�
posClassNum=zeros(classNum,cols-3);

%i cols-3�����ԣ�j ���������� kÿ�����Ե���������
for i = 1:cols-3
    for j = 1:posSampleNum
        for k = 1:classNum
            if(trainSet(j,i) == k-1)
                posClassNum(k,i)= posClassNum(k,i) + 1;
            end
            %�������������������� ��Ҫ������˹����
            if(i <=5)
                posConditionProbability(k,i)=(posClassNum(k,i)+1)/(posSampleNum+3);
            else
                posConditionProbability(k,i)=(posClassNum(k,i)+1)/(posSampleNum+2);
            end
        end
    end
end


%���㸺��������ɢ���Ե���������

%���������������� Ԥ�����ڴ�
negConditionProbability=zeros(classNum,cols-3);

%�������и�����ֵ�ĸ��� Ԥ�����ڴ�
negClassNum=zeros(classNum,cols-3);

%��ѭ��j cols-3�����ԣ���ѭ��k ����������
for i = 1:cols-3
    for j = posSampleNum+1:trainNum
        for k = 1:classNum
            if(trainSet(j,i) == k-1)
                negClassNum(k,i)= negClassNum(k,i) + 1;
            end
            %���㸺�������������� ��Ҫ������˹����
            if(i<=5)
                negConditionProbability(k,i)=(negClassNum(k,i)+1)/(negSampleNum+3);
            else 
                negConditionProbability(k,i)=(negClassNum(k,i)+1)/(negSampleNum+2);
            end
        end
    end
end

%�����������Ե��������ʣ��ø����ܶȺ�������˹����������

%�������������ܶȺͺ����ʵľ�ֵ�ͷ���
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

%��ֵ
posDensityMean = posDensitySum / posSampleNum;
posSugerMean = posSugerSum / posSampleNum;

negDensityMean = negDensitySum / negSampleNum;
negSugerMean = negSugerSum / negSampleNum;

%��������
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

%�����ܶȺ�������˹�����������������Ե���������
posDensityConditionProbability = (1/(sqrt(2*pi*posDensityVar)))*exp((-1/2)*((testSet(7)-posDensityMean)^2/posDensityVar));
% posDensityConditionProbability = normpdf(testSet(7),posDensityMean,posDensityVar);
posSugerConditionProbability = (1/(sqrt(2*pi*posSugerVar)))*exp((-1/2)*((testSet(8)-posSugerMean)^2/posSugerVar));
negDensityConditionProbability = (1/(sqrt(2*pi*negDensityVar)))*exp((-1/2)*((testSet(7)-negDensityMean)^2/negDensityVar));
negSugerConditionProbability = (1/(sqrt(2*pi*negSugerVar)))*exp((-1/2)*((testSet(8)-negSugerMean)^2/negSugerVar));




%% ����������
posAfterProbability=1;
negAfterProbability=1;
%��ɢ���Ե����������������
for i = 1:6
    posAfterProbability = posAfterProbability * posConditionProbability(testSet(i)+1,i);
    negAfterProbability = negAfterProbability * negConditionProbability(testSet(i)+1,i);
end
%�������=�������*��ɢ���Ե���������*�������Ե���������
%���ȡlog������ת��Ϊ�ӣ���ֹ��ֵ����
posAfterProbability = posAfterProbability * posRioProbability * posDensityConditionProbability * posSugerConditionProbability;
negAfterProbability = negAfterProbability * negRioProbability * negDensityConditionProbability * negSugerConditionProbability;

fprintf("���������=%f,���������=%f\t",posAfterProbability,negAfterProbability);

 %% �ж� �Ƚ�������������ʺ͸�����������ʴ�С
if(posAfterProbability > negAfterProbability)
    fprintf("��ϵ���ù�");
else
    fprintf("��ϵ������")
end
