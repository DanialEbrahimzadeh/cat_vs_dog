%% Test
clc
clear all
close all hidden

load('matlab1.mat');
% result = [ ];
% trnData = load('train_feat1.dta');
% DataTest = load('test_feat1.dta');
% tstData = DataTest;
% 
% trnLbl(1:12500) =-1;
% trnLbl(12501:25000) =1;
% trnLbl = trnLbl';
% 

% 
% rand_ind=randperm(25000);
% for i=1:25000
%     trntemp(i,:)=trnData(rand_ind(i),:);
%     trnlbltemp(i,:)=trnLbl(rand_ind(i),:);
% end
% trnData=trntemp(1:20000,:);
% trnLbl=trnlbltemp(1:20000,:);
% validData=trntemp(20001:25000,:);
% validLbl=trnlbltemp(20001:25000,:);
% 
% trnData=trntemp(1:20000,:);
% trnLbl=trnlbltemp(1:20000,:);
% 
% validData = trntemp(20001:25000,:);
% validLbl = trnlbltemp(20001:25000,:);
trnData=[trnData ; validData];
trnLbl=[trnLbl ; validLbl];
%% lib SVM
C = 0.04;

for i=1:64
    disp(i);
    trnDataT = trnData(:,i);
    tstDataT = tstData(:,i);
    opts = sprintf ('-s 0 -t 2 -r 1 -g 50 -c %f -h 0  -d 10', C);
    
model = svmtrain(trnLbl, trnData, opts); %#ok<SVMTRAIN>
svNum = sum(model.nSV);
    
[trnPLable, trnAccuracy, b1] = svmpredict (trnLbl, trnData, model, '-q');
svmTrnErr = 1 - trnAccuracy(1) / 100;

[validPLable, validAccuracy, b2] = svmpredict (validLbl, validData, model, '-q');
svmTstErr = 1 - validAccuracy(1) / 100;

tstLable = ones(12500,1);    
[tstPLable, tstAccuracy, b3] = svmpredict (tstLable, tstData, model, '-q');
svmTstErr = 1 - tstAccuracy(1) / 100;

result = [result; [i trnAccuracy(1) validAccuracy(1)]];
end
max_b3=max(b3);
min_b3=min(b3);
for i=1:12500
    if b3(i)>=0;
        per_test(i)=((b3(i)/(max_b3+1))*0.5)+0.5;
    else
        per_test(i)=((b3(i)/(min_b3-1))*0.5);
    end
end

for i=1:12500;
    per_final(i,1)=i;
    per_final(i,2)=per_test(i);
end
csvwrite('finalf1t2g50d10c0_04.csv',per_final);

% result = [result; [i trnAccuracy(1) validAccuracy(1)]];
