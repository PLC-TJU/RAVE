%% Frequency band/time window pre-optimization
%  Author: Pan Lincong
%  Edition date: 22 April 2023
%  Need the covariance toolbox: https://github.com/alexandrebarachant/covariancetoolbox

% Input
% Data: Task state data raw, 3D matrix of "channels*sampling points*samples number".
% Label: label, vector with length of samples

% Optional parameters
% RestData: rest state data raw, "channels*points*samples".
% Algorithm: Algorithm, ['CSP'/'MDRM'/'RAMDRM], default is 'CSP'
% fold: the fold of cross-validation, the default is cross-validation with maximum prime factors, when fold=1 means no cross-validation is used
% flag: cross-validation mode, integer ≥ 0
% flag=0: cross-validation of random sequences, when flag is set to 0, the sample assignment is different for each run, so it should be run several times to take the average.
% flag=n: n>1 integers, fixed Data first n samples as training set (first n/2 of each type), the rest samples as validation set

% Output
% paramSet_sort:list of parameters sorted by importance
% testAcc_sort: the evaluation correctness corresponding to the parameter list

function [paramSet_sort,testAcc_sort]=ParaSelcet(Data,Label,RestData,Algorithm,fold,flag)
if ~exist('Algorithm','var') || isempty(Algorithm)
    Algorithm='CSP';
end
if ~exist('fold','var') || isempty(fold)
    fold=factor(length(Label));
    fold=max(fold);
end
if ~exist('RestData','var') || isempty(RestData)
    RestData=Data;
end
if ~exist('flag','var') || isempty(flag)
    flag=1;
end

paramSet=paraSet;
times=paramSet(:,2);
freqs=paramSet(:,3);

labelType=unique(Label);
MI_left=Data(:,:,Label==labelType(1));
MI_right=Data(:,:,Label==labelType(2));
Rest_left=RestData(:,:,Label==1);
Rest_right=RestData(:,:,Label==2);

trainInd=1:size(MI_left,3);
for cv=1:fold
    if fold>1
        if flag==1
            trainIndtemp=trainInd;
            validIndtemp=[1:length(trainInd)/fold]+(cv-1)*length(trainInd)/fold;
            trainIndtemp(ismember(trainIndtemp,validIndtemp))=[];
        elseif flag==0
            indices = crossvalind('Kfold',length(Label)/2,fold);
            validIndtemp=find(indices==cv);
            trainIndtemp=find(indices~=cv);
        else
            if flag<0 || flag~=fix(flag)
                error('flag cannot be less than 0 and should be an integer!')
            end
            trainIndtemp=1:round(flag/2);
            validIndtemp=round(flag/2)+1:length(Label)/2;
        end
    else
        trainIndtemp=trainInd;
        validIndtemp=trainInd;
    end

    traindata=cat(3,MI_left(:,:,trainIndtemp),MI_right(:,:,trainIndtemp));
    validata=cat(3,MI_left(:,:,validIndtemp),MI_right(:,:,validIndtemp));
    train_restdata=cat(3,Rest_left(:,:,trainIndtemp),Rest_right(:,:,trainIndtemp));
    vali_restdata=cat(3,Rest_left(:,:,validIndtemp),Rest_right(:,:,validIndtemp));
    trainlabel=[ones(length(trainIndtemp),1);2*ones(length(trainIndtemp),1)];
    valilabel=[ones(length(validIndtemp),1);2*ones(length(validIndtemp),1)];

    parfor ind=1:size(paramSet,1)
        switch upper(Algorithm)
            case 'CSP'
                [~,testacc]=CSP_Classify(traindata,validata,trainlabel,valilabel,freqs{ind},times{ind},'SVM');
                testAcc(ind,cv)=testacc.SVM;
            case {'EA-CSP','EACSP'}
                [~,testacc]=EACSP_Classify(traindata,validata,trainlabel,valilabel,freqs{ind},times{ind},'SVM');
                testAcc(ind,cv)=testacc.SVM;
            case {'MDRM','MDM'}
                trainData=ERPs_Filter(traindata,freqs{ind},[],times{ind});
                valiData=ERPs_Filter(validata,freqs{ind},[],times{ind});
                covVali=covariances(valiData);
                covTrain=covariances(trainData);
                labelPred = mdm(covVali,covTrain,trainlabel);
                testAcc(ind,cv)=100*mean(eq(labelPred,valilabel));
            case {'RAMDRM','RA-MDRM','RA_MDRM','RAMDM','RA-MDM','RA_MDM'}
                if fold>1
                    testAcc(ind,cv)=RAMDRM(traindata,train_restdata,trainlabel,validata,vali_restdata,valilabel,freqs{ind},times{ind});
                else
                    % When fold=1, the optimal parameter of RAMDRM is equal to MDRM!
                    trainData=ERPs_Filter(traindata,freqs{ind},[],times{ind});
                    valiData=ERPs_Filter(validata,freqs{ind},[],times{ind});
                    covVali=covariances(valiData);
                    covTrain=covariances(trainData);
                    labelPred = mdm(covVali,covTrain,trainlabel);
                    testAcc(ind,cv)=100*mean(eq(labelPred,valilabel));
                end
            case 'TSLDA'
                trainData=ERPs_Filter(traindata,freqs{ind},[],times{ind});
                valiData=ERPs_Filter(validata,freqs{ind},[],times{ind});
                covVali=covariances(valiData);
                covTrain=covariances(trainData);
                labelPred = tslda(covVali,covTrain,trainlabel);
                testAcc(ind,cv)=100*mean(eq(labelPred,valilabel));
            case {'RA-TSLDA','RATSLDA','RA_TSLDA'}
                trainData=ERPs_Filter(traindata,freqs{ind},[],times{ind});
                valiData=ERPs_Filter(validata,freqs{ind},[],times{ind});
                [trainData,valiData]=DataAlignment('RA',trainData,valiData);
                covVali=covariances(valiData);
                covTrain=covariances(trainData);
                labelPred = tslda(covVali,covTrain,trainlabel);
                testAcc(ind,cv)=100*mean(eq(labelPred,valilabel));
        end
    end
    if flag>1
        break;
    end
end

[testAcc_sort,testAcc_ind]=sort(mean(testAcc,2),'descend');
paramSet_sort=paramSet(testAcc_ind,:);
end

