%% A classification framework for transfer learning
%  Author: Pan Lincong
%  Edition date: 22 April 2023
%  Need the covariance toolbox: https://github.com/alexandrebarachant/covariancetoolbox

%% Input parameters
% source domain
% sData: source data set all samples, cell{s} or 3D matrix:number of channels * number of time points * number of samples.
% sDataRest: all resting samples in the source dataset, cell{s} or 3D matrix: number of channels * number of time points * number of samples; 
% sLabel: the corresponding label of all samples in the source dataset, cell{s} or vector:number of samples*1.

% target domain training set
% traindata: the samples used for training in the target dataset.
% traindataRest: resting state samples used for training in the target dataset.
% trainlabel: the corresponding labels of the samples used for training in the target dataset, number of samples*1.

% target domain test set
% testdata: the samples used for testing in the target dataset.
% testlabel: the corresponding label of the samples used for testing in the target dataset, number of samples*1.

% other (optional)
% DAmethod: data alignment method, the data alignment method. ['None'/'EA'/'RA'], default is 'None'.
% Cmethod: classification method, the classification method. ['CSP','CTFSP','MDRM','TSLDA'], default is 'CSP'.

function [testAcc,labelPred]=ClassificationFramework(sData,sDataRest,sLabel,traindata,traindataRest,trainlabel,...
                                 testdata,testdataRest,testlabel,DAmethod,Cmethod)
if ~exist('DAmethod','var') || isempty(DAmethod)
    DAmethod='None';
end
if ~exist('Cmethod','var') || isempty(Cmethod)
    Cmethod='CSP';
end

if sum(strcmpi(Cmethod,'CTFSP'))
    DAmethod='None';
end

%% Data pre-alignment
% Source Domain
% sDatAll: all samples in the source dataset, number of channels * number of time points * number of samples.
% sDatAllRest: all samples of the source dataset in resting state, number of channels * number of time points * number of samples.
% sLabAll: corresponding labels of all samples in the source dataset, number of samples * 1.
sDatAll=[];sDatAllRest=[];sLabAll=[];
if ~isempty(sData)
    if iscell(sData)
        sP=nan(size(sData{1},1),size(sData{1},1),length(sData));
        for s=1:length(sData)
            X=sData{s};
            Xrest=sDataRest{s};
            y=sLabel{s};
            [Xtemp,~,sP(:,:,s)]=DataAlignment(DAmethod,X,[],'restdata1',Xrest);
            sDatAll=cat(3,sDatAll,Xtemp);
            sDatAllRest=cat(3,sDatAllRest,Xrest);
            sLabAll=cat(1,sLabAll,y);
        end
        if isempty(traindata) && s>1
            switch DAmethod
                case 'RA'
                    P=mean_covariances(sP,'riemann');
                otherwise
                    P=mean_covariances(sP,'arithmetic');
            end
        else
            P=sP;
        end
    else
        [sDatAll,~,P]=DataAlignment(DAmethod,sData,[],'restdata1',sDataRest);
        sLabAll=sLabel;
    end
end

if ~isempty(traindata)
    [traindata,~,P]=DataAlignment(DAmethod,traindata,[],'restdata1',traindataRest);
end

testdata=DataAlignment(DAmethod,[],testdata,'testFlag',1,'P',P);

traindatall = cat(3,traindata,sDatAll);
traindatallRest = cat(3,traindataRest,sDatAllRest);
trainlaball = cat(1,trainlabel,sLabAll);

%% Classification
switch upper(Cmethod)
    case {'CSP','EA-CSP','EA_CSP','EACSP'}
        [fTrain,fTest]=CSPfeature(traindatall,trainlaball,testdata);
        SVM = fitcsvm(fTrain,trainlaball);
        labelPred=predict(SVM,fTest);
        testAcc=100*mean(eq(labelPred,testlabel));
    case 'CTFSP'
        [fTrain,fTest]=CTFSP2(traindatall,trainlaball,testdata);
        SVM = fitcsvm(fTrain,trainlaball);
        labelPred=predict(SVM,fTest);
        testAcc=100*mean(eq(labelPred,testlabel));
    case {'EA-CTFSP','EACTFSP','EA_CTFSP'}
        [fTrain,fTest]=CTFSP2(traindata,trainlabel,testdata,[],'EA',sDatAll,sLabAll);
        SVM = fitcsvm(fTrain,trainlaball);
        labelPred=predict(SVM,fTest);
        testAcc=100*mean(eq(labelPred,testlabel));
    case {'MDM','MDRM'}
        covTest=covariances(testdata);
        covTrainAll=covariances(traindatall);
        labelPred = mdm(covTest,covTrainAll,trainlaball);
        testAcc=100*mean(eq(labelPred,testlabel));
    case {'RAMDRM','RA-MDRM','RA_MDRM','RAMDM','RA-MDM','RA_MDM'}
        [testAcc,labelPred]=RAMDRM(traindatall,traindatallRest,trainlaball,testdata,testdataRest,testlabel);
    case {'TSLDA','RA-TSLDA','RATSLDA','RA_TSLDA'}
        covTest=covariances(testdata);
        covTrainAll=covariances(traindatall);
        labelPred = tslda(covTest,covTrainAll,trainlaball);
        testAcc=100*mean(eq(labelPred,testlabel));
end

end
