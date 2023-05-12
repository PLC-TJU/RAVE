%% Common Time-Frequency-Spatial Patterns(CTFSP)
% Miao Y, Jin J, Daly I, et al. Learning Common Time-Frequency-Spatial Patterns for Motor 
% Imagery Classification. IEEE Trans Neural Syst Rehabil Eng, 2021, 29: 699-707.

% Author: Pan Lincong
% Edition date: 22 April 2023

%% Parameter Description
% input
% traindata: The task state samples of the training set, number of leads * number of time points * number of samples.
% trainlabel: the label of the training set
% testdata: task state samples of the test set, number of leads * number of time points * number of samples.

% optional inputs
% paraSet: spatial-temporal-frequency parameter, cell:3*1
% DAmethod: Data pre-alignment method, ['None'/'EA']
% sData: All task state samples of the source dataset, cell{s} or 3D matrix:number of leads * number of time points * number of samples.
% sLabel: Corresponding label of all samples in the source dataset, cell{s} or 1D vector:number of samples*1.

% output
% trainFeaSelect: training set (including target domain known training data and source domain data) features
% testFeaSelect: test set features

function [trainFeaSelect,testFeaSelect]=CTFSP2(traindata,trainlabel,testdata,paraSet,DAmethod,sData,sLabel)
if nargin<6
    sData=[];
    sLabel=[];
end
if nargin<5
    DAmethod='None';
end

fs=250;
if ~exist('paraSet','var') || isempty(paraSet)
    % This parameter setting is consistent with the original literature
    timewindows=[0,2.5;0.5,3;1,3.5];
    freqsbands=[8,13;8,10;10,13;13,30;13,18;18,23;23,30];
else
    timewindows=nan(size(paraSet,1),2);
    freqsbands=nan(size(paraSet,1),2);
    for i=1:size(paraSet,1)
        timewindows(i,:)=paraSet{i,2};
        freqsbands(i,:) =paraSet{i,3};
    end
    timewindows=unique(timewindows,'rows');
    freqsbands=unique(freqsbands,'rows');
end

trainFea=[];
testFea=[];
for t=1:size(timewindows,1)
    for f=1:size(freqsbands,1)
        tw=timewindows(t,:);
        fb=freqsbands(f,:);    
        if ~isempty(sData)
            if iscell(sData)
                sDataAll=[];sLabAll=[];
                sM=nan(size(sData{1},1),size(sData{1},1),length(sData));
                sCov=cell(length(sData),1);
                for s=1:length(sData)
                    sdata=ERPs_Filter(sData{s},fb,[],tw,fs);
                    switch upper(DAmethod)
                        case 'EA'
                            sCov{s}=covariances(sdata);
                            sM(:,:,s)=mean(sCov{s},3);
                            for i=1:size(sdata,3)
                                sdata(:,:,i)=sM(:,:,s)^(-0.5)*sdata(:,:,i);
                            end
                        otherwise
                    end

                    sDataAll=cat(3,sDataAll,sdata);
                    sLabAll=cat(1,sLabAll,sLabel{s});
                end
                if isempty(traindata) && s>1
                    M=mean(sM,3);
                else
                    M=sM;
                end
            else
                sdata=ERPs_Filter(sData,fb,[],tw,fs);
                switch upper(DAmethod)
                    case 'EA'
                        sCov=covariances(sdata);
                        M=mean(sCov,3);
                        sDataAll=nan(size(sdata));
                        for i=1:size(sdata,3)
                            sDataAll(:,:,i)=M^(-0.5)*sdata(:,:,i);
                        end
                    otherwise
                        sDataAll=sdata;
                end
                sLabAll=sLabel;
            end
        else
            sDataAll=[];sLabAll=[];
        end
        if ~isempty(traindata)
            trainData=ERPs_Filter(traindata,fb,[],tw,fs);
            switch upper(DAmethod)
                case 'EA'
                    Cov=covariances(trainData);
                    M=mean(Cov,3);
                    for i=1:size(trainData,3)
                        trainData(:,:,i)=M^(-0.5)*trainData(:,:,i);
                    end
                otherwise
            end
        else
            trainData=[];trainlabel=[];
        end
        allData=cat(3,trainData,sDataAll);
        allLabel=cat(1,trainlabel,sLabAll);
        testData=ERPs_Filter(testdata,fb,[],tw,fs);   
        switch upper(DAmethod)
            case 'EA'
                for i=1:size(testData,3)
                    testData(:,:,i)=M^(-0.5)*testData(:,:,i);
                end
            otherwise
        end       
        [trainfea,testfea]=CSPfeature(allData,allLabel,testData);
        trainFea=cat(2,trainFea,trainfea);
        testFea=cat(2,testFea,testfea);
    end
end

% LASSO
[B,FitInfo] = lasso(trainFea,allLabel,'CV',5,'Alpha',1,'Standardize',true);
idxMinMSE = FitInfo.IndexMinMSE;
coefMinMSE = B(:,idxMinMSE);
index=find(coefMinMSE);
trainFeaSelect=trainFea(:,index);
testFeaSelect=testFea(:,index);
