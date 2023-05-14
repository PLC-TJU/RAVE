%% Riemannian Geometry-based Ensemble Learning Classifier(RIGEL)
%  Author: Pan Lincong
%  Edition date: 22 April 2023
%  Need the covariance toolbox: https://github.com/alexandrebarachant/covariancetoolbox

%% Parameter Description
% Inputs
% traindata_task: Task state samples of the training set, 3D matrix: number of channels * number of time points * number of samples.
% traindata_rest: The resting state of the training set, 3D matrix: number of channels * number of time points * number of samples.
% trainlabel: The label of the training set, vector: number of samples * 1.
% testdata_task: the task state samples of the test set, 3D matrix: number of channels * number of time points * number of samples.
% testdata_rest: rest state samples of the test set, 3D matrix: number of channels * number of time points * number of samples.
% testlabel: The label of the test set, vector: number of samples * 1.

% Optional inputs
% sDataTask: all task state samples of the source dataset, 3D matrix: number of channels * number of time points * number of samples.
% sDataRest: all resting samples from the source dataset, 3D matrix: number of channels * number of time points * number of samples.
% sLabel: The corresponding label of all samples in the source dataset, vector: number of samples * 1.

% Output
% testacc: The classification accuracy of test set

function testacc=...
    RIGEL2(traindata_task,traindata_rest,trainlabel,testdata_task,testdata_rest,testlabel,...
    sDataTask,sDataRest,sLabel)

if ~exist('sDataTask','var')
    sDataTask=[];sDataRest=[];sLabel=[];
end

ParaAll=paraSet;
MaxNum=size(ParaAll,1);

labelPre=nan(size(testdata_task,3),MaxNum);
parfor num=1:MaxNum
    para=ParaAll(num,:);
    [~,~,~,~,labelPre(:,num)]=RIGAD...
        (traindata_task,traindata_rest,trainlabel,testdata_task,testdata_rest,testlabel,para,...
        sDataTask,sDataRest,sLabel);
end

Acc1=nan(MaxNum,1);
for num=1:MaxNum
    % Predicted value fusion decision
    labelpre1=mode(labelPre(:,1:num),2);
    Acc1(num)=100*mean(eq(labelpre1,testlabel));
end

testacc = Acc1(MaxNum);

end

