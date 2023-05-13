%% RA-MDRM
%  Author: Pan Lincong
%  Edition date: 22 April 2023
%  Need the covariance toolbox: https://github.com/alexandrebarachant/covariancetoolbox
function [trainAcc,testAcc,model,LabelPre]=RA_MDRM_Classify(traindata,trainlabel,testdata,testlabel,freqs,times,testdata_rest)
if nargin<7
    testdata_rest=testdata;
end
traindata=ERPs_Filter(traindata,freqs,[],times);
testdata=ERPs_Filter(testdata,freqs,[],times);
[model,trainAcc]=RAMDRM_Modeling(traindata,trainlabel);
[LabelPre,testAcc]=RAMDRM_Classify(testdata,testlabel,model,testdata_rest);
end

function [Model,TrainAcc]=RAMDRM_Modeling(traindata,trainlabel)
Cov=covariances(traindata);
labelInd=unique(trainlabel)';
try
    if length(labelInd)>=3
        M = mean_covariances(Cov(:,:,trainlabel==labelInd(end)),'riemann');
        MR{length(labelInd)}=M;
    else
        M = mean_covariances(Cov,'riemann');
    end
catch
    M=mean(Cov,3);
end
%RA
for i=1:size(Cov,3)
    C(:,:,i)=M^(-1/2)*Cov(:,:,i)*M^(-1/2);
end

if length(labelInd)>=3
    for class=labelInd(1:end-1)
        MR{class} = mean_covariances(C(:,:,trainlabel==class),'riemann');
    end
else
    for class=labelInd
        MR{class} = mean_covariances(C(:,:,trainlabel==class),'riemann');
    end
end
FeaInd=nchoosek(labelInd,2);
for trial=1:size(C,3)
    for k=1:size(FeaInd,1)
        Fea(trial,k)=real(distance(C(:,:,trial),MR{FeaInd(k,1)},'riemann')-distance(C(:,:,trial),MR{FeaInd(k,2)},'riemann'));
    end
end
SVMmodel = fitcsvm(Fea,trainlabel);
labelPred=predict(SVMmodel,Fea);
TrainAcc=100*mean(eq(labelPred,trainlabel));
Model.M=M;
Model.MR=MR;
Model.SVMmodel=SVMmodel;
end

function [prediction,TestAcc]=RAMDRM_Classify(testdata,testlabel,Model,testdata_rest)
MR=Model.MR;
M=Model.M;
labelInd=1:length(MR);
for i=1:length(MR)
    if isempty(MR{i})
        labelInd(i)=[];
    end
end
FeaInd=nchoosek(labelInd,2);
Fea=zeros(size(testdata,3),size(FeaInd,1));
for trial=1:size(testdata,3)
    X = squeeze(testdata(:, :,trial));
    temp = X * X' / (size(X,2)-1);
    if nargin < 4 || length(labelInd)<3
        M=RiemannCenterUpdate(trial,M,temp);
    else
        Xrest = squeeze(testdata_rest(:, :,trial));
        Cov_Rest = Xrest * Xrest' / (size(Xrest,2)-1);
        M=RiemannCenterUpdate(trial,M,Cov_Rest);
    end
    temp2=M^(-1/2)*temp*M^(-1/2);    
    for k=1:size(FeaInd,1)
        Fea(trial,k)=real(distance(temp2,MR{FeaInd(k,1)},'riemann')-distance(temp2,MR{FeaInd(k,2)},'riemann'));
    end   
end
prediction=predict(Model.SVMmodel,Fea);
TestAcc=100*mean(eq(prediction,testlabel));
end

function M=RiemannCenterUpdate(n,M,C)
M=M^0.5*(M^-0.5*C*M^-0.5)^(1/(n+1))*M^0.5;
end
