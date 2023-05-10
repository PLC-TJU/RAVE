%% CSP
% Koles Z J, Lazar M S, Zhou S Z. Spatial patterns underlying population 
% differences in the background EEG. Brain Topogr, 1990, 2(4): 275-284.

%  Author: Pan Lincong
%  Edition date: 22 April 2023

%% Description of parameters
% Inputs
% xTrain:training set, channels*points*trials or channels*channels*trials
% yTrain:training set label, 1*trials or trials*1
% xTest:test set, channels*points*trials or channels*channels*trials
% nFilters: the number of CSP filter orders, the actual number of output orders is 2*nFilters
% Outputs
% fTrain: training set features, trials*2nFilters
% fTest:test set features, trials*2nFilters
% CovXtrain:sample covariance matrix after training set filtering, 2nFilters*2nFilters*trials
% CovXtest:test set filtered sample covariance matrix, 2nFilters*2nFilters*trials
function [fTrain,fTest,CovXtrain,CovXtest]=CSPfeature(xTrain,yTrain,xTest,nFilters)
if ~exist('nFilters','var') || isempty(nFilters)
    nFilters=3;
end

classType=unique(yTrain);
xTrain0=xTrain(:,:,yTrain==classType(1));
xTrain1=xTrain(:,:,yTrain==classType(2));

if issymmetric(mean(xTrain,3))
    Sigma0=mean_covariances(xTrain0,'arithmetic');
    Sigma1=mean_covariances(xTrain1,'arithmetic');
else
    Sigma0=mean_covariances(covariances(xTrain0),'arithmetic');
    Sigma1=mean_covariances(covariances(xTrain1),'arithmetic');
end

[d,v]=eig(Sigma1\Sigma0);
[~,ids]=sort(diag(v),'descend');
W=d(:,ids([1:nFilters end-nFilters+1:end])); 

fTrain=zeros(size(xTrain,3),size(W,2));
fTest=zeros(size(xTest,3),size(W,2));
CovXtrain=zeros(size(W,2),size(W,2),size(xTrain,3));
CovXtest=zeros(size(W,2),size(W,2),size(xTest,3));
if issymmetric(mean(xTrain,3))
    for i=1:size(xTrain,3)
        CovXtrain(:,:,i)=W'*xTrain(:,:,i)*W;
        fTrain(i,:)=log10(diag(CovXtrain(:,:,i))/trace(CovXtrain(:,:,i)));
    end
    for i=1:size(xTest,3)
        CovXtest(:,:,i)=W'*xTest(:,:,i)*W;
        fTest(i,:)=log10(diag(CovXtest(:,:,i))/trace(CovXtest(:,:,i)));
    end
else
    for i=1:size(xTrain,3)
        CovXtrain(:,:,i)=W'*xTrain(:,:,i)*xTrain(:,:,i)'*W;
        fTrain(i,:)=log10(diag(CovXtrain(:,:,i))/trace(CovXtrain(:,:,i)));
    end
    for i=1:size(xTest,3)
        CovXtest(:,:,i)=W'*xTest(:,:,i)*xTest(:,:,i)'*W;
        fTest(i,:)=log10(diag(CovXtest(:,:,i))/trace(CovXtest(:,:,i)));
    end
end


