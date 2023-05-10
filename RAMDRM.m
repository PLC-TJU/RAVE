%% RA-MDRM
function [testAcc,LabelPre,model]=RAMDRM(traindata,traindata_rest,trainlabel,testdata,testdata_rest,testlabel,freqs,times)
if exist('freqs','var') && exist('times','var')
    traindata=ERPs_Filter(traindata,freqs,[],times);
    testdata=ERPs_Filter(testdata,freqs,[],times);
    if isempty(traindata_rest) && isempty(testdata_rest)
        traindata_rest=ERPs_Filter(traindata_rest,freqs,[],times);
        testdata_rest=ERPs_Filter(testdata_rest,freqs,[],times);
    end
end
model=RAMDRM_Modeling(traindata,traindata_rest,trainlabel);
[LabelPre,testAcc]=RAMDRM_Classify(testdata,testdata_rest,testlabel,model);
end

%%
function model = RAMDRM_Modeling(traindata,traindata_rest,trainlabel)
if ~exist('traindata_rest','var') || isempty(traindata_rest)
    traindata_rest=traindata;
end

method_mean='riemann';

% RA
covRest=covariances(traindata_rest);
M = mean_covariances(covRest,method_mean);

covTrain=covariances(traindata);
for s=1:size(covTrain,3)
    covTrain(:,:,s)=M^(-0.5)*covTrain(:,:,s)*M^(-0.5);
end

% estimation of center
labels = unique(trainlabel);
Nclass = length(labels);
C = cell(Nclass,1);
for i=1:Nclass
    C{i} = mean_covariances(covTrain(:,:,trainlabel==labels(i)),method_mean);
end

model.M=M;
model.C=C;
end

%%
function [prediction,testAcc]=RAMDRM_Classify(testdata,testdata_rest,testlabel,model)
if ~exist('testdata_rest','var') || isempty(testdata_rest)
    testdata_rest=testdata;
end
method_dist = 'riemann';

M=model.M;
C=model.C;
labels = unique(testlabel);

% RA
covTest=covariances(testdata);
covRest=covariances(testdata_rest);
prediction=nan(size(covTest,3),1);
testFea=nan(size(covTest,3),length(C));
for n=1:size(covTest,3)
    % OPS
    M=M^0.5*(M^-0.5*covRest(:,:,n)*M^-0.5)^(1/(n+1))*M^0.5;
    % RA
    covTest(:,:,n)=M^(-0.5)*covTest(:,:,n)*M^(-0.5);
    % Classification
    for i=1:length(C)
        testFea(n,i) = distance(covTest(:,:,n),C{i},method_dist);
    end
    [~,ind] = min(testFea(n,:),[],2);
    prediction(n) = labels(ind);
end
testAcc=100*mean(eq(prediction,testlabel));

end

