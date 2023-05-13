%% Riemannian Geometry-based Adaptive Boosting(RIGAD)
%  Author: Pan Lincong
%  Edition date: 22 April 2023
%  Need the covariance toolbox: https://github.com/alexandrebarachant/covariancetoolbox

%% Description of parameters
% Inputs:
% traindata_task Task state samples of the training set, number of channels * number of time points * number of samples;
% traindata_rest Resting-state samples of the training set, number of channels * number of time points * number of samples;
% trainlabel     Labels of the training set
% testdata_task  Task state samples of the test set, number of channels * number of time points * number of samples.;
% testdata_rest  Resting-state samples of the test set, number of channels * time points * number of samples;
% testlabel      Labels of the test set
% para           cell:3*1:[spatial-temporal-frequency parameter] or previously created model;

% Optional Inputs:
% sDataTask      Source data set of all task state samples, cell{s} or 3D matrix:number of leads * number of time points * number of samples;
% sDataRest      Source data set of all resting state samples, cell{s} or 3D matrix: number of leads * number of time points * number of samples;
% sLabel         The corresponding labels of all samples in the source dataset, cell{s} or vector:number of samples*1;

% Outputs:
% trainAcc       Classification accuracy of the training set;
% sourceAcc      Classification accuracy of the source dataset;
% testAcc        Classification accuracy of the test set;
% model          Classification Model;
% LabelPre       Prediction labels;
% trainFea       Training set features;
% sourceFea      source dataset features;
% testFea        Test set features.

function [trainAcc,sourceAcc,testAcc,model,LabelPre,trainFea,sourceFea,testFea]=RIGAD...
    (traindata_task,traindata_rest,trainlabel,testdata_task,testdata_rest,testlabel,para,...
    sDataTask,sDataRest,sLabel)
if ~exist('sDataTask','var') || isempty(sDataTask)
    sDataTask=[];
    sDataRest=[];
    sLabel=[];
end
if isstruct(para)
    model=para;
    para=model.para;
    flag=1;
else
    flag=0;
end
channels=para{1};
times=para{2};
freqs=para{3};

if ~isempty(traindata_task)
    traindata_task=ERPs_Filter(traindata_task,freqs,channels,times);
    traindata_rest=ERPs_Filter(traindata_rest,freqs,channels);
end

testdata_task=ERPs_Filter(testdata_task,freqs,channels,times);
testdata_rest=ERPs_Filter(testdata_rest,freqs,channels);
if ~isempty(sDataTask)
    if iscell(sDataTask)
        for s=1:length(sDataTask)
            sDataTask{s}=ERPs_Filter(sDataTask{s},freqs,channels,times);
            sDataRest{s}=ERPs_Filter(sDataRest{s},freqs,channels);
        end
    else
        sDataTask=ERPs_Filter(sDataTask,freqs,channels,times);
        sDataRest=ERPs_Filter(sDataRest,freqs,channels);
    end
end
if ~flag
    model=Ada_RAMDRM_Modeling(traindata_task,trainlabel,traindata_rest,sDataTask,sLabel,sDataRest);
    model.para=para;
end
if ~isempty(traindata_task)
    [~,trainAcc,trainFea]=Ada_RAMDRM_Classify(traindata_task,trainlabel,traindata_rest,model);
else
    trainAcc=nan;trainFea=nan;
end
[LabelPre,testAcc,testFea]=Ada_RAMDRM_Classify(testdata_task,testlabel,testdata_rest,model);

if ~isempty(sDataTask)
    if iscell(sDataTask)
        sourceAcc=[];sourceFea=[];
        for s=1:length(sDataTask)
            [~,sourceAccTemp,sourceFeaTemp]=Ada_RAMDRM_Classify(sDataTask{s},sLabel{s},sDataRest{s},model);
            sourceAcc=cat(2,sourceAcc,sourceAccTemp);
            sourceFea=cat(1,sourceFea,sourceFeaTemp);
        end
    else
        [~,sourceAcc,sourceFea]=Ada_RAMDRM_Classify(sDataTask,sLabel,sDataRest,model);
    end    
    sourceAcc=mean(sourceAcc);
else
    sourceFea=nan;
    sourceAcc=nan;
end

model.trainAcc=trainAcc;
model.sourceAcc=sourceAcc;
model.testAcc=testAcc;
end

%%
function [Model,TrainAcc]=Ada_RAMDRM_Modeling(traindata,trainlabel,restdata,sDataTask,sLabel,sDataRest)

if ~exist('sDataTask','var') || isempty(sDataTask)
    sCovAll=[];
    sLabAll=[];
else
    if iscell(sDataTask)
        sCovAll=[];sLabAll=[];
        sM=nan(size(sDataTask{1},1),size(sDataTask{1},1),length(sDataTask));
        for s=1:length(sDataTask)
            X=sDataTask{s};
            Xrest=sDataRest{s};
            y=sLabel{s};
            XrestCov=covariances(Xrest);
            sM(:,:,s)=mean_covariances(XrestCov,'riemann');
            XCov=covariances(X);
            Xtemp=nan(size(XCov));
            for i=1:size(XCov,3)
                Xtemp(:,:,i)=sM(:,:,s)^(-0.5)*XCov(:,:,i)*sM(:,:,s)^(-0.5);
            end
            sCovAll=cat(3,sCovAll,Xtemp);
            sLabAll=cat(1,sLabAll,y);
        end
        if isempty(traindata) && s>1
            M=mean_covariances(sM,'riemann');
        else
            M=sM;
        end
    else
        sCovRest=covariances(sDataRest);
        M=mean_covariances(sCovRest,'riemann');
        sCov=covariances(sDataTask);
        sCovAll=nan(size(sCov));
        for i=1:size(sCov,3)
            sCovAll(:,:,i)=M^(-0.5)*sCov(:,:,i)*M^(-0.5);
        end
        sLabAll=sLabel;
    end
end

% Target domain training set
if ~isempty(traindata)
    CovRest=covariances(restdata);
    M=mean_covariances(CovRest,'riemann');
    Cov=covariances(traindata);
    for i=1:size(Cov,3)
        Cov(:,:,i)=M^(-0.5)*Cov(:,:,i)*M^(-0.5);
    end
end

allCov=cat(3,Cov,sCovAll);
allLabel=cat(1,trainlabel,sLabAll);

[alpha,d1,d2,acc]=AdaBoost_Rieman_2class(allCov,allLabel);
TrainAcc=acc(end);
Model.M=M;
Model.alpha=alpha;
Model.d1=d1;
Model.d2=d2;

end

%%
function [prediction,TestAcc,fea]=Ada_RAMDRM_Classify(testdata,testlabel,restdata,Model)
Cov_Rest=covariances(restdata);
Cov_Test=covariances(testdata);
M=Model.M;
d1=Model.d1;
d2=Model.d2;
alpha=Model.alpha;
fea=zeros(size(testdata,3),length(d1));
for trial=1:size(testdata,3)
    temp=Cov_Test(:, :,trial);
    M=RiemannCenterUpdate(trial,M,Cov_Rest( :, :,trial));
    temp2=M^(-1/2)*temp*M^(-1/2);
    for k=1:length(d1)
        fea(trial,k)=real(distance(temp2,d1{k},'riemann')-distance(temp2,d2{k},'riemann'));
    end
end
Fea=sign(fea);
result=sign(Fea*alpha);
result(result>=0)=2;
result(result<0)=1;
prediction=result;
TestAcc=100*mean(eq(prediction,testlabel));
end

%% AdaBoost
function [alpha,d1,d2,acc]=AdaBoost_Rieman_2class(SampleCov,label)
% transformation labels 1, 2 as -1, +1
Label(label==1,1)=-1;
Label(label==2,1)=1;
% Define the maximum number of rounds/base models T
T=30;
% Initialization weights
W1=zeros(T,length(Label)/2);
W1(1,:)=2/length(Label);
W2=zeros(T,length(Label)/2);
W2(1,:)=2/length(Label);
result=zeros(length(Label),1);
Dm=zeros(length(Label),T);
labelpre=ones(length(Label),T);
alpha=zeros(T,1);
acc=zeros(T,1);
for m=1:T
    % Base model classification
    [d1{m}, num] = AdaBoost_meanTensor(SampleCov(:,:,Label==-1),W1(m,:));
    [d2{m}, num] = AdaBoost_meanTensor(SampleCov(:,:,Label==1),W2(m,:));
    for n=1:length(Label)
        labelpre(n,m)=sign(real(distance(SampleCov(:,:,n),d1{m},'riemann')-distance(SampleCov(:,:,n),d2{m},'riemann')));
    end
    % Accuracy Calculation
    Dm((labelpre(:,m)-Label~=0),m)=1;
    Dm((labelpre(:,m)-Label==0),m)=0;
    err1(m)=W1(m,:)*Dm(Label==-1,m);
    alpha1(m)=0.5*log10((1-err1(m))/err1(m));
    err2(m)=W2(m,:)*Dm(Label==1,m);
    alpha2(m)=0.5*log10((1-err2(m))/err2(m));
    err(m)=(err1(m)+err2(m))/2;
    alpha(m)=0.5*log10((1-err(m))/err(m));
    %Weight update
    Z1=0;Z2=0;
    if err1(m)>0
        labelpre1(:,m)=labelpre(Label==-1,m);
        for i=1:length(Label(Label==-1))
            Z1=Z1+W1(m,i)*exp(-alpha1(m)*labelpre1(i,m)*(-1));
        end
        for i=1:length(Label(Label==-1))
            W1(m+1,i)=W1(m,i)*exp(-alpha1(m)*labelpre1(i,m)*(-1))/Z1;
        end
    else
        W1(m+1,:)=W1(m,:);
    end
    if err2(m)>0
        labelpre2(:,m)=labelpre(Label==1,m);
        for i=1:length(Label(Label==1))
            Z2=Z2+W2(m,i)*exp(-alpha2(m)*labelpre2(i,m)*(1));
        end
        for i=1:length(Label(Label==1))
            W2(m+1,i)=W2(m,i)*exp(-alpha2(m)*labelpre2(i,m)*(1))/Z2;
        end
    else
        W2(m+1,:)=W2(m,:);
    end
    result=result+alpha(m)*labelpre(:,m);
    acc(m)=mean(eq(sign(result),Label));
    if acc(m)==1
        break;
    end
end
    modelnum=find(acc==max(acc));
    alpha=alpha(1:modelnum(1));
    d1=d1(1:modelnum(1));
    d2=d2(1:modelnum(1));
    acc=acc(1:modelnum(1));
end

function [m, num] = AdaBoost_meanTensor(p,w)
m0 = mean(p,3);
t = 1;
epsilon = 1e-8;
[~,d,n] = size(p);
X0 = zeros(d,d);
for i=1:n
    X0 = X0 + logmap(m0,p(:,:,i))*w(i);
end
X = X0;
m = expmap(m0,t*X);
m0 = m;
num = 1;
dist=zeros(1000,1);
dist(1)=norm(X,2);
flag=0;
while(dist(num)>=epsilon && num <1000)
    X = zeros(d,d);
    for i=1:n
        X = X + logmap(m0,p(:,:,i))*w(i);
    end
    m = expmap(m0,t*X);
    if(norm(X,2)>norm(X0,2))
        t = t/2;
        X = X0;
    end
    X0 = X;
    m0 = m;
    num = num+1;
    dist(num)=norm(X,2);
    if num>10
        if dist(num)-dist(num-1)==0
            flag=flag+1;
        else
            flag=0;
        end
        if flag>=5
            break;
        end
    end
end
end

function M=RiemannCenterUpdate(n,M,C)
M=M^0.5*(M^-0.5*C*M^-0.5)^(1/(n+1))*M^0.5;
end