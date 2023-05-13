%% CSP+LDA/SVM
%  Author: Pan Lincong
%  Edition date: 22 April 2023
function [trainAcc,testAcc]=CSP_Classify(traindata,testdata,trainlabel,testlabel,freqs,times,Classifier)
if nargin<7
    Classifier='SVM';
end
if nargin<6 || isempty(times)
    times=[0,4];
end
if nargin<5 || isempty(freqs)
    freqs=[5,32];
end
trainData=ERPs_Filter(traindata,freqs,[],times);
testData=ERPs_Filter(testdata,freqs,[],times);

% csp_filter=CSP(trainData,'label',trainlabel);
% trainfea=CSP(trainData,'csp_filter',csp_filter);
% testfea=CSP(testData,'csp_filter',csp_filter);
[trainfea,testfea]=CSPfeature(trainData,trainlabel,testData);

switch upper(Classifier)
    case 'SVM'
        SVM = fitcsvm(trainfea,trainlabel);
        labelPred=predict(SVM,trainfea);
        trainAcc.SVM=100*mean(eq(labelPred,trainlabel));
        labelPred=predict(SVM,testfea);
        testAcc.SVM=100*mean(eq(labelPred,testlabel));
    case 'LDA'
        LDAModel=ldatrain(trainfea, trainlabel);
        [~,trainAcc.LDA]=ldapredict(trainfea,LDAModel,trainlabel);
        [~,testAcc.LDA]=ldapredict(testfea,LDAModel,testlabel);
end

end
