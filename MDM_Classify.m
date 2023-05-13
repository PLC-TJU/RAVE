%% MDRM
%  Author: Pan Lincong
%  Edition date: 22 April 2023
%  Need the covariance toolbox: https://github.com/alexandrebarachant/covariancetoolbox
function [trainAcc,testAcc]=MDM_Classify(traindata,trainlabel,testdata,testlabel,freqs,times)
    traindata=ERPs_Filter(traindata,freqs,[],times);
    testdata=ERPs_Filter(testdata,freqs,[],times);
    
    covTrain=covariances(traindata);
    covTest=covariances(testdata);
       
    LabelPre = mdm(covTrain,covTrain,trainlabel);
    trainAcc=100*mean(eq(LabelPre,trainlabel));

    LabelPre = mdm(covTest,covTrain,trainlabel);
    testAcc=100*mean(eq(LabelPre,testlabel));

end


