%% main
%  Author: Pan Lincong
%  Edition date: 22 April 2023

% Non-cross-session classification - gradually increase the training sample with a fixed sequence
% Calculated only for the target dataset, i.e., using the Session 2 dataset
% Training samples in order: first 30, 60, 90
% Test samples in order: back 90, 60, 30

% Need the covariance toolbox: https://github.com/alexandrebarachant/covariancetoolbox
    
% List of algorithms:
% 1, CSP
% 2. MDRM
% 3. CTFSP
% 4、EA-CSP
% 5, EA-CTFSP
% 6、RA-MDRM
% 7、RIGEL

clear;clc;
filepath='.\dataset';
files=dir([filepath,'\*D2.mat']);

fs=250;
resttime=fs*0+1:fs*3;
tasktime=fs*3+1:fs*7;

trainNum=[30,60,90];
testAcc=nan(7,length(trainNum),length(files));
for subject=1:length(files)
    target=load([filepath,'\',files(subject).name]);
    dataTask=target.data(:,tasktime,:);
    dataRest=target.data(:,resttime,:);
    label=target.label;
    RIGEL=struct();
    RIGELCSP=struct();
    for tN=1:length(trainNum)
        trainInd=1:trainNum(tN);
        testdInd=trainNum(tN)+1:length(label);
        traindataTask=dataTask(:,:,trainInd);
        traindataRest=dataRest(:,:,trainInd);
        trainlabel=label(trainInd);
        testdataTask=dataTask(:,:,testdInd);
        testdataRest=dataRest(:,:,testdInd);
        testlabel=label(testdInd);
        
        DAmethod='None';
        % 1.CSP
        % 2.MDM
        Algorithms={'CSP','MDM'};
        for AlgNum=1:length(Algorithms)
            tic;
            [Para_sort,ValiAcc_sort]=ParaSelcet(traindataTask,trainlabel,traindataRest,Algorithms{AlgNum},[],0);
            traindataTask0=ERPs_Filter(traindataTask,Para_sort{1,3},Para_sort{1,1},Para_sort{1,2});
            traindataRest0=ERPs_Filter(traindataRest,Para_sort{1,3},Para_sort{1,1});
            testdataTask0=ERPs_Filter(testdataTask,Para_sort{1,3},Para_sort{1,1},Para_sort{1,2});
            testdataRest0=ERPs_Filter(testdataRest,Para_sort{1,3},Para_sort{1,1});
            testAcc(AlgNum,tN,subject)=ClassificationFramework([],[],[],traindataTask0,traindataRest0,trainlabel,...
                testdataTask0,testdataRest0,testlabel,DAmethod,Algorithms{AlgNum});
            Info.Para_sort{AlgNum,tN,subject}=Para_sort;
            Info.ValiAcc_sort{AlgNum,tN,subject}=ValiAcc_sort;
            Info.cost(AlgNum,tN,subject)=toc;
        end
        
        % 3.CTFSP
        Algorithm='CTFSP';
        AlgNum=3;
        tic;
        testAcc(AlgNum,tN,subject)=ClassificationFramework([],[],[],traindataTask,traindataRest,trainlabel,...
            testdataTask,testdataRest,testlabel,[],Algorithm);
        Info.Para_sort{AlgNum,tN,subject}=[];
        Info.ValiAcc_sort{AlgNum,tN,subject}=[];
        Info.cost(AlgNum,tN,subject)=toc;
        
        % 4.EA-CSP
        DAmethod='EA';Algorithm='EA-CSP';
        AlgNum=4;
        tic;
        [Para_sort,ValiAcc_sort]=ParaSelcet(traindataTask,trainlabel,traindataRest,Algorithm,[],0);
        traindataTask0=ERPs_Filter(traindataTask,Para_sort{1,3},Para_sort{1,1},Para_sort{1,2});
        traindataRest0=ERPs_Filter(traindataRest,Para_sort{1,3},Para_sort{1,1});
        testdataTask0=ERPs_Filter(testdataTask,Para_sort{1,3},Para_sort{1,1},Para_sort{1,2});
        testdataRest0=ERPs_Filter(testdataRest,Para_sort{1,3},Para_sort{1,1});
        testAcc(AlgNum,tN,subject)=ClassificationFramework([],[],[],traindataTask0,traindataRest0,trainlabel,...
            testdataTask0,testdataRest0,testlabel,DAmethod,Algorithm);
        Info.Para_sort{AlgNum,tN,subject}=Para_sort;
        Info.ValiAcc_sort{AlgNum,tN,subject}=ValiAcc_sort;
        Info.cost(AlgNum,tN,subject)=toc;

        % 5.EA-CTFSP
        Algorithm='EA-CTFSP';
        AlgNum=5;
        tic;
        testAcc(AlgNum,tN,subject)=ClassificationFramework([],[],[],traindataTask,traindataRest,trainlabel,...
            testdataTask,testdataRest,testlabel,[],Algorithm);
        Info.Para_sort{AlgNum,tN,subject}=[];
        Info.ValiAcc_sort{AlgNum,tN,subject}=[];
        Info.cost(AlgNum,tN,subject)=toc;

        % 6.RA-MDRM
        DAmethod='RA';
        Algorithm='RA-MDM';
        AlgNum=6;
        tic;
        [Para_sort,ValiAcc_sort]=ParaSelcet(traindataTask,trainlabel,traindataRest,Algorithm,[],0);
        traindataTask0=ERPs_Filter(traindataTask,Para_sort{1,3},Para_sort{1,1},Para_sort{1,2});
        traindataRest0=ERPs_Filter(traindataRest,Para_sort{1,3},Para_sort{1,1});
        testdataTask0=ERPs_Filter(testdataTask,Para_sort{1,3},Para_sort{1,1},Para_sort{1,2});
        testdataRest0=ERPs_Filter(testdataRest,Para_sort{1,3},Para_sort{1,1});
        testAcc(AlgNum,tN,subject)=ClassificationFramework([],[],[],traindataTask0,traindataRest0,trainlabel,...
            testdataTask0,testdataRest0,testlabel,DAmethod,Algorithm);
        Info.Para_sort{AlgNum,tN,subject}=Para_sort;
        Info.ValiAcc_sort{AlgNum,tN,subject}=ValiAcc_sort;
        Info.cost(AlgNum,tN,subject)=toc;

        % 7.RIGEL
        AlgNum=7;
        Algorithm='RIGEL';
        tic;
        testAcc(AlgNum,tN,subject)=...
            RIGEL2(traindataTask,traindataRest,trainlabel,testdataTask,testdataRest,testlabel);
        Info.Para_sort{AlgNum,tN,subject}=[];
        Info.ValiAcc_sort{AlgNum,tN,subject}=[];
        Info.cost(AlgNum,tN,subject)=toc;

    end
end
save('Result','testAcc','Info');
