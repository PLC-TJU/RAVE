%% main
%  Author: Pan Lincong
%  Edition date: 22 April 2023

% Cross-session classification - gradually increase training samples with fixed sequences
% Source domain data: Session 1
% Target domain data: Session 2 
% Training samples in order: source domain data + first 0, 30, 60, 90 of target domain
% Test samples in order: 120, 90, 60, 30 after the target domain

% Need the covariance toolbox: https://github.com/alexandrebarachant/covariancetoolbox

% Algorithm list:
% 1. CSP
% 2. MDRM
% 3. CTFSP
% 4. EA-CSP
% 5. EA-CTFSP
% 6. RA-MDRM
% 7. RIGEL

clc;
filepath='.\dataset';
files=dir([filepath,'\*.mat']);

fs=250;
resttime=fs*0+1:fs*3;
tasktime=fs*3+1:fs*7;

trainNum=[0,30,60,90];
testAcc=nan(7,length(trainNum),length(files)/2);
for subject=1:length(files)/2
    source=load([filepath,'\',files(subject*2-1).name]);
    sdataTask=source.data(:,tasktime,:);
    sdataRest=source.data(:,resttime,:);
    slabel=source.label;
    target=load([filepath,'\',files(subject*2).name]);
    tdata=target.data;
    tlabel=target.label;
    RIGEL=struct();
    RIGELCSP=struct();
    for tN=1:length(trainNum)
        if trainNum(tN)==0
            traindataTask=[];
            traindataRest=[];
            trainlabel=[];
        else
            traindataTask=tdata(:,tasktime,1:trainNum(tN));
            traindataRest=tdata(:,resttime,1:trainNum(tN));
            trainlabel=tlabel(1:trainNum(tN));
        end
        testdataTask=tdata(:,tasktime,trainNum(tN)+1:end);
        testdataRest=tdata(:,resttime,trainNum(tN)+1:end);
        testlabel=tlabel(trainNum(tN)+1:end);
        
        DAmethod='None';
        % 1.CSP
        % 2.MDRM
        Algorithms={'CSP','MDRM'};
        for AlgNum=1:length(Algorithms)
            tic;
            if trainNum(tN)==0
                [Para_sort,ValiAcc_sort]=ParaSelcet(sdataTask,slabel,[],Algorithms{AlgNum});
                traindataTask0=[];
                traindataRest0=[];
            else
                [Para_sort,ValiAcc_sort]=ParaSelcet(cat(3,sdataTask,traindataTask),[slabel;trainlabel],cat(3,sdataRest,traindataRest),Algorithms{AlgNum},[],size(sdataTask,3));
                traindataTask0=ERPs_Filter(traindataTask,Para_sort{1,3},Para_sort{1,1},Para_sort{1,2});
                traindataRest0=ERPs_Filter(traindataRest,Para_sort{1,3},Para_sort{1,1});
            end
            sdataTask0=ERPs_Filter(sdataTask,Para_sort{1,3},Para_sort{1,1},Para_sort{1,2});
            sdataRest0=ERPs_Filter(sdataRest,Para_sort{1,3},Para_sort{1,1});
            testdataTask0=ERPs_Filter(testdataTask,Para_sort{1,3},Para_sort{1,1},Para_sort{1,2});
            testdataRest0=ERPs_Filter(testdataRest,Para_sort{1,3},Para_sort{1,1});
            testAcc(AlgNum,tN,subject)=ClassificationFramework(sdataTask0,sdataRest0,slabel,traindataTask0,traindataRest0,trainlabel,...
                testdataTask0,testdataRest0,testlabel,DAmethod,Algorithms{AlgNum});
            Info.Para_sort{AlgNum,tN,subject}=Para_sort;
            Info.ValiAcc_sort{AlgNum,tN,subject}=ValiAcc_sort;
            Info.cost(AlgNum,tN,subject)=toc;
        end
        
        % 3.CTFSP
        Algorithm='CTFSP';
        AlgNum=3;
        tic;
        testAcc(AlgNum,tN,subject)=ClassificationFramework(sdataTask,sdataRest,slabel,traindataTask,traindataRest,trainlabel,...
            testdataTask,testdataRest,testlabel,[],Algorithm);
        Info.Para_sort{AlgNum,tN,subject}=[];
        Info.ValiAcc_sort{AlgNum,tN,subject}=[];
        Info.cost(AlgNum,tN,subject)=toc;
        
        % 4.EA-CSP
        DAmethod='EA';Algorithm='EA-CSP';
        AlgNum=4;
        tic;
        if trainNum(tN)==0
            [Para_sort,ValiAcc_sort]=ParaSelcet(sdataTask,slabel,[],Algorithm);
            traindataTask0=[];
            traindataRest0=[];
        else
            [Para_sort,ValiAcc_sort]=ParaSelcet(cat(3,sdataTask,traindataTask),[slabel;trainlabel],cat(3,sdataRest,traindataRest),Algorithm,[],size(sdataTask,3));
            traindataTask0=ERPs_Filter(traindataTask,Para_sort{1,3},Para_sort{1,1},Para_sort{1,2});
            traindataRest0=ERPs_Filter(traindataRest,Para_sort{1,3},Para_sort{1,1});
        end
        sdataTask0=ERPs_Filter(sdataTask,Para_sort{1,3},Para_sort{1,1},Para_sort{1,2});
        sdataRest0=ERPs_Filter(sdataRest,Para_sort{1,3},Para_sort{1,1});
        testdataTask0=ERPs_Filter(testdataTask,Para_sort{1,3},Para_sort{1,1},Para_sort{1,2});
        testdataRest0=ERPs_Filter(testdataRest,Para_sort{1,3},Para_sort{1,1});
        testAcc(AlgNum,tN,subject)=ClassificationFramework(sdataTask0,sdataRest0,slabel,traindataTask0,traindataRest0,trainlabel,...
            testdataTask0,testdataRest0,testlabel,DAmethod,Algorithm);
        Info.Para_sort{AlgNum,tN,subject}=Para_sort;
        Info.ValiAcc_sort{AlgNum,tN,subject}=ValiAcc_sort;
        Info.cost(AlgNum,tN,subject)=toc;

        % 5.EA-CTFSP
        Algorithm='EA-CTFSP';
        AlgNum=5;
        tic;
        testAcc(AlgNum,tN,subject)=ClassificationFramework(sdataTask,sdataRest,slabel,traindataTask,traindataRest,trainlabel,...
            testdataTask,testdataRest,testlabel,[],Algorithm);
        Info.Para_sort{AlgNum,tN,subject}=[];
        Info.ValiAcc_sort{AlgNum,tN,subject}=[];
        Info.cost(AlgNum,tN,subject)=toc;

        % 6.RA-MDRM
        DAmethod='RA';
        Algorithm='RA-MDRM';
        AlgNum=6;
        tic;
        if trainNum(tN)==0
            [Para_sort,ValiAcc_sort]=ParaSelcet(sdataTask,slabel,[],Algorithm);
            traindataTask0=[];
            traindataRest0=[];
        else
            [Para_sort,ValiAcc_sort]=ParaSelcet(cat(3,sdataTask,traindataTask),[slabel;trainlabel],cat(3,sdataRest,traindataRest),Algorithm,[],size(sdataTask,3));
            traindataTask0=ERPs_Filter(traindataTask,Para_sort{1,3},Para_sort{1,1},Para_sort{1,2});
            traindataRest0=ERPs_Filter(traindataRest,Para_sort{1,3},Para_sort{1,1});
        end
        sdataTask0=ERPs_Filter(sdataTask,Para_sort{1,3},Para_sort{1,1},Para_sort{1,2});
        sdataRest0=ERPs_Filter(sdataRest,Para_sort{1,3},Para_sort{1,1});
        testdataTask0=ERPs_Filter(testdataTask,Para_sort{1,3},Para_sort{1,1},Para_sort{1,2});
        testdataRest0=ERPs_Filter(testdataRest,Para_sort{1,3},Para_sort{1,1});
        testAcc(AlgNum,tN,subject)=ClassificationFramework(sdataTask0,sdataRest0,slabel,traindataTask0,traindataRest0,trainlabel,...
            testdataTask0,testdataRest0,testlabel,DAmethod,Algorithm);
        Info.Para_sort{AlgNum,tN,subject}=Para_sort;
        Info.ValiAcc_sort{AlgNum,tN,subject}=ValiAcc_sort;
        Info.cost(AlgNum,tN,subject)=toc;


        % 7.RIGEL
        AlgNum=7;
        Algorithm='RIGEL';
        tic;
        if trainNum(tN)==0
            testAcc(AlgNum,tN,subject)=...
                RIGEL2(sdataTask,sdataRest,slabel,testdataTask,testdataRest,testlabel);
        else
            testAcc(AlgNum,tN,subject)=...
                RIGEL2(traindataTask,traindataRest,trainlabel,testdataTask,testdataRest,testlabel,...
                sdataTask,sdataRest,slabel);
        end
        Info.Para_sort{AlgNum,tN,subject}=[];
        Info.ValiAcc_sort{AlgNum,tN,subject}=[];
        Info.cost(AlgNum,tN,subject)=toc;

    end
end
save('ResultOfTL','testAcc','Info');
