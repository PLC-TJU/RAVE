%% main
% 跨时间: 源数据集Day1,目标数据集Day2
% 训练样本:
% 源数据集全部样本 +
% 目标数据集前0,20,40,60
% 测试样本:
% 目标数据集后80,60,40,20
clear;clc;

fs=250;
resttime=fs*0+1:fs*3;
tasktime=fs*3+1:fs*7;

trainNum=[0,20,40,60];
testAcc=nan(8,length(trainNum),9);
for subject=1:9

    [data1,label1,data2,label2]=BNCI004_2015(subject);
    data1=data1(:,:,label1<3);
    label1=label1(label1<3);
    data2=data2(:,:,label2<3);
    label2=label2(label2<3);
    % session1
    sdataTask=data1(:,tasktime,:);
    sdataRest=data1(:,resttime,:);
    slabel=label1;
    % session2
    tdata=data2;
    tlabel=label2;

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
        % 2.MDM
        Algorithms={'CSP','MDM'};
        for AlgNum=1:length(Algorithms)
            warning(['正在计算第',num2str(tN),'轮',Algorithms{AlgNum},'算法的分类结果。']);
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
        warning(['正在计算第',num2str(tN),'轮',Algorithm,'算法的分类结果。']);
        tic;
        testAcc(AlgNum,tN,subject)=ClassificationFramework(sdataTask,sdataRest,slabel,traindataTask,traindataRest,trainlabel,...
            testdataTask,testdataRest,testlabel,[],Algorithm);
        Info.Para_sort{AlgNum,tN,subject}=[];
        Info.ValiAcc_sort{AlgNum,tN,subject}=[];
        Info.cost(AlgNum,tN,subject)=toc;
        
        % 4.EA-CSP
        DAmethod='EA';Algorithm='EA-CSP';
        AlgNum=4;
        warning(['正在计算第',num2str(tN),'轮',Algorithm,'算法的分类结果。']);
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
        warning(['正在计算第',num2str(tN),'轮',Algorithm,'算法的分类结果。']);
        tic;
        testAcc(AlgNum,tN,subject)=ClassificationFramework(sdataTask,sdataRest,slabel,traindataTask,traindataRest,trainlabel,...
            testdataTask,testdataRest,testlabel,[],Algorithm);
        Info.Para_sort{AlgNum,tN,subject}=[];
        Info.ValiAcc_sort{AlgNum,tN,subject}=[];
        Info.cost(AlgNum,tN,subject)=toc;

        % 6.RA-MDM
        % 7.RA-CSP
        DAmethod='RA';
        Algorithms={'RA-MDM','RA-CSP'};
        for AlgNum=1:length(Algorithms)
            warning(['正在计算第',num2str(tN),'轮',Algorithms{AlgNum},'算法的分类结果。']);
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
            testAcc(AlgNum+5,tN,subject)=ClassificationFramework(sdataTask0,sdataRest0,slabel,traindataTask0,traindataRest0,trainlabel,...
                testdataTask0,testdataRest0,testlabel,DAmethod,Algorithms{AlgNum});
            Info.Para_sort{AlgNum+5,tN,subject}=Para_sort;
            Info.ValiAcc_sort{AlgNum+5,tN,subject}=ValiAcc_sort;
            Info.cost(AlgNum+5,tN,subject)=toc;
        end

        % 8.RIGEL
        AlgNum=8;
        Algorithm='RIGEL';
        warning(['正在计算第',num2str(tN),'轮',Algorithm,'算法的分类结果。']);
        tic;
        if trainNum(tN)==0
            [RIGEL(tN).trainacc,RIGEL(tN).testacc,RIGEL(tN).trainfea,RIGEL(tN).testfea,RIGEL(tN).model,RIGEL(tN).Info]=...
                RIGEL2(sdataTask,sdataRest,slabel,testdataTask,testdataRest,testlabel);
        else
            [RIGEL(tN).trainacc,RIGEL(tN).testacc,RIGEL(tN).trainfea,RIGEL(tN).testfea,RIGEL(tN).model,RIGEL(tN).Info]=...
                RIGEL2(traindataTask,traindataRest,trainlabel,testdataTask,testdataRest,testlabel,...
                sdataTask,sdataRest,slabel);
        end
        testAcc(AlgNum,tN,subject)=max(RIGEL(tN).testacc(end,:));
        Info.Para_sort{AlgNum,tN,subject}=RIGEL(tN).Info.Para_sort;
        Info.ValiAcc_sort{AlgNum,tN,subject}=RIGEL(tN).Info.ValiAcc_sort;
        Info.cost(AlgNum,tN,subject)=toc;

    end
    save(['ResultOfTL_BNCI004_2015\newResult',num2str(subject)],'testAcc','Info');
    save(['ResultOfTL_BNCI004_2015\newResultSub_',num2str(subject)],'RIGEL','trainlabel','testlabel','slabel')
end

AAA=permute(testAcc,[2,3,1]);
AAA=AAA(:,:,[1,3,2,4,5,7,6,8]);
