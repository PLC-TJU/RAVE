%% main
% 非跨时间分类——逐渐增加固定序列的训练样本
% 仅对目标数据集进行计算，即使用Day2的数据集
% 训练样本为：前40、80、120、160
% 测试样本为：后160、120、80、40
clear;clc;

fs=250;
resttime=fs*0+1:fs*3;
tasktime=fs*3+1:fs*7;

trainNum=[40,80,120,160];
testAcc=nan(8,length(trainNum),12);
for subject=1:12

    [~,~,data2,label2]=BNCI001_2015(subject);
    data2=data2(:,:,label2<3);
    label2=label2(label2<3);
    dataTask=data2(:,tasktime,:);
    dataRest=data2(:,resttime,:);
    label=label2;

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
            warning(['正在计算第',num2str(tN),'轮',Algorithms{AlgNum},'算法的分类结果。']);
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
        warning(['正在计算第',num2str(tN),'轮',Algorithm,'算法的分类结果。']);
        tic;
        testAcc(AlgNum,tN,subject)=ClassificationFramework([],[],[],traindataTask,traindataRest,trainlabel,...
            testdataTask,testdataRest,testlabel,[],Algorithm);
        Info.Para_sort{AlgNum,tN,subject}=[];
        Info.ValiAcc_sort{AlgNum,tN,subject}=[];
        Info.cost(AlgNum,tN,subject)=toc;
        
        % 4.EA-CSP
        DAmethod='EA';Algorithm='EA-CSP';
        AlgNum=4;
        warning(['正在计算第',num2str(tN),'轮',Algorithm,'算法的分类结果。']);
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
        warning(['正在计算第',num2str(tN),'轮',Algorithm,'算法的分类结果。']);
        tic;
        testAcc(AlgNum,tN,subject)=ClassificationFramework([],[],[],traindataTask,traindataRest,trainlabel,...
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
            [Para_sort,ValiAcc_sort]=ParaSelcet(traindataTask,trainlabel,traindataRest,Algorithms{AlgNum},[],0);
            traindataTask0=ERPs_Filter(traindataTask,Para_sort{1,3},Para_sort{1,1},Para_sort{1,2});
            traindataRest0=ERPs_Filter(traindataRest,Para_sort{1,3},Para_sort{1,1});
            testdataTask0=ERPs_Filter(testdataTask,Para_sort{1,3},Para_sort{1,1},Para_sort{1,2});
            testdataRest0=ERPs_Filter(testdataRest,Para_sort{1,3},Para_sort{1,1});
            testAcc(AlgNum+5,tN,subject)=ClassificationFramework([],[],[],traindataTask0,traindataRest0,trainlabel,...
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
        [RIGEL(tN).trainacc,RIGEL(tN).testacc,RIGEL(tN).trainfea,RIGEL(tN).testfea,RIGEL(tN).model,RIGEL(tN).Info]=...
            RIGEL2(traindataTask,traindataRest,trainlabel,testdataTask,testdataRest,testlabel);
        testAcc(AlgNum,tN,subject)=max(RIGEL(tN).testacc(end,:));
        Info.Para_sort{AlgNum,tN,subject}=RIGEL(tN).Info.Para_sort;
        Info.ValiAcc_sort{AlgNum,tN,subject}=RIGEL(tN).Info.ValiAcc_sort;
        Info.cost(AlgNum,tN,subject)=toc;

    end
    save(['Result_BNCI001_2015\newResult',num2str(subject)],'testAcc','Info');
    save(['Result_BNCI001_2015\newResultSub_',num2str(subject)],'RIGEL','trainlabel','testlabel')
end

AAA=permute(testAcc,[2,3,1]);
AAA=AAA(:,:,[1,3,2,4,5,7,6,8]);
