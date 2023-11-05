%% BNCI 001-2014
function [data1,label1,data2,label2]=BNCI001_2014(ID)

path='dataset\BNCI001-2014';

% session 1
load([path,'\A0',num2str(ID),'T.mat'],'data');
num=1:length(data);
num=num(end-5:end);
data1=[];
label1=[];
for n=1:6
    X=data{num(n)}.X;
    trial=data{num(n)}.trial;
    fs=data{num(n)}.fs;
    temp=nan(22,6*fs,length(trial));
    for t=1:length(trial)
        temp(:,:,t)=X(trial(t):trial(t)+6*fs-1,1:22)';
    end
    data1=cat(3,data1,temp);
    label1=cat(1,label1,data{num(n)}.y);
end

% session 2
load([path,'\A0',num2str(ID),'E.mat'],'data');
num=1:length(data);
num=num(end-5:end);
data2=[];
label2=[];
for n=1:6
    X=data{num(n)}.X;
    trial=data{num(n)}.trial;
    fs=data{num(n)}.fs;
    temp=nan(22,6*fs,length(trial));
    for t=1:length(trial)
        temp(:,:,t)=X(trial(t):trial(t)+6*fs-1,1:22)';
    end
    data2=cat(3,data2,temp);
    label2=cat(1,label2,data{num(n)}.y);
end

end
