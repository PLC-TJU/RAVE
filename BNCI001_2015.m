%% BNCI 001-2014
function [data1,label1,data2,label2]=BNCI001_2015(ID)

path='dataset\BNCI001-2015';

if ID<10
    file1=[path,'\S0',num2str(ID),'A.mat'];
    file2=[path,'\S0',num2str(ID),'B.mat'];
else
    file1=[path,'\S',num2str(ID),'A.mat'];
    file2=[path,'\S',num2str(ID),'B.mat'];
end

% session 1
load(file1,'data');
X=data.X;
trial=data.trial;
fs=data.fs;
label1=data.y;
data1=nan(13,8*fs,length(trial));
for t=1:length(trial)
    data1(:,:,t)=X(trial(t)-3*fs+1:trial(t)+5*fs,:)';
end

%降采样250Hz
data1=resample(data1,250,fs,'Dimension',2);

% session 2
load(file2,'data');
X=data.X;
trial=data.trial;
fs=data.fs;
label2=data.y;
data2=nan(13,8*fs,length(trial));
for t=1:length(trial)
    data2(:,:,t)=X(trial(t)-3*fs+1:trial(t)+5*fs,:)';
end

%降采样250Hz
data2=resample(data2,250,fs,'Dimension',2);


end
