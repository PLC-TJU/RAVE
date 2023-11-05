%% BNCI 001-2014
function [data1,label1,data2,label2]=BNCI004_2015(ID)

path='dataset\BNCI004-2014';
name={'A','C','D','E','F','G','H','J','L'};
file=[path,'\',name{ID},'.mat'];
load(file,'data');

%Rest 3s + Task 7s
timed=[0,10];

% session 1
X=data{1}.X;
trial=data{1}.trial;
fs=data{1}.fs;
label1=data{1}.y;
data1=nan(30,(timed(2)-timed(1))*fs,length(trial));
for t=1:length(trial)
    data1(:,:,t)=X(trial(t)+timed(1)*fs:trial(t)+timed(2)*fs-1,:)';
end

% session 2
X=data{2}.X;
trial=data{2}.trial;
fs=data{2}.fs;
label2=data{2}.y;
data2=nan(30,(timed(2)-timed(1))*fs,length(trial));
for t=1:length(trial)
    data2(:,:,t)=X(trial(t)+timed(1)*fs:trial(t)+timed(2)*fs-1,:)';
end

%downsampled
data1=resample(data1,250,fs,'Dimension',2);
data2=resample(data2,250,fs,'Dimension',2);

%MI(4: HAND MI, 5: FEET MI)
data1=data1(:,:,label1>3);
data2=data2(:,:,label2>3);
label1=label1(label1>3);
label2=label2(label2>3);
label1=label1-3;
label2=label2-3;

end
