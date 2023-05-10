%% Data pre-processing
%  Author: Pan Lincong
%  Edition date: 22 April 2023

% input
% data:channels*points*samples
function Data=ERPs_Filter(data,freqs,channel,timewindow,fs,filterorder,filterflag)
%Filter parameters
if nargin < 7
    filterflag = 'filtfilt' ;
end
if nargin < 6
    filterorder = 5;   
end
if nargin < 5
    fs=250;
end
if nargin < 4
    timewindow=[];
end
if nargin < 3
    channel=[];
end
if nargin < 2
    error('ERPs_Filter函数的输入参数不够！');
end

%Channel Selection
if ~isempty(channel) 
    if channel(end)>size(data,1)
        warning('所选导联超出数据限制，已取消导联筛选！')
    end
    data=data(channel,:,:);
end

%Time Window Selection
if ~isempty(timewindow)
    if length(timewindow)==1
        timewindow=[0,timewindow];
    end
    if timewindow(2)*fs>size(data,2)
        
        timewindow(2)=size(data,2)/fs;
        warning(['所选时间窗超出数据限制，已自动调整时间窗为:',num2str(timewindow(1)),'-',num2str(timewindow(2)),'s'])
    end
    data=data(:,round(timewindow(1)* fs) + 1:round(timewindow(2) * fs),:);
end

% Bandpass filtering
Data=zeros(size(data));
if iscell(freqs)
    f_a=freqs{1};
    f_b=freqs{2};
else
    filtercutoff = [2*freqs(1)/fs 2*freqs(2)/fs];
    [f_b, f_a] = butter(filterorder,filtercutoff);
end
for s=1:size(data,3)
    data1=detrend(data(:,:,s)'); 
    switch filterflag
        case 'filter'           
            data2 = filter(f_b,f_a,data1);
        case 'filtfilt'
            data2 = filtfilt(f_b,f_a,data1);
    end
    Data(:,:,s)=data2';
end