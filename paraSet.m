%% We use this function to generate 441 representative combinations of time-frequency parameters
%  Author: Pan Lincong
%  Edition date: 22 April 2023

% Modelset
% The first column is the channel
% The second column is the time window.
% Third column is the frequency band
function Modelset=paraSet

% time windows
MinTime=1.5;
MaxTime=4;
MinStep=0.5;
num1=0;
for low=0:MinStep:MaxTime-MinTime %MaxTime-MinTime
    for high=low+MinTime:MinStep:MaxTime
        num1=num1+1;
        timewindow(num1,1)=low;
        timewindow(num1,2)=high;
    end
end

% frequency bands
MinFreq=4;
MaxFreq=40;
MinStep=8;
num2=0;
for low=MinFreq:4:12
    for high=low+MinStep:4:MaxFreq
        num2=num2+1;
        freqwindow(num2,1)=low;
        freqwindow(num2,2)=high;
    end
end

% channels
channelset{1,1}=[1:28];
Modelset=cell(size(freqwindow,1)*size(timewindow,1)*size(channelset,1),3);
for c=1:size(channelset,1)
    for n=1:size(freqwindow,1)
        for m=1:size(timewindow,1)
            ind=m+(n-1)*size(timewindow,1)+(c-1)*size(timewindow,1)*size(freqwindow,1);            
            Modelset{ind,1}=channelset{c};
            Modelset{ind,2}=timewindow(m,:);
            Modelset{ind,3}=freqwindow(n,:);
        end
    end
end

end
