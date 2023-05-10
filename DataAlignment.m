%% Data alignment methods (abridged version)
% EA
% He H, Wu D. Transfer Learning for Brain–Computer Interfaces: A Euclidean 
% Space Data Alignment Approach. IEEE Trans Biomed Eng, 2020, 67(2): 399-410.
% RA
% Zanini P, Congedo M, Jutten C, et al. Transfer Learning: A Riemannian 
% Geometry Framework With Applications to Brain-Computer Interfaces. 
% IEEE Trans Biomed Eng, 2018, 65(5): 1107-1116.

%  Author: Pan Lincong
%  Edition date: 22 April 2023

function [trainData,testData,P]=DataAlignment(method,traindata,testdata,varargin)
% Input
% method: data alignment method, char
% traindata: training dataset, channels*points*trials
% testdata: test dataset, channels*points*trials, can be empty, but not missing

% Optional input
% 'restdata1',restdata1: resting state sample corresponding to the training data set, channels*points*trials
% 'restdata2',restdata2: resting samples corresponding to the test dataset, channels*points*trials
% 'testFlag': default is 0. When the testFlag is not 0, only the data alignment of the test samples is performed, so parameters such as P need to be entered
% 'P': generally is the projection matrix P obtained from the training phase, which only works when testFlag~=0

% Output
% trainData: the training data set after alignment process, possibly channels*points*trials or channels*channels*trials
% testData: Aligned test data set, possibly channels*points*trials or channels*channels*trials
% P: Projection matrix for data alignment, typically channels*channels

% 解析成对出现的参数名/参数值
[restdata1,restdata2,testFlag,P] = parseInputs(varargin{:});

if isempty(testdata) && testFlag
    error('testdata cannot be empty when testFlag is not 0!');
end


%数据对齐方案
switch upper(method)
    case 'EA'
        if ~testFlag
            [trainData,testData,P]=EA(traindata,testdata);
        else
            testData=nan(size(testdata));
            for j=1:size(testdata,3)
                testData(:,:,j)=P*testdata(:,:,j);
            end
            trainData=testData;
        end
    case 'RA'
        if ~testFlag
            [trainData,testData,P]=RA(traindata,restdata1,testdata);
        else
            testData=nan(size(testdata));
            for j=1:size(testdata,3)
                testData(:,:,j)=P*testdata(:,:,j);
            end
            trainData=testData;
        end
    case 'NONE'
        if ~testFlag
            trainData=traindata;
            testData=testdata;
        else
            trainData=testdata;
            testData=testdata;
        end
        P=1;
    otherwise
        error('使用了不存在的数据对齐方法！')
end

end

%% 数据对齐 EA
% He H, Wu D. Transfer Learning for Brain–Computer Interfaces: A Euclidean 
% Space Data Alignment Approach [J]. IEEE Trans Biomed Eng, 2020, 67(2): 399-410.
function [trainData,testData,P]=EA(traindata,testdata)
trainData=zeros(size(traindata));
if issymmetric(mean(traindata,3))
    meanCov=mean_covariances(traindata,'arithmetic'); 
    P=meanCov^(-0.5);
    for i=1:size(traindata,3)
        trainData(:,:,i)=P*traindata(:,:,i)*P';
    end
else
    covData=covariances(traindata);
    meanCov=mean_covariances(covData,'arithmetic'); 
    P=meanCov^(-0.5);
    for i=1:size(traindata,3)
        trainData(:,:,i)=P*traindata(:,:,i);
    end
end

if nargin < 2 || isempty(testdata)
    testData=[];
else
    testData=zeros(size(testdata));
    if issymmetric(mean(testdata,3))
        for i=1:size(testdata,3)
            testData(:,:,i)=P*testdata(:,:,i)*P';
        end        
    else
        for i=1:size(testdata,3)
            testData(:,:,i)=P*testdata(:,:,i);
        end
    end
end
end

%% 数据对齐 RA
function [trainData,testData,P]=RA(traindata,restdata,testdata)
if nargin < 2 || isempty(restdata)
    restdata=traindata;
end
if issymmetric(mean(restdata,3))
    covRest=restdata;
else
    covRest=covariances(restdata);
end
P=mean_covariances(covRest,'riemann'); 
P=P^(-0.5);

if issymmetric(mean(traindata,3))
    covData1=traindata;
    trainData=zeros(size(covData1));
    for s=1:size(covData1,3)
        trainData(:,:,s)=P*covData1(:,:,s)*P;
    end
else
    trainData=zeros(size(traindata));
    for s=1:size(traindata,3)
        trainData(:,:,s)=P*traindata(:,:,s);
    end
end

if nargin < 3 || isempty(testdata)
    testData=[];
else
    if issymmetric(mean(testdata,3))
        covData2=testdata;
        testData=zeros(size(covData2));
        for s=1:size(testdata,3)
            testData(:,:,s)=P*covData2(:,:,s)*P;
        end
    else
        testData=zeros(size(testdata));
        for s=1:size(testdata,3)
            testData(:,:,s)=P*testdata(:,:,s);
        end
    end
end

end

%% 
function [restdata1,restdata2,testFlag,P] = parseInputs(varargin)
if mod(nargin,2)~=0
    error('输入参数个数不对，应为成对出现!');
end
pnames = {'restdata1','restdata2','testFlag','P'};
dflts =  {[],[],0,[]};
[restdata1,restdata2,testFlag,P] = parseArgs(pnames, dflts, varargin{:});
validateattributes(restdata1,{'double'},{},mfilename,'restdata1');
validateattributes(restdata2,{'double'},{},mfilename,'restdata2');
validateattributes(testFlag,{'double'},{'nonempty'},mfilename,'testFlag');
validateattributes(P,{'double'},{},mfilename,'P');
end

%%
function [varargout] = parseArgs(pnames,dflts,varargin)
nparams = length(pnames);
varargout = dflts;
setflag = false(1,nparams);
unrecog = {};
nargs = length(varargin);
dosetflag = nargout>nparams;
dounrecog = nargout>(nparams+1);

if mod(nargs,2)~=0
    m = message('stats:internal:parseArgs:WrongNumberArgs');
    throwAsCaller(MException(m.Identifier, '%s', getString(m)));
end
 
for j=1:2:nargs
    pname = varargin{j};
    if ~ischar(pname)
        m = message('stats:internal:parseArgs:IllegalParamName');
        throwAsCaller(MException(m.Identifier, '%s', getString(m)));
    end
    
    mask = strncmpi(pname,pnames,length(pname)); % look for partial match
    if ~any(mask)
        if dounrecog
            unrecog((end+1):(end+2)) = {varargin{j} varargin{j+1}};
            continue
        else
            m = message('stats:internal:parseArgs:BadParamName',pname);
            throwAsCaller(MException(m.Identifier, '%s', getString(m)));
        end
    elseif sum(mask)>1
        mask = strcmpi(pname,pnames); % use exact match to resolve ambiguity
        if sum(mask)~=1
            m = message('stats:internal:parseArgs:AmbiguousParamName',pname);
            throwAsCaller(MException(m.Identifier, '%s', getString(m)));
        end
    end
    varargout{mask} = varargin{j+1};
    setflag(mask) = true;
end
 
% Return extra stuff if requested
if dosetflag
    varargout{nparams+1} = setflag;
    if dounrecog
        varargout{nparams+2} = unrecog;
    end
end
end
