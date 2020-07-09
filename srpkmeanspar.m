function [outputs]= srpkmeanspar(inputDir,K,p,varargin)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Parallel SRPK-means||
%
%   inputs:
%   inputDir        - Input data directory
%   K               - #clusters
%   l               - Oversampling factor
%   iterMax         - Maximum #parkmeans iterations
%   workers         - #workers
%   runs            - #runs
%   convTh          - Convergence threshold
%   openParPool     - Opens parpool if true
%
%   outputs:
%   initTimeArray   - Initialization running times for each run   
%   timeArray       - Kmeans search running times for each run
%   itersArray      - #parkmeans search iterations for each run
%   sseArray        - SSEs for each run
%   Cfinal          - The best set of prototypes
%
% Reference:
%   [1] J. Hämäläinen, T. Kärkkäinen, and T. Rossi, Scalable Initialization
%   Methods for Large-Scale Clustering, Arxiv preprint, 2020.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

pnames =   {    'l'  'iterMax'  'workers' 'runs' 'convTh' 'r' 'openParPool' 'Nsplits' 'initMaxIter'};
defaults = {    2*K  300        10         1      0        5   1             8        5};

[l,iterMax,workers,runs,convTh,r,openParPool,Nsplits,initMaxIter] =...
    internal.stats.parseArgs(pnames, defaults, varargin{:});
sseBest = realmax;

if(rem(workers,Nsplits)~=0)
    error('Not valid number of workers or Nsplits!')
end

% load data to workers
if(openParPool)
    parpool(workers)
end
WNs = vec2mat(1:workers,workers/Nsplits);
spmd
    X = dlmread([inputDir,'/part_',num2str(workers),'_',...
        num2str(labindex),'.txt']);
    [N,M] = size(X);
    L = ones(N,1);
    Nglob = gplus(N);
    W = []; rii = [1,-1];
    WNsi = WNs(any(WNs == labindex, 2),:);
    Cinit = [];
end

outputs = struct('initTimeArray', zeros(1,runs),'timeArray', zeros(1,runs),...
'itersArray',  zeros(1,runs), 'sseArray',  zeros(1,runs),'Cfinal',zeros(K,M{1}));

for run =1:runs
    disp(['run = ' num2str(run)])
    tic
    % Initialization
    ranObs = repmat(randi(workers/Nsplits,[1,Nsplits]),[workers/Nsplits,1]);
    spmd
        if(WNsi(1) == labindex)
            I_mat = randi([1 2],M,p);
            labBroadcastSubset(WNsi(1),WNsi,I_mat);
            R = rii(I_mat);
            Xp = (1/sqrt(p))*(X*R);
        else
            I_mat = labBroadcastSubset(WNsi(1),WNsi);
            R = rii(I_mat);
            Xp = (1/sqrt(p))*(X*R);
        end
        if(WNsi(ranObs(labindex)) == labindex)
            C = Xp(randi(N),:);
            labBroadcastSubset(WNsi(ranObs(labindex)),WNsi,C);
        else
            C = labBroadcastSubset(WNsi(ranObs(labindex)),WNsi);
        end
        D = sum(Xp.^2,2) + repmat(sum(C.^2,2)',N,1) - 2*(Xp*C');
        psi = gplussubset(sum(D),WNsi);
        Cps = C;
        L = ones(N,1);
        Nsampled = 1;
        for ir = 1:r
            ra = rand(N,1);
            Cp = gopsubset(@vertcat,Xp(l*D/psi>ra,:),WNsi);
            Cps = [Cps; Cp];
            [~,Lnew] = min(bsxfun(@plus,-2*(Xp*Cp'),sum(Cp.*Cp,2)'),[],2);
            [D,idnew] = min([D, sum((Xp-Cp(Lnew,:)).^2,2)],[],2);
            hidnew = idnew>1; 
            L(hidnew) = Lnew(hidnew) + Nsampled;
            psi = gplussubset(sum(D),WNsi);
            Nsampled = Nsampled + size(Cp,1);
        end
        W_Part = accumarray(L,1);
        W_Part(end+1:Nsampled)=0;
        W = gplussubset(W_Part,WNsi,WNsi(1));
        tic;
        if(labindex==WNsi(1))
            Cw = kmeansppw2(Cps,K,W);
        end
        serialTime = toc;
    end
    timeI1 = toc; disp(['phase1 init time:',num2str(timeI1)])
    serTimes= zeros(1,size(WNs,1));
    for serTimei = 1:size(WNs,1)
        serTimes(serTimei) = serialTime{WNs(serTimei,1)};
    end
    serialTimeMax = max(serTimes);
    disp(['Serial time: ', num2str(serialTimeMax)])
    tic;
    bestCInd = parsubrpkmeans;
    timeI2 = toc; disp(['phase2 init time:',num2str(timeI2)])
        outputs.initTimeArray(run) = timeI1+timeI2;
    disp(['Initialization time: ', num2str(outputs.initTimeArray(run)),...
        ' seconds'])
     clear D; clear Cps; clear Lnew; clear Cp; clear hidnew;    
    % Search
    [ctrs,time,iters,sse] = parkmeans;
    outputs.timeArray(run) = time;
    outputs.itersArray(run) = iters;
    outputs.sseArray(run) = sse;
    outputs.serialTime(run) = serialTimeMax;
    
    % Init Error
    spmd
        [~,L] = min(bsxfun(@plus,-2*(X*Cinit'),sum(Cinit.*Cinit,2)'),[],2);
        ssepart = sum(sum((X-Cinit(L,:)).^2,2));
        sseInit = gplus(ssepart,1);
    end
    outputs.sseiArray(run) = sseInit{1};
    
    if(sse < sseBest)
        sseBest = sse;
        outputs.Cfinal = ctrs;
    end
end

if(openParPool)
    delete(gcp)
end


    %
    % Parallel subset RPK-means
    %
    function bestCInd = parsubrpkmeans
    warning off all;
    spmd
        if(WNsi(1) == labindex)
            Cpr = Cw;
            labBroadcastSubset(WNsi(1),WNsi,Cpr);
        else
            Cpr = labBroadcastSubset(WNsi(1),WNsi);
        end
        iters = 0; L = ones(N,1);
        while( iters < initMaxIter)
            [~,L] = min(bsxfun(@plus,-2*(Xp*Cpr'),sum(Cpr.*Cpr,2)'),[],2);
            Lsparse = sparse(L,1:N,1,K,N,N);
            Xcsum = Lsparse*Xp;
            Ncsum = full(sum(Lsparse,2));
            Sxcsum = gplussubset(Xcsum,WNsi);
            Sncsum = gplussubset(Ncsum,WNsi);
            Cpr = bsxfun(@rdivide,Sxcsum,Sncsum); 
            iters = iters+1;
        end
        [~,L] = min(bsxfun(@plus,-2*(Xp*Cpr'),sum(Cpr.*Cpr,2)'),[],2);
        Lsparse = sparse(L,1:N,1,K,N,N);
        Xcsum = Lsparse*X;
        Ncsum = full(sum(Lsparse,2));
        Sxcsum = gplussubset(Xcsum,WNsi);
        Sncsum = gplussubset(Ncsum,WNsi);
        C = bsxfun(@rdivide,Sxcsum,Sncsum); 
        ssesubset = gplussubset(sum(sum((X-C(L,:)).^2,2)),WNsi,WNsi(1));
    end
    ssesubsetArr = zeros(1,size(WNs,1));
    for candC = 1:size(WNs,1)
        ssesubsetArr(candC) = ssesubset{WNs(candC,1)};
    end
    [~,minInd] = min(ssesubsetArr);
    bestCInd = WNs(minInd,1);
    end


    %
    % Parallel K-means
    %
    function [ctrs,time,iters,sse] = parkmeans
    warning off all;
    tic;
    spmd
        if(bestCInd == labindex)
            Cinit = C;
            labBroadcast(bestCInd,C);
        else
            C = labBroadcast(bestCInd);
            Cinit = C;
        end
        iters = 0; converged = 0; L = ones(N,1);
        while( ~converged && (iters < iterMax))
            L1 = L;
            [~,L] = min(bsxfun(@plus,-2*(X*C'),sum(C.*C,2)'),[],2);
            Lsparse = sparse(L,1:N,1,K,N,N);
            Xcsum = Lsparse*X;
            Ncsum = full(sum(Lsparse,2));
            Sxcsum = gplus(Xcsum);
            Sncsum = gplus(Ncsum);
            C = bsxfun(@rdivide,Sxcsum,Sncsum); 
            newlabels = gplus(sum(L1-L~=0));
            newlabels = newlabels/Nglob;
            if(newlabels <= convTh)
                converged = 1;
            end
            iters = iters+1;
            if(labindex==1)
                disp(['iter: ', num2str(iters), ', newassignments: ',...
                    num2str(newlabels)])
            end
        end
    end
    time = toc;
    disp(['Search time:         ', num2str(time), ' seconds'])
    
    % Error
    spmd
        [~,L] = min(bsxfun(@plus,-2*(X*C'),sum(C.*C,2)'),[],2);
        ssepart = sum(sum((X-C(L,:)).^2,2));
        sse = gplus(ssepart,1);
    end
    sse = sse{1}; ctrs = C{1}; iters = iters{1};
    end
end