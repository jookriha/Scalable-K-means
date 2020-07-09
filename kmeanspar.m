function [outputs]= kmeanspar(inputDir,K,varargin)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Parallel K-means||
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
% References:
%   [1] B. Bahmani, B. Moseley, A. Vattani, R. Kumar, and S. Vassilvitskii,
%       Scalable k-means++. Proceedings of the VLDB Endowment,5(7):622-633,
%       2012.
%   [2] J. Hämäläinen, T. Kärkkäinen, and T. Rossi, Scalable Initialization
%       Methods for Large-Scale Clustering, Arxiv preprint, 2020.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

pnames =   {    'l'  'iterMax'  'workers' 'runs' 'convTh' 'r' 'openParPool'};
defaults = {    2*K  300        10         1      0        5   1 };

[l,iterMax,workers,runs,convTh,r,openParPool] =...
    internal.stats.parseArgs(pnames, defaults, varargin{:});
sseBest = realmax;

% load data to workers
if(openParPool)
    parpool(workers)
end
spmd
    X = dlmread([inputDir,'/part_',num2str(workers),'_',...
        num2str(labindex),'.txt']);
    [N,M] = size(X);
    L = ones(N,1);
    Nglob = gplus(N);
    W = []; Cinit = [];
end

outputs = struct('initTimeArray', zeros(1,runs),'timeArray', zeros(1,runs),...
'itersArray',  zeros(1,runs), 'sseArray',  zeros(1,runs),'Cfinal',zeros(K,M{1}));

for run =1:runs
    disp(['run = ' num2str(run)])
    tic

    % Initialization
    ranObs = randi(workers);    
    spmd
        if(ranObs == labindex)
            C_rand = X(randi(N),:);
            C = labBroadcast(ranObs,C_rand);
        else
            C = labBroadcast(ranObs);
        end
        D = sum(X.^2,2) + repmat(sum(C.^2,2)',N,1) - 2*(X*C');
        psi = gplus(sum(D));
        Cps = C;
        L = ones(N,1);
        Nsampled = 1;
        for ir = 1:r
            ra = rand(N,1);
            Cp = gop(@vertcat,X(l*D/psi>ra,:));
            Cps = [Cps; Cp];
            [~,Lnew] = min(bsxfun(@plus,-2*(X*Cp'),sum(Cp.*Cp,2)'),[],2);
            [D,idnew] = min([D, sum((X-Cp(Lnew,:)).^2,2)],[],2);
            hidnew = idnew>1; 
            L(hidnew) = Lnew(hidnew) + Nsampled;
            psi = gplus(sum(D));
            Nsampled = Nsampled + size(Cp,1);
        end
        W_Part = accumarray(L,1);
        W_Part(end+1:Nsampled)=0;
        W = gplus(W_Part,1);
        tic;
        if(labindex==1)
            Cw = kmeansppw2(Cps,K,W); 
        end
        serialTime = toc;
    end
    outputs.initTimeArray(run) = toc;
    disp(['Serial time: ', num2str(serialTime{1})])
    disp(['Initialization time: ', num2str(outputs.initTimeArray(run)),...
        ' seconds'])
    clear D; clear Cps; clear Lnew; clear Cp; clear hidnew;

    % Search
    [ctrs,time,iters,sse] = parkmeans;
    outputs.timeArray(run) = time;
    outputs.itersArray(run) = iters;
    outputs.sseArray(run) = sse;
    outputs.serialTime(run) = serialTime{1};
    
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
    % Parallel K-means
    %
    function [ctrs,time,iters,sse] = parkmeans
    warning off all;
    tic;
    spmd
        if(1 == labindex)
            Cinit = C;
            C = labBroadcast(1,Cw);
        else
            C = labBroadcast(1);
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