%Generates Figure 1 from paper
clear
eps = 0.1;
tau = 0.1;
cher = 2.5;

filterErr = [];
medianErr = [];
ransacErr = [];
LRVErr = [];
sampErr = [];
noisySampErr = []; 
prunedErr = [];
ds = 100:50:400;

for d = ds
    N = 10*floor(d/eps^2);
    fprintf('Training with dimension = %d, number of samples = %d \n', d, round(N, 0))
    sumFilterErr = 0;
    sumMedErr = 0;
    sumRansacErr = 0;
    sumLRVErr = 0;
    sumSampErr = 0;
    sumNoisySampErr = 0;
    sumPrunedErr = 0;

		% Fix the covariance matrix
		% =========================
    % Identity matrices at different scales.
    C = eye(d);
    % C = 0.5 * eye(d);
    % C = 2 * eye(d);
    
    % Diagonal matrices at different scales.
    % C = diag(rand(d,1));
    % C = 2 * diag(rand(d,1));
    
    % Non-diagonal and random matrices at different scales.
    % [Q, R] = qr(rand(d, d));
    % C = Q * diag(rand(d,1)) * Q';
    % C = Q * 2 * diag(rand(d,1)) * Q';
    
    % Tridiagonal matrices at different scales.
    % h = [eps 1 eps];
    % C = convmtx(h, d);
    % C = C(:, 2:end-1);
    % C = C(:,2:end-1) / (1 + eps*2);
    
    % Single Gaussian
    % ---------------
    X =  mvnrnd(zeros(1,d), C, round((1-eps)*N)) + ones(round((1-eps)*N), d);
    true_mean = ones(1, d);

    % Mixture of 2 Gaussians
    % ----------------------
    % G = gmdistribution([ones(1,d); (1 - (1.0 / sqrt(d))) * ones(1, d)], C);
    % X = random(G, round((1-eps)*N));
    % true_mean = (1 - (0.5 / sqrt(d))) * ones(1, d);
    
    fprintf('Sampling Error w/o noise...');
    sumSampErr = sumSampErr + norm(mean(X) - true_mean);
    fprintf('done\n')

    Y1 = randi([0 1], round(0.5*eps*N), d); 
    Y2 = [12*randi([0 1], round(0.5*eps*N), 1),...
          -2*randi([0 1], round(0.5*eps*N), 1),...
          zeros(round(0.5*eps*N), d-2)];
    X = [X; Y1; Y2];

    fprintf('Sampling Error with noise...');
    sumNoisySampErr = sumNoisySampErr + norm(mean(X) - true_mean);
    fprintf('done\n')
    
    
    fprintf('Pruning...');
    [prunedMean, ~] = pruneGaussianMean(X, eps);
    sumPrunedErr = sumPrunedErr + norm(prunedMean - true_mean);
    fprintf('done\n')
    
    fprintf('Median...')
    gm = geoMedianGaussianMean(X);
    sumMedErr = sumMedErr + norm(gm - true_mean);
    fprintf('done\n')
    
    fprintf('Ransac...')
    sumRansacErr = sumRansacErr + norm(ransacGaussianMean(X, eps, tau) - true_mean);
    fprintf('done\n')

    fprintf('LRV...')
    sumLRVErr = sumLRVErr + norm(agnosticMeanGeneral(X, eps) - true_mean);
    fprintf('done\n')

    fprintf('Filter...')
    sumFilterErr = sumFilterErr + norm(filterGaussianMean(X, eps, tau, cher) - true_mean);
    fprintf('done\n')

    medianErr = [medianErr sumMedErr];
    ransacErr = [ransacErr sumRansacErr];
    filterErr = [filterErr sumFilterErr];
    LRVErr = [LRVErr sumLRVErr];
    sampErr = [sampErr sumSampErr];
    noisySampErr = [noisySampErr sumNoisySampErr];
    prunedErr = [prunedErr sumPrunedErr];
end

noisySampErr = noisySampErr - sampErr;
prunedErr = prunedErr - sampErr;
medianErr = medianErr - sampErr;
ransacErr = ransacErr - sampErr;
LRVErr = LRVErr - sampErr;
filterErr = filterErr - sampErr;

plot(ds, noisySampErr, ds, prunedErr, ds, medianErr, '-ro', ds, ransacErr, ds, LRVErr, ds, filterErr, '-.b', 'LineWidth', 2)
xlabel('Dimension')
ylabel('Excess L2 error')
legend('Sampling Error (with noise)', 'Naive Pruning', 'Geometric Median', 'RANSAC', 'LRV', 'Filter')
