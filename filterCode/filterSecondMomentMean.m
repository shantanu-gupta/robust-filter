function [ estMean ] = filterSecondMomentMean(data)
%filterGaussianMean Run the filter algorithm on a bounded 2nd moment
%distribution
    [N, d] = size(data);
    empiricalMean = mean(data);
    threshold = 9;
    centeredData = bsxfun(@minus, data, empiricalMean)/sqrt(N);

    [U, S, ~] = svdsecon(centeredData', 1);

    lambda = S(1,1)^2;
    v = U(:,1);

    %If the largest eigenvalue is about right, just return
    if lambda < threshold
       estMean = empiricalMean;
    %Otherwise, project in direction of v and filter
    else
        projectedData1 = data * v;
        med = median(projectedData1);

        projectedData = [abs(data*v - med) data];
        sortedProjectedData = sortrows(projectedData);
        T_max = sortedProjectedData(end,1);
        Ts = linspace(0, T_max, 100);
        probs = Ts / sum(Ts);
        T = randsample(Ts, 1, true, probs);
        for i = 1:N
            if sortedProjectedData(i,1) > T
                break
            end
        end
        if i == 1 || i == N
            estMean = empiricalMean;
        else 
            estMean = filterSecondMomentMean(sortedProjectedData(1:i, 2:end));
        end
    end
end
