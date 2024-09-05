%% Part B
clear;
clc;
close all;

%Loading data
data = cell2mat(struct2cell(load("DataNew.mat")));

clustNum = 5; %Number of clusters (=k)

%Setting k data points as the center of the initial clusters:
init_centers1 = randperm(size(data,2), clustNum); %Initial condition1
init_centers2 = randperm(size(data,2), clustNum); %initial condition2
init_clustMat1 = data(:, init_centers1);
init_clustMat2 = data(:, init_centers2);

%Computing k-means:
[clust_ind1, clustMat1] = kmeansCal(data, clustNum, init_clustMat1);
[clust_ind2, clustMat2] = kmeansCal(data, clustNum, init_clustMat2);

%Scattering clustered data points:
apply_kmeans(data, clustNum, clust_ind1, clust_ind2, clustMat1, clustMat2);


%% Part C

%******************************4 clusters*********************************
clustNum = 4;

init_centers1 = randperm(size(data,2), clustNum);
init_centers2 = randperm(size(data,2), clustNum);
init_clustMat1 = data(:, init_centers1);
init_clustMat2 = data(:, init_centers2);
[clust_ind1, clustMat1] = kmeansCal(data, clustNum, init_clustMat1);
[clust_ind2, clustMat2] = kmeansCal(data, clustNum, init_clustMat2);

apply_kmeans(data, clustNum, clust_ind1, clust_ind2, clustMat1, clustMat2);
%*************************************************************************



%******************************6 clusters*********************************
clustNum = 6;

init_centers1 = randperm(size(data,2), clustNum);
init_centers2 = randperm(size(data,2), clustNum);
init_clustMat1 = data(:, init_centers1);
init_clustMat2 = data(:, init_centers2);
[clust_ind1, clustMat1] = kmeansCal(data, clustNum, init_clustMat1);
[clust_ind2, clustMat2] = kmeansCal(data, clustNum, init_clustMat2);

apply_kmeans(data, clustNum, clust_ind1, clust_ind2, clustMat1, clustMat2);
%*************************************************************************


%% Part D


%******************************4 clusters*********************************
clustNum = 4;

[clust_ind1,clustMat1] = kmeans(data.', clustNum, 'Start', 'sample'); %Using the function kmeans with random sampling as the initial points
[clust_ind2,clustMat2] = kmeans(data.', clustNum, 'Start', 'sample');
clustMat1 = clustMat1.';
clustMat2 = clustMat2.';

apply_kmeans(data, clustNum, clust_ind1, clust_ind2, clustMat1, clustMat2);
%*************************************************************************

%******************************5 clusters*********************************
clustNum = 5;

[clust_ind1,clustMat1] = kmeans(data.', clustNum, 'Start', 'sample');
[clust_ind2,clustMat2] = kmeans(data.', clustNum, 'Start', 'sample');
clustMat1 = clustMat1.';
clustMat2 = clustMat2.';

apply_kmeans(data, clustNum, clust_ind1, clust_ind2, clustMat1, clustMat2);
%*************************************************************************


%******************************6 clusters*********************************
clustNum = 6;

[clust_ind1,clustMat1] = kmeans(data.', clustNum, 'Start', 'sample');
[clust_ind2,clustMat2] = kmeans(data.', clustNum, 'Start', 'sample');
clustMat1 = clustMat1.';
clustMat2 = clustMat2.';

apply_kmeans(data, clustNum, clust_ind1, clust_ind2, clustMat1, clustMat2);
%*************************************************************************


%% Part E
clc;
close all;

%******************************4 clusters*********************************
clustNum = 4;
Z = linkage(data.', 'ward');
clust_ind = cluster(Z, 'Maxclust',clustNum);
apply_hier(data, clustNum, clust_ind);
%*************************************************************************


%******************************5 clusters*********************************
clustNum = 5;
Z = linkage(data.', 'ward');
clust_ind = cluster(Z, 'Maxclust',clustNum);
apply_hier(data, clustNum, clust_ind);
%*************************************************************************


%******************************6 clusters*********************************
clustNum = 6;
Z = linkage(data.', 'ward');
clust_ind = cluster(Z, 'Maxclust',clustNum);
apply_hier(data, clustNum, clust_ind);
%*************************************************************************


%% Functions

function [clust_ind, clustMat] = kmeansCal(dataMat, clustNum, init_clustMat)
%dataMat dimension = features*points (2*1000)
%init_clustMat & clustMat dimension = features*clustNum
%clust_ind dimension = 1*clustNum

    features = size(dataMat,1); 
    N = size(dataMat,2); %Number of data points

    
    
    data_clust_mat = [dataMat; zeros(1,N)]; %Adding the corresponding cluster to the last row of the data

    clust_ind = data_clust_mat(end,:); %Cluster index for each data point
    clustMat = zeros(size(init_clustMat)); %Cluseters center coordination
    new_clustMat = init_clustMat; 
    iter = 0; %Iteration
    while ~isempty(find(new_clustMat-clustMat >= 0.000001, 1))
        clustMat = new_clustMat;
        data_clust_mat(end, :) = zeros(1,N); %Resetting cluster indices
        disp("iter " + iter);
        iter = iter+1;

        for i = 1:N %For each data point:
            dist_arr = zeros(clustNum,1); %Distances between each data point and each cluster center
            for j = 1:clustNum %For each cluster
                dist_arr(j) = sqrt(sum((dataMat(:,i) - clustMat(:,j)).^2));
            end
            nearest_clust = find(dist_arr == min(dist_arr), 1); %Finding the nearest cluster to data point i
            data_clust_mat(end, i) = nearest_clust;
        end
        
        %New clustMat matrix:
        new_clustMat = zeros(size(clustMat));
        for i = 1:clustNum
            new_clustMat(:, i) = mean(data_clust_mat(1:end-1, data_clust_mat(end,:)==i),2);
        end
        clust_ind = data_clust_mat(end,:); %Updating cluster indices
    end
end




function apply_kmeans(data, clustNum, clust_ind1, clust_ind2, clustMat1, clustMat2)
    figure;
    
    %******Main data
    subplot(131);
    scatter(data(1,:), data(2,:));
    title("Data Scatter");
    xlabel("X1"); ylabel("X2");
    
    %******kmeans with initial conditions1
    subplot(132);
    hold on;
    legend_str = strings(1,clustNum+1);
    for i = 1:clustNum
        scatter(data(1,clust_ind1==i), data(2,clust_ind1==i));
        legend_str(i) = "Cluster "+i;
    end
    legend_str(end) = "Cluster Centers";
    scatter(clustMat1(1,:), clustMat1(2,:), 'black', 'filled', 'o');
    
    title("Data Scatter After Clustering with size "+clustNum+" (Initial centers1)");
    xlabel("X1"); ylabel("X2");
    legend(legend_str);
    
    %******kmeans with initial conditions2
    subplot(133);
    hold on;
    legend_str = strings(1,clustNum+1);
    for i = 1:clustNum
        scatter(data(1,clust_ind2==i), data(2,clust_ind2==i));
        legend_str(i) = "Cluster "+i;
    end
    legend_str(end) = "Cluster Centers";
    scatter(clustMat2(1,:), clustMat2(2,:), 'black', 'filled', 'o');
    
    title("Data Scatter After Clustering with size "+clustNum+" (Initial centers2)");
    xlabel("X1"); ylabel("X2");
    legend(legend_str);
end





function apply_hier(data, clustNum, clust_ind)
    figure;
    
    %******Main data
    subplot(121);
    scatter(data(1,:), data(2,:));
    title("Data Scatter");
    xlabel("X1"); ylabel("X2");
    
    %******LVQ
    subplot(122);
    hold on;
    legend_str = strings(1,clustNum);
    for i = 1:clustNum
        scatter(data(1,clust_ind==i), data(2,clust_ind==i));
        legend_str(i) = "Cluster "+i;
    end
    
    title("Data Scatter After Clustering with size "+clustNum);
    xlabel("X1"); ylabel("X2");
    legend(legend_str);
end



