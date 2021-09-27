function [ IDX ] = SMKNN_clustering( X, K )
% SMKNN clustering.
%   IDX = SMKNN_clustering(X, K) partitions the points in the N-by-P data matrix X into K clusters and 
% returns an N-by-1 vector IDX containing the cluster indices of each point.
% 
%   input_args:
%       X : Data sets to be clustered (Rows of X correspond to points, columns correspond to variables)
%       K : The number of clusters
%
%   output_args:
%       IDX : The final clusters' label
%     
%   Demo for UCI data sets : 
%       dataSet = importdata('Sticks.mat');   
%       K = 3;
%       [ clusterLabel ] = SMKNN_clustering( dataSet, K );

data = data_preparation( X );



end

