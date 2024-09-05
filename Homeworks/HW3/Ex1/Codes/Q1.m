%% Part A
clc;
close all;

load("SampleData.mat"); %Loading data

%Seperating class 0 & 1:
cls0_data = TrainingData(:, TrainingLabels==0);
cls1_data = TrainingData(:, TrainingLabels==1);

%Scattering data points:
figure;
scatter(cls0_data(1,:), cls0_data(2,:));
hold on;
scatter(cls1_data(1,:), cls1_data(2,:));
title("Training Data Scatter");
xlabel("X1"); ylabel("X2");
legend("Class 0", "Class 1");


%% Part B
clc;
N = size(TrainingData,2); %Total number of data points
val_ind = randperm(N, 0.3*N); %Validation data points index
val_ind = sort(val_ind);

train_ind = 1:N; %Training data points index
train_ind(val_ind) = [];

%Seperating training and validation data points:
val_data = TrainingData(:, val_ind);
train_data = TrainingData(:, train_ind);

%Seperating labels:
val_labels = TrainingLabels(val_ind);
train_labels = TrainingLabels(train_ind);

%% Part C
clc;
close all;

N_train = length(train_labels);
N_val = length(val_labels);



% Defining the range of values to search:
num_neurons_range = 1:10:N_train;
sigma_range = 0.1:0.1:1;

% Optimum values for sigma and the number of neurons:
best_acc = 0;
best_n = 0;
best_sigma = 0;

% Loop over the range of values
for n = num_neurons_range
    disp("n = "+n);
    for sigma = sigma_range
        % Create RBF network:
        net = newrb(train_data, train_labels, 0, sigma, n, 10);

        % Calculate the output labels for the validation data
        out_labels = sim(net, val_data);

        out_labels(out_labels >= 0.5) = 1;
        out_labels(out_labels < 0.5) = 0;
        

        % Calculate the accuracy of the network
        acc = sum(out_labels == val_labels) / N_val;

        % Check if this is the best result so far
        if acc > best_acc
            best_acc = acc;
            best_n = n;
            best_sigma = sigma;
        end
    end
end

% % Print the best results
disp("Best accuracy: "+ best_acc * 100 + "%");
disp("Best number of neurons: "+ best_n);
disp("Best sigma: "+ best_sigma);



%**************Training a network using the best parameters****************
final_net = newrb(train_data, train_labels, 0, best_sigma, best_n, 1);
view(final_net);

% Calculate the output labels for the validation data
out_labels = sim(final_net, val_data);
out_labels(out_labels >= 0.5) = 1;
out_labels(out_labels < 0.5) = 0;

final_acc = sum(out_labels == val_labels) / N_val
%**************************************************************************



%*****************Plotting the final classification results****************
figure;

subplot(121);
scatter(val_data(1,val_labels==0), val_data(2,val_labels==0));
hold on;
scatter(val_data(1,val_labels==1), val_data(2,val_labels==1));

title("Validation Data Scatter (True Labels)");
xlabel("X1"); ylabel("X2");
legend("Class 0", "Class 1");

subplot(122);
scatter(val_data(1,out_labels==0), val_data(2,out_labels==0));
hold on;
scatter(val_data(1,out_labels==1), val_data(2,out_labels==1));

title("Validation Data Scatter (Predicted Labels)");
xlabel("X1"); ylabel("X2");
legend("Class 0", "Class 1");
%**************************************************************************






