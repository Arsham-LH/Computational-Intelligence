%% Question2: Scattering Training and Test Data points
clc; close all;

load('Ex2.mat'); %Loading the data

%****************************Training Data:*****************************
class0_TrainData = TrainData(1:3, TrainData(4,:) == 0);
class1_TrainData = TrainData(1:3, TrainData(4,:) == 1);

figure;
scatter3(class0_TrainData(1,:), class0_TrainData(2,:), class0_TrainData(3,:));
hold on;
scatter3(class1_TrainData(1,:), class1_TrainData(2,:), class1_TrainData(3,:));

title("Training Data scatter");
xlabel("X"); ylabel("Y"); zlabel("Z");

legend("Class 0", "Class 1");
%**********************************************************************



%****************************Test Data:*****************************
figure;
scatter3(TestData(1,:), TestData(2,:), TestData(3,:));

title("Test Data scatter");
xlabel("X"); ylabel("Y"); zlabel("Z");
%**********************************************************************


%% Question2: a
clc;

n = size(TrainData, 2); %Total number of training and validation points
val_n = 0.2*n; %Number of validation points

valPoints_ind = sort(randperm(n, val_n)); %Validation points (random)

ValData = TrainData(:, valPoints_ind); %Validation data
LearnData = TrainData; LearnData(:, valPoints_ind) = []; %Training Data

%Seperating classes:
class0_ValData = ValData(1:3, ValData(4,:) == 0); %[3*90]
class1_ValData = ValData(1:3, ValData(4,:) == 1); %[3*90]
class0_LearnData = LearnData(1:3, LearnData(4,:) == 0); %[3*90]
class1_LearnData = LearnData(1:3, LearnData(4,:) == 1); %[3*90]



hiddenSize = 10; %The maximum number of neurons in the hidden layer
trainsNum = 7; %The number of trainings for each hiddenSize


train_perf_arr = zeros(1, hiddenSize); %Array of MSE values for training data points
val_perf_arr = zeros(1, hiddenSize); %Array of MSE values for validation data points


for i = 1:hiddenSize

    %Creating the network:
    net = feedforwardnet(i);
    net.divideParam.trainRatio = 1;
    net.divideParam.testRatio = 0;
    net.divideParam.valRatio = 0;
 
    train_tmp_perf_arr = zeros(1, length(trainsNum)); %MSE values for each training (on training data points)
    val_tmp_perf_arr = zeros(1, length(trainsNum)); %MSE values for each training (on validation data points)

    tmp_pred_y_arr = zeros(trainsNum, n); %Predicted outputs for each training

    for j = 1:trainsNum
        trained_net = train(net, LearnData(1:3, :), LearnData(4,:)); %Training the network using training data points
        tmp_pred_y_arr(j, :) = round(trained_net([LearnData(1:3, :),ValData(1:3, :)])); %Predicting the outputs using the network

        %Seperating the output of Learning points and Validation points:
        learn_tmp_pred_y_arr = tmp_pred_y_arr(j, 1:(end-val_n));
        val_tmp_pred_y_arr = tmp_pred_y_arr(j, (end-val_n+1):end);

        %Computing MSE on Training and Validation data points:
        train_tmp_perf_arr(j) = perform(trained_net, learn_tmp_pred_y_arr, LearnData(4,:));
        val_tmp_perf_arr(j) = perform(trained_net, val_tmp_pred_y_arr, ValData(4,:));
    end


    %Averaging MSE over different trainings:
    val_perf_arr(i) = mean(val_tmp_perf_arr);
    train_perf_arr(i) = mean(train_tmp_perf_arr);

end


% %Plotting MSE for training and validation data points for different numbers of hidden neurons:
figure;

subplot(211);
plot(1:hiddenSize, train_perf_arr);
title("MSE for Training Data");
xlabel("Size of the Hidden Layer"); ylabel("Average MSE over " + trainsNum + " trainings");

subplot(212);
plot(1:hiddenSize, val_perf_arr);
title("MSE for Validation Data");
xlabel("Size of the Hidden Layer"); ylabel("Average MSE over " + trainsNum + " trainings");





%The best number of neurons = argmin(average MSE):
finalSize = find(val_perf_arr == min(val_perf_arr));

%Creating the final network:
net = feedforwardnet(finalSize);
net.divideParam.trainRatio = 1;
net.divideParam.testRatio = 0;
net.divideParam.valRatio = 0;

trained_net = train(net, LearnData(1:3, :), LearnData(4,:)); %Training the network using training data points

%Predicting the outputs using the network:
learn_pred_y = round(trained_net(LearnData(1:3, :))); 
val_pred_y = round(trained_net(ValData(1:3, :)));
test_pred_y = round(trained_net(TestData));

save("Testlabel_a.mat", "test_pred_y");

%Computing MSE on data points:
train_perf = perform(trained_net, learn_pred_y, LearnData(4,:))
val_perf = perform(trained_net, val_pred_y, ValData(4,:))



%*****************Classification Scatters************************

figure('Name', "Training Data Classification");
subplot(211);
scatter3(class0_LearnData(1,:), class0_LearnData(2,:), class0_LearnData(3,:));
hold on;
scatter3(class1_LearnData(1,:), class1_LearnData(2,:), class1_LearnData(3,:));
title("Training Data Labels");
xlabel("X"); ylabel("Y"); zlabel("Z");
legend("Class 0", "Class 1");

subplot(212);
scatter3(LearnData(1, learn_pred_y==0), LearnData(2, learn_pred_y==0), LearnData(3, learn_pred_y==0));
hold on;
scatter3(LearnData(1, learn_pred_y==1), LearnData(2, learn_pred_y==1), LearnData(3, learn_pred_y==1));
title("Training Data Predicted Labels");
xlabel("X"); ylabel("Y"); zlabel("Z");
legend("Class 0", "Class 1");




figure('Name', "Validation Data Classification");
subplot(211);
scatter3(class0_ValData(1,:), class0_ValData(2,:), class0_ValData(3,:));
hold on;
scatter3(class1_ValData(1,:), class1_ValData(2,:), class1_ValData(3,:));
title("Validation Data Labels");
xlabel("X"); ylabel("Y"); zlabel("Z");
legend("Class 0", "Class 1");

subplot(212);
scatter3(ValData(1, val_pred_y==0), ValData(2, val_pred_y==0), ValData(3, val_pred_y==0));
hold on;
scatter3(ValData(1, val_pred_y==1), ValData(2, val_pred_y==1), ValData(3, val_pred_y==1));
title("Validation Data Predicted Labels");
xlabel("X"); ylabel("Y"); zlabel("Z");
legend("Class 0", "Class 1");




figure('Name', "Test Data Classification");
scatter3(TestData(1, test_pred_y==0), TestData(2, test_pred_y==0), TestData(3, test_pred_y==0));
hold on;
scatter3(TestData(1, test_pred_y==1), TestData(2, test_pred_y==1), TestData(3, test_pred_y==1));
title("Test Data Predicted Labels");
xlabel("X"); ylabel("Y"); zlabel("Z");
legend("Class 0", "Class 1");
%********************************************************************






%% Question2: b
clc;

n = size(TrainData, 2); %Total number of training and validation points
val_n = 0.2*n; %Number of validation points

valPoints_ind = sort(randperm(n, val_n)); %Validation points (random)

ValData = TrainData(:, valPoints_ind); %Validation data
LearnData = TrainData; LearnData(:, valPoints_ind) = []; %Training Data

%Seperating classes:
class0_ValData = ValData(1:3, ValData(4,:) == 0); %[3*90]
class1_ValData = ValData(1:3, ValData(4,:) == 1); %[3*90]
class0_LearnData = LearnData(1:3, LearnData(4,:) == 0); %[3*90]
class1_LearnData = LearnData(1:3, LearnData(4,:) == 1); %[3*90]



hiddenSize = 10; %The maximum number of neurons in the hidden layer
trainsNum = 7; %The number of trainings for each hiddenSize


train_perf_arr = zeros(2, hiddenSize); %Array of MSE values for training data points
val_perf_arr = zeros(2, hiddenSize); %Array of MSE values for validation data points

train_labels = [LearnData(4,:) == 0; LearnData(4,:) == 1];
val_labels = [ValData(4,:) == 0; ValData(4,:) == 1];

for i = 1:hiddenSize

    %Creating the network:
    net = feedforwardnet(i);
    net.divideParam.trainRatio = 1;
    net.divideParam.testRatio = 0;
    net.divideParam.valRatio = 0;
 
    train_tmp_perf_arr = zeros(2, length(trainsNum)); %MSE values for each training (on training data points)
    val_tmp_perf_arr = zeros(2, length(trainsNum)); %MSE values for each training (on validation data points)

    tmp_pred_y_arr = zeros(trainsNum, 2, n); %Predicted outputs for each training

    for j = 1:trainsNum
        trained_net = train(net, LearnData(1:3, :), train_labels); %Training the network using training data points
        tmp_pred_y_arr(j, :, :) = trained_net([LearnData(1:3, :),ValData(1:3, :)]); %Predicting the outputs using the network

        %Seperating the output of Learning points and Validation points:
        learn_tmp_pred_y_arr = squeeze(tmp_pred_y_arr(j, :, 1:(end-val_n)));
        val_tmp_pred_y_arr = squeeze(tmp_pred_y_arr(j, :, (end-val_n+1):end));

        %Computing MSE on validation data points:
        train_tmp_perf_arr(:,j) = perform(trained_net, learn_tmp_pred_y_arr, train_labels);
        val_tmp_perf_arr(:,j) = perform(trained_net, val_tmp_pred_y_arr, val_labels);
    end


    %Averaging MSE over different trainings:
    val_perf_arr(:,i) = mean(val_tmp_perf_arr,2);
    train_perf_arr(:,i) = mean(train_tmp_perf_arr,2);

end


%Plotting MSE for training and validation data points for different numbers of hidden neurons:
figure;

subplot(211);
plot(1:hiddenSize, train_perf_arr(1,:), 'Linewidth', 1);
hold on;
plot(1:hiddenSize, train_perf_arr(2,:), 'Linewidth', 1);
title("MSE for Training Data");
xlabel("Size of the Hidden Layer"); ylabel("Average MSE over " + trainsNum + " trainings");
legend('Output 1', 'Output 2');

subplot(212);
plot(1:hiddenSize, val_perf_arr(1,:), 'Linewidth', 1);
hold on;
plot(1:hiddenSize, val_perf_arr(2,:), 'Linewidth', 1);
title("MSE for Validation Data");
xlabel("Size of the Hidden Layer"); ylabel("Average MSE over " + trainsNum + " trainings");
legend('Output 1', 'Output 2');





%The best number of neurons = argmin(average MSE):
finalSize = find(mean(val_perf_arr,1) == min(mean(val_perf_arr,1)));

%Creating the final network:
net = feedforwardnet(finalSize);
net.divideParam.trainRatio = 1;
net.divideParam.testRatio = 0;
net.divideParam.valRatio = 0;

trained_net = train(net, LearnData(1:3, :), train_labels); %Training the network using training data points

%Predicting the outputs using the network:
learn_pred_y = trained_net(LearnData(1:3, :)); 
val_pred_y = trained_net(ValData(1:3, :));
test_pred_y = trained_net(TestData);
save("Testlabel_b.mat","test_pred_y");

%Computing MSE on training and validation data points:
train_perf = perform(trained_net, learn_pred_y, LearnData(4,:))
val_perf = perform(trained_net, val_pred_y, ValData(4,:))


%*****************Classification Scatters************************
figure('Name', "Training Data Classification");
subplot(211);
scatter3(class0_LearnData(1,:), class0_LearnData(2,:), class0_LearnData(3,:));
hold on;
scatter3(class1_LearnData(1,:), class1_LearnData(2,:), class1_LearnData(3,:));
title("Training Data Labels");
xlabel("X"); ylabel("Y"); zlabel("Z");
legend("Class 0", "Class 1");

subplot(212);
class0_ind = find(learn_pred_y(1,:) >= learn_pred_y(2,:));
class1_ind = find(learn_pred_y(1,:) < learn_pred_y(2,:));

scatter3(LearnData(1, class0_ind), LearnData(2, class0_ind), LearnData(3, class0_ind));
hold on;
scatter3(LearnData(1, class1_ind), LearnData(2, class1_ind), LearnData(3, class1_ind));
title("Training Data Predicted Labels");
xlabel("X"); ylabel("Y"); zlabel("Z");
legend("Class 0", "Class 1");



figure('Name', "Validation Data Classification");
subplot(211);
scatter3(class0_ValData(1,:), class0_ValData(2,:), class0_ValData(3,:));
hold on;
scatter3(class1_ValData(1,:), class1_ValData(2,:), class1_ValData(3,:));
title("Validation Data Labels");
xlabel("X"); ylabel("Y"); zlabel("Z");
legend("Class 0", "Class 1");

subplot(212);
class0_ind = find(val_pred_y(1,:) >= val_pred_y(2,:));
class1_ind = find(val_pred_y(1,:) < val_pred_y(2,:));

scatter3(ValData(1, class0_ind), ValData(2, class0_ind), ValData(3, class0_ind));
hold on;
scatter3(ValData(1, class1_ind), ValData(2, class1_ind), ValData(3, class1_ind));
title("Validation Data Predicted Labels");
xlabel("X"); ylabel("Y"); zlabel("Z");
legend("Class 0", "Class 1");




figure('Name', "Test Data Classification");
class0_ind = find(test_pred_y(1,:) >= test_pred_y(2,:));
class1_ind = find(test_pred_y(1,:) < test_pred_y(2,:));

scatter3(TestData(1, class0_ind), TestData(2, class0_ind), TestData(3, class0_ind));
hold on;
scatter3(TestData(1, class1_ind), TestData(2, class1_ind), TestData(3, class1_ind));
title("Test Data Predicted Labels");
xlabel("X"); ylabel("Y"); zlabel("Z");
legend("Class 0", "Class 1");
%*************************************************************************







