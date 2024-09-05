%% Q2: A
clc; close all;
table = readtable('iris.csv'); %Imporitng data from excel
data = table2array(table(:, 1:end-1)); %Dimension = 150*4

pointsPerClass = 50; % number of data points available for each class
properties = 4; %Number of properties available for each flower

slct_data = data(1:2*pointsPerClass, :); %Keeping only the first two classes. Dimension = 100*4


class1_data = slct_data(1:pointsPerClass, :); %Data for class1. Dimension = 50*4
class2_data = slct_data(pointsPerClass+1:end, :); %Data for class2. Dimension = 50*4
for i = 1:properties-1 %Each property (as the first property to be shown)
    for j = i+1:properties %Each property (as the second property to be shown)
        figure;
        class1_x = class1_data(:, i); %Defining x-axis for points from class1
        class1_y = class1_data(:, j); %Defining y-axis for points from class1
        scatter(class1_x, class1_y, 'blue', 'Linewidth', 0.5);

        hold on;
        class2_x = class2_data(:, i);
        class2_y = class2_data(:, j);
        scatter(class2_x, class2_y, 'red', 'Linewidth', 0.5);

        title("Properties scatter for class 1&2, Properties "+i+"&"+j);
        xlabel("Property "+i);
        ylabel("Property "+j);
        l = legend('Class 1', 'Class 2', 'Location','best');
        set(l, 'Color' , 'yellow');
    end
end



%% Q2: B
clc;
sampleSize = 5; %Number of sample points

% %Selecting randomly 5 data points from each class, and keeping only selected properties (3 & 4):
% class1_randPoints = randperm(pointsPerClass, sampleSize);
% class2_randPoints = randperm(pointsPerClass, sampleSize);
class1_randPoints = cell2mat(struct2cell(load('class1_randPoints_Q2b.mat')));
class2_randPoints = cell2mat(struct2cell(load('class2_randPoints_Q2b.mat')));

class1_sampleData = class1_data(class1_randPoints, [3,4]); %Dimension = 5*2
class2_sampleData = class2_data(class2_randPoints, [3,4]); %Dimension = 5*2

figure;
sample_class1_x = class1_sampleData(:, 1); %Defining x-axis for points from class1
sample_class1_y = class1_sampleData(:, 2); %Defining y-axis for points from class1
scatter(sample_class1_x, sample_class1_y, 'blue', 'Linewidth', 0.5);

hold on;
sample_class2_x = class2_sampleData(:, 1); %Defining x-axis for points from class1
sample_class2_y = class2_sampleData(:, 2); %Defining y-axis for points from class1
scatter(sample_class2_x, sample_class2_y, 'red', 'Linewidth', 0.5);

title("Properties scatter for class 1&2, Properties "+i+"&"+j);
xlabel("Property "+i);
ylabel("Property "+j);
xlim([1,5.5]);
ylim([0,1.8]);
l = legend('Class 1', 'Class 2', 'Location','best');
set(l, 'Color' , 'yellow');


%% Q2: C,D,E - online
clc;


sampleSize = 0.8 * pointsPerClass; %number of learning points (80%)

%**********Selecting random learning points, and saving them for subsequent runs*****
% class1_learnPoints = randperm(pointsPerClass, sampleSize); 
% class2_learnPoints = randperm(pointsPerClass, sampleSize);
% save('class1_learnPoints.mat', 'class1_learnPoints');
% save('class2_learnPoints.mat', 'class2_learnPoints');
%************************************************************

%Loading previously saved random points:
class1_learnPoints = cell2mat(struct2cell(load('class1_learnPoints.mat')));
class2_learnPoints = cell2mat(struct2cell(load('class2_learnPoints.mat')));

%Selecting data points using random numbers (For properties 3&4)
class1_learnData = class1_data(class1_learnPoints, [3,4]); %Dimension = 40*2
class2_learnData = class2_data(class2_learnPoints, [3,4]); %Dimension = 40*2

%Mixing selected data of the two classes, for creating a learning task. (Data is shuffled in order to mix the two classes)
total_learnData = [class1_learnData;class2_learnData]; %Dimension = 80*2
total_learnData = total_learnData(randperm(size(total_learnData,1)), :); %Dimension = 80*2

%Optimum values:
o = ismember(total_learnData(:,1), class2_learnData(:,1)) &...
ismember(total_learnData(:,2), class2_learnData(:,2)); %Dimension = 80*1

%Setting initial values and learning rate:
w1_init = -1;
w2_init = 1;
theta_init = 1;
eta = 0.1;

%Defining the maximum number of iterations
iterNum = 10;

%Defining current values of the parameters:
w1 = w1_init;
w2 = w2_init;
theta = theta_init;

%Defining arrays for the parameters (in order to sketch them. For part D)
w1_arr = ones(iterNum+1, 1)*w1;
w2_arr = ones(iterNum+1, 1)*w2;
theta_arr = ones(iterNum+1, 1)*theta;

e = 0; %error value in each epoch
for it=1:iterNum
    e = 0;
    for l = 1:size(total_learnData, 1)
        %Selecting current values of the inputs:
        x1 = total_learnData(l, 1); 
        x2 = total_learnData(l, 2);

        %Calculating the output of the network:
        y = (w1*x1 + w2*x2 >= theta);

        %Updating the values of the parameters, in case an error has occured
        if y ~= o(l)
            theta = theta - eta*(o(l) - y);
            w1 = w1 + (o(l) - y) * x1;
            w2 = w2 + (o(l) - y) * x2;
            e = e + abs(o(l) - y);
        end
    end

    %Storing current values in the arrays:
    w1_arr(it+1:end) = w1;
    w2_arr(it+1:end) = w2;
    theta_arr(it+1:end) = theta;

    %Ending the process, if no error exists:
    if (e == 0)
        break;
    end
end


%******************PART D: parameters vs epoch number********************
figure;
plot(1:iterNum+1, w1_arr, 'LineWidth',1);
hold on;
plot(1:iterNum+1, w2_arr, 'LineWidth',1);
plot(1:iterNum+1, theta_arr, 'LineWidth',1);

title('Value of the parameters vs iteration');
xlabel('Epoch number');
ylabel('Parameters values');
legend('w1', 'w2', '\theta', 'Location', 'best');
%**********************************************************




%*******************PART E: learning points and fitted line****************
figure;

%Sketching learning points:
scatter(class1_learnData(:, 1), class1_learnData(:, 2), 'blue', 'Linewidth', 0.5);
hold on;
scatter(class2_learnData(:, 1), class2_learnData(:, 2), 'red', 'Linewidth', 0.5);

%Plotting the fitted line:
x1_arr = 0:0.1:6;
x2_arr = (-w1*x1_arr + theta)/w2;
plot(x1_arr, x2_arr, 'Linewidth', 1);


title("Properties scatter for class 1&2, Properties "+i+"&"+j);
xlabel("Property "+i + " (x1)");
ylabel("Property "+j + " (x2)");
xlim([1,5.5]);
ylim([0,1.8]);
l = legend('Class 1', 'Class 2', 'Fitted line', 'Location','best');
set(l, 'Color' , 'yellow');
%*********************************************************************







%% Q2: C,D,E - batch
clc;
sampleSize = 0.8 * pointsPerClass;

%**********Selecting random learning points, and saving them for subsequent runs*****
% class1_learnPoints = randperm(pointsPerClass, sampleSize); 
% class2_learnPoints = randperm(pointsPerClass, sampleSize);
% save('class1_learnPoints.mat', 'class1_learnPoints');
% save('class2_learnPoints.mat', 'class2_learnPoints');
%************************************************************

class1_learnPoints = cell2mat(struct2cell(load('class1_learnPoints.mat')));
class2_learnPoints = cell2mat(struct2cell(load('class2_learnPoints.mat')));

class1_learnData = class1_data(class1_learnPoints, [3,4]); %Dimension = 40*2
class2_learnData = class2_data(class2_learnPoints, [3,4]); %Dimension = 40*2

total_learnData = [class1_learnData;class2_learnData]; %Dimension = 80*2
total_learnData = total_learnData(randperm(size(total_learnData,1)), :); %Dimension = 80*2

o = ismember(total_learnData(:,1), class2_learnData(:,1)) & ismember(total_learnData(:,2), class2_learnData(:,2)); %Optimum value. Dimension = 80*1

w1_init = -1;
w2_init = 1;
theta_init = 1;

eta = 0.1;

iterNum = 50;


w1 = w1_init;
w2 = w2_init;
theta = theta_init;


w1_arr = ones(iterNum+1, 1)*w1;
w2_arr = ones(iterNum+1, 1)*w2;
theta_arr = ones(iterNum+1, 1)*theta;

e = 0; %error value in each epoch
for it=1:iterNum
    %Defining parameters for storing all changes, in order to sum them up, 
    % until the end of the iteration
    w1_c = 0;
    w2_c = 0;
    theta_c = 0;

    e = 0;
    for l = 1:size(total_learnData, 1)
        x1 = total_learnData(l, 1);
        x2 = total_learnData(l, 2);
        y = (w1*x1 + w2*x2 >= theta);
        if y ~= o(l)
            theta_c = theta_c - eta*(o(l) - y);
            w1_c = w1_c + (o(l) - y) * x1;
            w2_c = w2_c + (o(l) - y) * x2;
            e = e + abs(o(l) - y);
        end
    end

    %Updating current values, ONLY at the end of the iteration:
    w1 = w1+w1_c;
    w2 = w2+w2_c;
    theta = theta + theta_c;

    w1_arr(it+1:end) = w1;
    w2_arr(it+1:end) = w2;
    theta_arr(it+1:end) = theta;

    if (e == 0)
        break;
    end
end


%******************PART D: parameters vs epoch number********************
figure;
plot(1:iterNum+1, w1_arr, 'LineWidth',1);
hold on;
plot(1:iterNum+1, w2_arr, 'LineWidth',1);
plot(1:iterNum+1, theta_arr, 'LineWidth',1);

title('Value of the parameters vs iteration');
xlabel('Epoch number');
ylabel('Parameters values');
legend('w1', 'w2', '\theta', 'Location', 'best');
%**********************************************************




%*******************PART E: learning points and fitted line****************
figure;

scatter(class1_learnData(:, 1), class1_learnData(:, 2), 'blue', 'Linewidth', 0.5);
hold on;
scatter(class2_learnData(:, 1), class2_learnData(:, 2), 'red', 'Linewidth', 0.5);

x1_arr = 0:0.1:6;
x2_arr = (-w1*x1_arr + theta)/w2;
plot(x1_arr, x2_arr, 'Linewidth', 1);


title("Properties scatter for class 1&2, Properties "+i+"&"+j);
xlabel("Property "+i + " (x1)");
ylabel("Property "+j + " (x2)");
xlim([1,5.5]);
ylim([0,1.8]);
l = legend('Class 1', 'Class 2', 'Fitted line', 'Location','best');
set(l, 'Color' , 'yellow');
%*********************************************************************


%% Q2: F (testing phase)

%**************Please set the final values for the network:**************
w1_final = -13.3;
w2_final = 155.5;
theta_final = 48.6;
%**************************************************************

%Finding test points (points that are not learning points!):
class1_testPoints = 1:pointsPerClass; class1_testPoints(class1_learnPoints) = [];
class2_testPoints = 1:pointsPerClass; class2_testPoints(class2_learnPoints) = [];

%Selecting data points:
class1_testData = class1_data(class1_testPoints, [3,4]); %Dimension = 10*2
class2_testData = class2_data(class2_testPoints, [3,4]); %Dimension = 10*2

%Mixing selected data of the two classes.
total_testData = [class1_testData;class2_testData]; %Dimension = 20*2

%Defining optimum values:
o_test = ismember(total_testData(:,1), class2_testData(:,1)) &...
ismember(total_testData(:,2), class2_testData(:,2)); %Dimension = 20*1

e_test = 0; %Total error in the test phase
for t=1:size(total_testData,1)
    x1 = total_testData(t, 1);
    x2 = total_testData(t, 2);
    
    y = (w1_final*x1 + w2_final*x2 - theta_final >=0 );
    if y ~= o_test(t)
        e_test = e_test + abs(o_test(t)-y);
    end
end
disp("e_test = "+e_test);

figure('Name', 'Test Phase');
%Sketching learning points:
scatter(class1_data(:, 3), class1_data(:, 4), 'blue', 'Linewidth', 0.5);
hold on;
scatter(class2_data(:, 3), class2_data(:, 4), 'red', 'Linewidth', 0.5);

%Plotting the fitted line:
x1_arr = 0:0.1:6;
x2_arr = (-w1_final*x1_arr + theta_final)/w2_final;
plot(x1_arr, x2_arr, 'Linewidth', 1);


title("Properties scatter for class 1&2, Properties "+i+"&"+j);
xlabel("Property "+i + " (x1)");
ylabel("Property "+j + " (x2)");
ylim([0,1.8]);
l = legend('Class 1', 'Class 2', 'Fitted line', 'Location','best');
set(l, 'Color' , 'yellow');



%% Q2: G
clc;

class1_data = data(1:pointsPerClass, :); %Data for class1. Dimension = 50*4
class2_data = data(pointsPerClass+1:2*pointsPerClass, :); %Data for class2. Dimension = 50*4
class3_data = data(2*pointsPerClass+1:end, :); %Data for class3. Dimension = 50*4

for i = 1:properties-2 %Each property (as the first property to be shown)
    for j = i+1:properties-1 %Each property (as the second property to be shown)
        for k = j+1:properties %Each property (as the third property to be shown)
            figure;
            class1_x = class1_data(:, i); %Defining x-axis for points from class1
            class1_y = class1_data(:, j); %Defining y-axis for points from class1
            class1_z = class1_data(:, k); %Defining z-axis for points from class1
            scatter3(class1_x, class1_y, class1_z, 'blue', 'Linewidth', 0.5);

            hold on;
            class2_x = class2_data(:, i);
            class2_y = class2_data(:, j);
            class2_z = class2_data(:, k);
            scatter3(class2_x, class2_y, class2_z, 'red', 'Linewidth', 0.5);

            class3_x = class3_data(:, i);
            class3_y = class3_data(:, j);
            class3_z = class3_data(:, k);
            scatter3(class3_x, class3_y, class3_z, 'green', 'Linewidth', 0.5);

            title("Properties scatter for alll 3 classes, Properties "+i+"&"+j+"&"+k);
            xlabel("Property "+i);
            ylabel("Property "+j);
            zlabel("Property "+k);
            l = legend('Class 1', 'Class 2', 'Class 3', 'Location','best');
            set(l, 'Color' , 'yellow'); 
        end
    end
end




