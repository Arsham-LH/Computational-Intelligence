%% Q1: a and b
clc;

%********Defining x and y arrays***************
xstep = 0.01;
xmin = -1;
xmax = 1;
x_arr = xmin:xstep:xmax;

ymin = -1;
ymax = 1;
ystep = 0.01;
y_arr = ymin:ystep:ymax;
%**********************************************


%*********Defining initial values for parameters:*********
beta = 1;
w1_init = 1;
w2_init = 1;
T_init = 1;
%************************************************

Q1(x_arr, y_arr, beta, w1_init, w2_init, T_init); %Calling function related to Question1

%% Q1:c

clc;

%********Defining x and y arrays***************
xstep = 0.01;
xmin = -1;
xmax = 1;
x_arr = xmin:xstep:xmax;

ymin = -1;
ymax = 1;
ystep = 0.01;
y_arr = ymin:ystep:ymax;
%**********************************************


%*********Defining initial values for parameters*********
beta = 100;
%For OR function, we set:
w1_init = 2;
w2_init = 2;
T_init = 1;
%************************************************
Q1(x_arr, y_arr, beta, w1_init, w2_init, T_init); %Calling function related to Question1




%% Q1:d

clc;

%********Defining x and y arrays***************
xstep = 0.01;
xmin = -1;
xmax = 1;
x_arr = xmin:xstep:xmax;

ymin = -1;
ymax = 1;
ystep = 0.01;
y_arr = ymin:ystep:ymax;
%**********************************************


%*********Defining initial values for parameters*********
beta = 0.01;
%For OR function, we set:
w1_init = 2;
w2_init = 2;
T_init = 1;
%************************************************
Q1(x_arr, y_arr, beta, w1_init, w2_init, T_init); %Calling function related to Question1

%% functions

function Q1(x_arr, y_arr, beta, w1_init, w2_init, T_init)
    
    f = figure;
    
    [w1, w2, T] = createPanels(f, w1_init, w2_init, T_init); %Creating interactive panels for parameters

    %Initializing parameters:
    w1.Value = w1_init;
    w2.Value = w2_init;
    T.Value = T_init;

    %Defining Callback function in order to update the figure:
    w1.Callback = @selection;
    w2.Callback = @selection;
    T.Callback = @selection;

    
    z = activation(w1.Value, w2.Value, x_arr, y_arr, T.Value, beta); %Computing the output of the network
    
    %**********Plotting surface (output vs inputs) with current parameters:********
    hAxes = axes('Parent',f);
    hPlot = surf(hAxes,x_arr,y_arr,z);
    title("Output of the nueron vs inputs. beta = "+beta);
    xlabel('X');
    ylabel('Y');
    zlabel('Output');
    colorbar;
    rotate3d on;
    %**************************************************
    
    
    %Defining the callback function (to be run after changing parameters)
    function selection(src, event)
            %Storing updated values of the parameters in 3 new variables:
            w1_val = w1.Value;
            w2_val = w2.Value;
            T_val = T.Value;

            z = activation(w1.Value, w2.Value, x_arr, y_arr, T.Value, beta);%Re-computing the output of the network with updated parameters

            clf(f); %Clearing previous figure

            
            [w1, w2, T] = createPanels(f, w1_val, w2_val, T_val); %Creating new interactive panels
            
            %Defining callback function:
            w1.Callback = @selection;
            w2.Callback = @selection;
            T.Callback = @selection;

            %Setting current values as updated values
            w1.Value = w1_val;
            w2.Value = w2_val;
            T.Value = T_val;

            %**********Plotting surface (output vs inputs) with current parameters:********
            hAxes = axes('Parent',f);
            hPlot = surf(hAxes,x_arr,y_arr,z);
            title("Output of the nueron vs inputs. beta = "+beta);
            xlabel('X');
            ylabel('Y');
            zlabel('Output');
            colorbar;
            rotate3d on;
            %*******************************************************
    end
end



function act = activation(w1, w2, x_arr, y_arr, T, beta)
%This function computes output of the network with inputs as vectors of x_arr & y_arr
%act Dimension = length(y_arr) * length(x_arr)
%act(i,j) = output for y=y_arr(i) & x=x_arr(j)

    act = zeros(length(y_arr), length(x_arr));

    for i = 1:length(y_arr) %For each x
        for j = 1:length(x_arr) %For each y
            net = w1*x_arr(j) + w2*y_arr(i);
            act(i,j) = 1/(1+exp(-beta*(net-T)));
        end
    end
end


function [w1, w2, T] = createPanels(f, w1_val, w2_val, T_val)
%This function creates interactive panels for parameters w1, w2, T

    %Creating panels:
    p1 = uipanel(f,'Position',[0.05 0 0.3 0.075], 'BackgroundColor', 'white', 'Title',"W1 = "+w1_val);
    p2 = uipanel(f,'Position',[0.35 0 0.3 0.075], 'BackgroundColor', 'white', 'Title',"W2 = "+w2_val);
    p3 = uipanel(f,'Position',[0.65 0 0.3 0.075], 'BackgroundColor', 'white', 'Title',"T = "+T_val);
    
    %*************Creating slider controls:*******************
    w1 = uicontrol(p1,'Style','slider', 'Value',w1_val, 'Position',[0,0,0,0]);
    w1.Min = -5; %Setting minimum value in slider
    w1.Max = 5; %Setting maximum value in slider
    
    
    w2 = uicontrol(p2,'Style','slider', 'Value',w2_val, 'Position',[0,0,0,0]);
    w2.Min = -5;
    w2.Max = 5;
    
    
    T = uicontrol(p3,'Style','slider', 'Value',T_val, 'Position',[0,0,0,0]);
    T.Min = -5;
    T.Max = 5;
    %*********************************************************

    %Changing units to normalized, in order to set the position adaptively:
    h = findobj(f, 'Type', 'uicontrol');
    set(h, 'Units', 'normalized');
    set(h, 'Position', [0.1 0.1 0.8 0.8]);

end
