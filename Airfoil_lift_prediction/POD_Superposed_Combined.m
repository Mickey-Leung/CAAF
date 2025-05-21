%%
clear;clc;close all;

% Set validation case and input parameters
datasets = ["airfoil_5","airfoil_11","cylinder_5","cylinder_11","cylinder_0_5_1"];
case_name = ["None-5", "None-11", "Cylinder-5","Cylinder-11","Cylinder2-0"];
case_index = 5; % specify validation case index
train_index = [5]; % specify training case index
nt_train = 10000;
is_indiv = 0; % Is individual training case?

p_list = [2,4,6,8,10]; % number of sensors on each surface

colorRange1 = [-0.5 0.5;-0.5 0.5;-1 0.2; -1 0.2; -1 0.2]; % colormap range for plotting average pressure reconstruction
colorRange2 =[0 0.004; 0 0.006; 0 0.015; 0 0.015; 0 0.015]; % colormap range for plotting error of pressure reconstruction
LiftRange =[0 0.2; 0 0.8; -1 1; -1 1; -3 3]; % Y range of lift plot
% t_n = 1; % time snapshot for reconstruction plot
% -------------------------------------------------------------------------
% Load files and find optimal sensor locations via POD

% Load data for upper surface
data1 = load(datasets(case_index)+"_wall1_surfacepressure_span.dat");
coord1 = load(datasets(case_index)+"_xy_sort_wall1.dat");
nx = coord1(1,1);
nt = size(data1,1);
% nt_train = nt; % ONLY FOR SINGLE CASE
time = data1(:,1);
time_range = range(time);
coord1 = coord1(2:nx+1,:);

if is_indiv
    train_index = case_index;
    nt_train = nt;
end

% Get coordinates and pressure for the validation case specified by
% case_index
X1=coord1(:,1); % X,Y coordinates in ascending order
Y1=coord1(:,2);
P1=data1(:,2:end);

% Use selected grid points of cylinder-5 as sensor coordinates
coord_temp = load(datasets(3)+"_xy_sort_wall1.dat");
nx_train = coord_temp(1,1);
coord_temp = coord_temp(2:nx_train+1,:);
X1_train = coord_temp(:,1);
Y1_train = coord_temp(:,2);

P1_all = [];
for i = train_index
    
    data1 = load(datasets(i)+"_wall1_surfacepressure_span.dat");
    data_temp = zeros(nt_train,nx_train+1);
    
    if i == 3
        P1_all=[P1_all;data1(1:nt_train,2:end)];
        continue;
    end
        
    coord_temp = load(datasets(i)+"_xy_sort_wall1.dat");
    coord_temp = coord_temp(2:coord_temp(1,1)+1,:);
    X_temp = coord_temp(:,1); % X coordinates for current case's grid
    
%     if size(data2,2) ~= nx_train+1
%     X_temp = X2;
%     if size(X2,1) ~= size(data2,2)-1
%         X_temp = X2_new;
%     end
    for t = 1:nt_train
        data_temp(t,2:end) = interp1(X_temp,data1(t,2:end)',X1_train,'spline','extrap')'; % Interpolate to train grid 
    end
%     data2 = data_temp;
%     end
    P1_all=[P1_all;data_temp(1:nt_train,2:end)];
end
% -------------------------------------------------------------------------

% Load data for lower surface
data2 = load(datasets(case_index)+"_wall2_surfacepressure_span.dat");
coord2 = load(datasets(case_index)+"_xy_sort_wall2.dat");
nx = coord2(1,1);
coord2 = coord2(2:nx+1,:);

% Get coordinates and pressure for the validation case specified by
% case_index
X2=coord2(:,1); % X,Y coordinates in ascending order
Y2=coord2(:,2);
P2=data2(:,2:end);
% P2=data2(nt_train:end,2:end);

% Use selected grid points for cylinder-5 as sensor coordinates
coord_temp = load(datasets(3)+"_xy_sort_wall2.dat");
coord_temp = coord_temp(2:nx_train+1,:);
X2_train = coord_temp(:,1);
Y2_train = coord_temp(:,2);

P2_all = [];
for i = train_index
    
    data2 = load(datasets(i)+"_wall2_surfacepressure_span.dat");
    data_temp = zeros(nt_train,nx_train+1);
    
    if i == 3
        P2_all=[P2_all;data2(1:nt_train,2:end)];
        continue;
    end
        
    coord_temp = load(datasets(i)+"_xy_sort_wall2.dat");
    coord_temp = coord_temp(2:coord_temp(1,1)+1,:);
    X_temp = coord_temp(:,1); % X coordinates for current case's grid
    
%     if size(data2,2) ~= nx_train+1
%     X_temp = X2;
%     if size(X2,1) ~= size(data2,2)-1
%         X_temp = X2_new;
%     end
    for t = 1:nt_train
        data_temp(t,2:end) = interp1(X_temp,data2(t,2:end)',X2_train,'spline','extrap')'; % Interpolate to train grid 
    end
%     data2 = data_temp;
%     end
    P2_all=[P2_all;data_temp(1:nt_train,2:end)];
end

% Concatenate two surfaces
P_all = [P1_all,P2_all];

for p = p_list
    r = p; % given rank
    % Find optimal sensor locations using POD
    [Psi,S,V]=svd(P_all','econ'); % SVD on A transpose
    Psi_r=Psi(:,1:r); % get first r left singular vectors
    if (p==r)
        % QR sensor selection, p=r
        [Q,R,pivot] = qr(Psi_r','vector');
    elseif (p>r)
        % Oversampled QR sensors, p>r
        [Q,R,pivot] = qr(Psi_r*Psi_r','vector');
    end
    pivot_train = pivot(1:p); % pivot indices for p sensors

    % Sort pivot indeces into upper or lower surface
    pivot = [];  % pivot indices for 2 surfaces in validation grid
    pivot1_train = [];
    pivot2_train = [];
    X_sensor_all = [];
    Y_sensor_all = [];
    uniform = [0,0.2,0.4,0.6,1.0];
    for i = 1:p
        if pivot_train(i) <= nx_train
            % surface 1
            pivot1_train(end+1) = pivot_train(i);
            % Find nearest location on the grid points of the validation case
            [~,pivot(end+1)] = min(abs(X1_train(pivot_train(i))-X1));
            X_sensor_all(end+1) = X1_train(pivot_train(i));
            Y_sensor_all(end+1) = Y1_train(pivot_train(i));
            % [~,index]=min(abs([1/8 2/8 3/8 4/8 5/8 6/8 7/8]-X1));
            % disp(index)
        else
            % surface 2
            pivot2_train(end+1) = pivot_train(i)-nx_train;
            % Find nearest location on the grid points of the validation case
            [~,pivot_temp] = min(abs(X2_train(pivot_train(i)-nx_train)-X2));
            pivot(end+1) = pivot_temp + nx_train; % index for surface 2 must + nx_train
            X_sensor_all(end+1) = X2_train(pivot_train(i)-nx_train);
            Y_sensor_all(end+1) = Y2_train(pivot_train(i)-nx_train);
            % [~,index]=min(abs([1/8 2/8 3/8 4/8 5/8 6/8 7/8]-X2));
            % disp(index+nx_train)
        end
    end

    X_sensor1=X1_train(pivot1_train);
    Y_sensor1=Y1_train(pivot1_train);

    X_sensor2=X2_train(pivot2_train);
    Y_sensor2=Y2_train(pivot2_train);
    %% ------------------------------------------------------------------------

    % Pressure Reconstruction
    % 2 surfaces
    C = zeros(p,2*nx_train); % Create measurement matrix C: y = Cx
    for i = 1:p
       C(i,pivot_train(i)) = 1; 
    end

    % Compute reconstructed pressure matrix
    P12 = [P1,P2];
    P_rec_train = zeros(size(P12,1),2*nx_train); % Reconstructed pressure matrix
    P_rec_1 = zeros(size(P1,1),nx); % Reconstructed pressure for surface 1 interpolated to validation grid
    P_rec_2 = zeros(size(P2,1),nx); % Reconstructed pressure for surface 2 interpolated to validation grid
    for t = 1:size(P12,1)
       a = (C*Psi_r)\P12(t,pivot)'; % compute pseudoinverse
       P_rec_train(t,:) = Psi_r*a;

       P_rec_1(t,:) = interp1(X1_train,P_rec_train(t,1:nx_train)',X1,'spline','extrap'); % Interpolate to validation grid
       P_rec_2(t,:) = interp1(X2_train,P_rec_train(t,nx_train+1:end)',X2,'spline','extrap'); % Interpolate to validation grid
    end
    P_rec_1_t = mean(P_rec_1,1); % Time averaged reconstructed pressure
%     P_rec_1_t = P_rec_1(1,:); % Instantneous reconstructed pressure
    P1_t = mean(P1,1); % Time averaged simulated pressure
    error1 = vecnorm(P_rec_1 - P1)./rms(P1,1)./nt; % L2 norm of the pressure difference at each location normalized by the RMS * nt

    P_rec_2_t = mean(P_rec_2,1); % Time averaged reconstructed pressure
%     P_rec_2_t = P_rec_2(1,:); % Instantneous
    P2_t = mean(P2,1); % Time averaged simulated pressure
    error2 = vecnorm(P_rec_2 - P2)./rms(P2,1)./nt;

    error = max([error1,error2],[],'all'); % report max error

    %% ------------------------------------------------------------------------
    % Lift Reconstruction
    % Upper surface
    CL_rec_temp = zeros(size(P12,1),1);
    CL_rec = zeros(size(P12,1),1);
    CD_rec = zeros(size(P12,1),1);
    for t = 1:size(P12,1)
        LF_1 = -trapz(X1,P_rec_1(t,:));   % Compute lift force using trapezoidal integration
        LF_2 = trapz(X2,P_rec_2(t,:));
        CL_temp = (LF_1 + LF_2);   % Compute CL: LF/(0.5*rho*u^2*A)
        CL_rec(t) = CL_temp;
%         CL_rec_temp(t) = CL_temp;
%         
%         DF_1 = trapz(Y1,P_rec_1(t,:));   % Compute drag force using trapezoidal integration
%         DF_2 = trapz(Y2,P_rec_2(t,:));
%         CD_temp = abs(DF_1 + DF_2);   % Compute CL: LF/(0.5*rho*u^2*A)
%         CD_rec(t) = CD_temp;
%         
%         CL_rec(t) = (CD_rec(t)*cos(5/180*pi)-CL_rec_temp(t)*sin(5/180*pi))/(0.5*0.1);
    end

    % Load CL
    CL = readmatrix(datasets(case_index)+"_CL");
    time_CL = CL(:,1);
    CL = CL(:,2);
    CL_q=interp1(time_CL,CL,time,'spline','extrap'); % Interpolate CL to time steps used for pressure
%     CL_q = CL_q(nt_train:end,:); % For testing split

%     error_CL = norm(CL_rec - CL_q)/rms(CL_q)/nt; % compute lift reconstruction error
    error_CL = norm(CL_rec - CL_q)/norm(CL_q-mean(CL_q)); % compute lift reconstruction error

    %% ------------------------------------------------------------------------
    % Visualization

    % Sensor Locations
    % ColorMaps;
    % ColorMaps2;
    figure 
    box on
    hold on
    plot(X1,Y1,'Color','blue', "lineWidth", 1);
    plot(X_sensor1,Y_sensor1,'x','MarkerSize',20,'MarkerEdgeColor',[1 0 0], "lineWidth", 2);
    plot(X2,Y2,'Color','blue', "lineWidth", 1);
    plot(X_sensor2,Y_sensor2,'x','MarkerSize',20,'MarkerEdgeColor',[1 0 0], "lineWidth", 2);
    xlim([-0.1 1.1]);
    ylim([-0.15 0.15]);
    xlabel('x/c');
    ylabel('y/c');
    labels = cellstr(string(1:p));
    set(gcf,'Position',[100 100 1500 500]);
    set(gca,'FontSize',30);
    title(p+" Sensors", 'Interpreter', 'none');
    labelpoints(X_sensor_all,Y_sensor_all,labels,'N',0.5,'FontSize', 18);
%     saveas(gcf,"POD Sensors/"+datasets(case_index)+"_POD_"+p+".png");
%     saveas(gcf,"POD_Sensors/Train-123/Sensors_"+p+".png");
    saveas(gcf,"POD_Sensors/Indiv_Case/"+case_name(case_index)+"_POD_"+p+".png");
    % -------------------------------------------------------------------------
%     % Pressure reconstruction
%     % Plot pressure at surface as colormap
%     figure
%     box on
%     hold on
%     colormap(parula);
% 
%     subplot(2,1,1);
%     plot(X1,Y1,'Color','blue', "lineWidth", 1);
%     plot(X2,Y2,'Color','blue', "lineWidth", 1);
%     z = zeros(size(X1));
%     surface([X1,X1]',[Y1,Y1]',[z,z]',[P_rec_1_t;P_rec_1_t],...
%     'facecol','no','edgecol','interp','linew',5);
%     z = zeros(size(X2));
%     surface([X2,X2]',[Y2,Y2]',[z,z]',[P_rec_2_t;P_rec_2_t],...
%     'facecol','no','edgecol','interp','linew',5);
%     colorbar; 
%     caxis(colorRange1(case_index,:));
%     xlim([-0.1 1.1]);
%     ylim([-0.15 0.15]);
%     xlabel('x/c');
%     ylabel('y/c');
%     title("Averaged Pressure Reconstruction with " +p+ " sensors");
% %     title("Instantaneous Pressure Reconstruction with " +p+ " sensors");
%     set(gca,'FontSize',24);
% 
%     subplot(2,1,2); 
%     plot(X1,Y1,'Color','blue', "lineWidth", 1);
%     plot(X2,Y2,'Color','blue', "lineWidth", 1);
%     z = zeros(size(X1));
%     surface([X1,X1]',[Y1,Y1]',[z,z]',[P1_t;P1_t],...
%     'facecol','no','edgecol','interp','linew',5);
%     z = zeros(size(X2));
%     surface([X2,X2]',[Y2,Y2]',[z,z]',[P2_t;P2_t],...
%     'facecol','no','edgecol','interp','linew',5);
%     xlim([-0.1 1.1]);
%     ylim([-0.15 0.15]);
%     xlabel('x/c');
%     ylabel('y/c');
%     title("Original Data");
%     % colormap(red_blue);
%     colorbar;
%     caxis(colorRange1(case_index,:));
%     % caxis([-1 0.2]) 
%     set(gcf,'Position',[100 100 1500 1000]);
%     set(gca,'FontSize',24);
% 
%     % Plot pressure difference
%     figure
%     box on
%     hold on
% 
%     xlabel('x/c');
%     ylabel('y/c');
%     plot(X1,Y1,'Color','blue', "lineWidth", 1);
%     plot(X2,Y2,'Color','blue', "lineWidth", 1);
%     z = zeros(size(X1));
%     surface([X1,X1]',[Y1,Y1]',[z,z]',[error1;error1],...
%     'facecol','no','edgecol','interp','linew',5);
%     z = zeros(size(X2));
%     surface([X2,X2]',[Y2,Y2]',[z,z]',[error2;error2],...
%     'facecol','no','edgecol','interp','linew',5); 
%     xlim([-0.1 1.1]);
%     ylim([-0.15 0.15]);
%     title("Reconstruction Error with " +p+ " sensors");
% 
%     % colormap(red_blue_heavy);
%     colormap(parula);
%     colorbar;
%     caxis(colorRange2(case_index,:)); 
%     set(gcf,'Position',[100 100 1500 500]);
%     annotation('textbox',[0.60 0.36 0.2 0.025],'String',...
%     "max error = "+round(error,3,'significant'),'FitBoxToText','on','FontSize',30);
%     set(gca,'FontSize',30);

    % ------------------------------------------------------------------------
    % Lift Reconstruction
    figure
    hold on
    box on
    plot(time,CL_rec,'b','linewidth',2);
    plot(time,CL_q,':','color','r','linewidth',2);
%     plot(time(nt_train:end),CL_rec,'b','linewidth',2);
%     plot(time(nt_train:end),CL_q,':','color','r','linewidth',2);
%     legend('Reconstructed CL','Simulated CL','location','best');
    xlabel('Time $tU_{\infty}/c$','Interpreter','latex');
    ylabel('$C_{L}$','Interpreter','latex');
    xlim([50 60]);
    ylim([-2 3]);
%     title('Lift Coefficient for '+case_name(case_index)+" with "+p+ " Sensors", 'Interpreter', 'none');
    str = sprintf('$\\varepsilon$  = %1.2f', error_CL);
    annotation('textbox',[0.15 0.23 0.2 0.025],'String',...
    str,'FitBoxToText','on', 'FontSize',30,'Interpreter','latex');
    set(gca,'FontSize',30);
    set(gcf,'Position',[100 100 1200 750]);
%     saveas(gcf,"Superposed POD/Train-123/"+datasets(case_index)+"_CL_rec_"+p+"_clean.png");
%     saveas(gcf,"Lift Reconstruction/"+datasets(case_index)+"_CL_rec_"+p+"_clean.png");
%     writematrix(pivot_train,"POD_Sensors/Train-123/Sensors_"+p+".txt");
    writematrix(pivot_train,"POD_Sensors/Indiv_Case/"+case_name(case_index)+"_POD_"+p+".txt");
end

