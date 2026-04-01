clear;clc;close all;
set(0,'DefaultAxesFontName','Times New Roman');
set(0,'DefaultTextFontName','Times New Roman');
set(0,'DefaultAxesFontSize',16);
set(0,'DefaultTextFontSize',16);
set(0,'defaulttextinterpreter','latex');
set(0,'defaultAxesTickLabelInterpreter','latex');
set(0,'defaultLegendInterpreter','latex');

% cd 'C:\Users\10233\Box\Research\Data\Channel_data'  % Laptop
cd 'C:\Users\Mickey\Box\Research\Data\Channel_data' % Desktop
load("U_large.mat")
% load("tau_w.mat")
load("X.mat")
load("Y.mat")
% load("Z.mat")

Retau = 186.0;
utau = 0.057231058999996;
mu = 3.076923076923077e-04;
itime_list = 10000:100:99900; % 900 time steps to extract
ix_list = 100:50:300; % X indices of wall shear stress probes
iz_list = 100:100:300; % Z indices of wall shear stress probes
ix_samples = -20:20; % X indices of velocity probes relative to the stress sensor index
iy_samples = 2:81; % Y indices of velocity probes
iz_samples = 1;  % Z indices of velocity probes

% sizes
Nx = length(ix_samples);
Ny = length(iy_samples);
Nzi = length(iz_list);
Nxi = length(ix_list);
Nt = length(itime_list);
[Xgrid, Ygrid] = meshgrid((x(ix_samples+100))*Retau, ym(iy_samples)*Retau);
% Lx = x(ix_samples(end))-x(ix_samples(1));
% Ly = y(iy_samples(end)) - y(iy_samples(1));


%%
% u = squeeze(U_probe(:,1,1,:,:)); % Shape: (nt, nx, ny)
u = U_probe; % Shape: (nt, nxi, nzi, nx, ny)

% Y is non-uniform and X is uniform
x_uniform = Xgrid(1, :);  % Non-uniform x points
y_non_uniform = Ygrid(:, 1);      % Uniform y points

% Create new uniform x-axis
y_uniform = linspace(min(y_non_uniform), max(y_non_uniform), 200);

% Interpolate along x-axis (y remains uniform for interpolation)
[Xq, Yq] = meshgrid(x_uniform, y_uniform);
dx = Xq(1, 2) - Xq(1, 1);  % Grid spacing in x-direction
dy = Yq(2, 1) - Yq(1, 1);  % Grid spacing in y-direction

u_uniform = zeros(Nt,Nxi,Nzi,numel(x_uniform), numel(y_uniform));
% interpolate to uniform grid
for t = 1:Nt
    for xi = 1:Nxi
        for zi = 1:Nzi
            u_uniform(t, xi, zi, :, :) = interp2(x_uniform, y_non_uniform, squeeze(u(t, xi, zi, :, :))', Xq, Yq, 'spline')';
        end
    end
end

% Compute time-averaged mean velocity
u_mean = squeeze(mean(u_uniform, [1,2,3]));  % Shape: (nx, ny)

% Subtract mean to get velocity fluctuations
u_prime = u_uniform - permute(u_mean, [3, 4,5,1, 2]);  % Shape: (nt, nxi, nzi, nx, ny)

%% Computing X-Corr -----------------
% Initialize cross-correlation storage for averaging over z
maxlag = 0;
nlags = 2*maxlag + 1;
[Nt, Nxi, Nzi, Nx, Ny] = size(u_prime);

CrossCorr = zeros(Nxi,Nzi,Nx, Ny);
anchor_xi = 21;
% anchor_yi = 8;  % y+ = 10
anchor_yi = 30; % y+ = 40


for xi = 1:Nxi
    for zi = 1:Nzi
        % Get anchor point u
        u_anchor = reshape(u_prime(:,xi,zi,anchor_xi,anchor_yi),1,[]);
        for i = 1:Nx
            for j = 1:Ny
                % Extract velocity time series at all locations
                u_t = reshape(u_prime(:,xi,zi,i,j),1,[]);
        
                % Compute autocorrelation
                R_u = dot(u_t, u_anchor);
                R_u = R_u / (rms(u_t,"all")*rms(u_anchor,"all"));
                CrossCorr(xi,zi,i, j) = CrossCorr(xi,zi,i, j) + R_u;
            end
        end
    end
end

% % Average over z and x
CrossCorr_target = squeeze(mean(CrossCorr/Nt,[1,2]));

%% Plot X-Corr contour
close all;

% Generate spatial grid
[Xgrid, Ygrid] = meshgrid(x_uniform, y_uniform);
% Turn absolute into relative coordiantes 
Xgrid = Xgrid-x_uniform(anchor_xi);
% Ygrid = Ygrid-y_uniform(anchor_yi);

% Plot cross-correlation function
figure('Units', 'inches', 'Position', [1 1 8 3.2], 'color', [1 1 1]), box on, hold on
set(gca, 'LineWidth', 2, 'fontsize', 17)
contourf(Xgrid, Ygrid, CrossCorr_target', 1000, 'LineStyle', 'none', 'LineColor', 'none');% Contour plot
shading interp
axis tight;
axis equal;

colorbar()
clim([0,1])
yticks(0:20:80)
ylim([0,80])
xlim([-100,100])
xlabel('$\Delta x^+$', 'Interpreter', 'latex')
ylabel('$y^+$', 'Interpreter', 'latex')

saveas(gcf,'WSS_autocorr_40','epsc')
% title('$R_{uu}$')
saveas(gcf,'WSS_autocorr_40.png')
hold off

% Plot cross-correlation function
figure('position', [50 50 800 500], 'color', [1 1 1]), box on, hold on
set(gca, 'LineWidth', 2, 'fontsize', 20)
contour_levels = linspace(min(CrossCorr_target(:)), max(CrossCorr_target(:)), 25); 
contour(Xgrid, Ygrid, CrossCorr_target', contour_levels, 'LineWidth', 1.5); 
colorbar()
clim([0,1])
% ylim([0,50])
xlabel('$\Delta x^+$', 'Interpreter', 'latex')
ylabel('$y^+$', 'Interpreter', 'latex')
% saveas(gcf,'WSS_X_corr','epsc')
% saveas(gcf,'WSS_X_corr.png')
title('$R_{uu}$')
hold off

%% Extract Characteristic Length Scales
% Extract central row and column of autocorrelation
% center_row = ceil(size(mean_autocorr_normalized, 1)/2);
% center_col = ceil(size(mean_autocorr_normalized, 2)/2);
autocorr_y = CrossCorr_target(anchor_xi, :);  % y slice
autocorr_x = CrossCorr_target(:, anchor_yi);  % x slice

% Compute thresholds (e.g., 1/e or 0.5)
% threshold = 1/exp(1);
threshold = 0.95;

% Find x-direction length scale
idx_x = find(autocorr_x > threshold, 1, 'last');
L_x = abs(x_uniform(idx_x)-x_uniform(anchor_xi));  % Physical length scale in x

% Find y-direction length scale
idx_y = find(autocorr_y > threshold, 1, 'last');
L_y = abs(y_uniform(idx_y)-y_uniform(anchor_yi));  % Physical length scale in y

fprintf('Characteristic length scales at y+ = %.2f :\n L_x^+ = %.2f \n L_y^+ = %.2f \n', y_uniform(anchor_yi), L_x, L_y);


