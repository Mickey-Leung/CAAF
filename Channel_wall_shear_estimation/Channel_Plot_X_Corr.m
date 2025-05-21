clear;clc;close all;
set(0,'DefaultAxesFontName','Times New Roman');
set(0,'DefaultTextFontName','Times New Roman');
set(0,'DefaultAxesFontSize',16);
set(0,'DefaultTextFontSize',16);
set(0,'defaulttextinterpreter','latex');
set(0,'defaultAxesTickLabelInterpreter','latex');
set(0,'defaultLegendInterpreter','latex');

load("CrossCorr_target.mat")
load("Xgrid.mat")
load("Ygrid.mat")

% Plot cross-correlation function
figure('Units', 'inches', 'Position', [1 1 8 3.5], 'color', [1 1 1]), box on, hold on
set(gca, 'LineWidth', 2, 'fontsize', 17)
contourf(Xgrid, Ygrid, CrossCorr_target', 1000, 'LineStyle', 'none', 'LineColor', 'none');% Contour plot
shading interp
axis tight;
axis equal;
colorbar()
clim([0,1])
ylim([0,50])
xticks(-50:25:50)
xlabel('$\Delta x^+$', 'Interpreter', 'latex')
ylabel('$y^+$', 'Interpreter', 'latex')
saveas(gcf,'WSS_X_corr','epsc')
saveas(gcf,'WSS_X_corr.png')


