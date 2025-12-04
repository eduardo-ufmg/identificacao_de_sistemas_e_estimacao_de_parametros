clearvars; close all; clc;

load data.mat;

x0 = 38.07; y0 = -19.52; theta0 = 0;

res = 0.0962;
ml = laserMap('map.pgm',res);

x_odo = data(:,1);
y_odo = data(:,2);
theta_odo = data(:,3);

delta_odo = [[x_odo(1); diff(x_odo)], [y_odo(1); diff(y_odo)], [theta_odo(1); diff(theta_odo)]];
w = delta_odo(:,3)/0.1;

v = zeros(size(w,1),1);
for i = 1:numel(v)
    v(i) = sqrt(delta_odo(i,1)*delta_odo(i,1) + delta_odo(i,2)*delta_odo(i,2))/0.1;
end

x_amcl = data(:,4);
y_amcl = data(:,5);
theta_amcl = data(:,6);

laser = data(:,7:end); dlaser = pi/360;