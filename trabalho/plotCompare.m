clearvars; close all; clc;

load data.mat;

x0 = 38.07; y0 = -19.52; theta0 = 0;

ytam = 0.0962*810; xtam = 0.0962*1654;
map = obstaclesImg('icex_2.pgm',[-xtam/2, xtam/2],[-ytam/2,ytam/2]);
figure(1); map.draw; hold on; plot(x0,y0,'r*'); hold off;

x_odo = sensorData(:,1);
y_odo = sensorData(:,2);
theta_odo = sensorData(:,3);

x_amcl = sensorData(:,4);
y_amcl = sensorData(:,5);
theta_amcl = sensorData(:,6);
figure(2); map.draw; hold on; plot(x0+x_odo,y0+y_odo,'b',x_amcl,y_amcl,'r'); hold off;

laser = sensorData(:,7:end); dlaser = pi/360;
figure(3); map.draw; hold on;
for i = 1:size(laser,2)
    thetal = theta0 - pi/2;
    xl = x0 + laser(1,i)*cos(thetal + (i-1)*dlaser);
    yl = y0 + laser(1,i)*sin(thetal + (i-1)*dlaser);
    plot(xl,yl,'.r');
end

ml = mapLaser('icex_2.pgm',0.0962);

l0 = ml.simLaser([x0; y0; theta0],max(max(laser)),0.0962,-pi/2,pi/2,361);

figure(3); hold on;
for i = 1:size(l0,2)
    thetal = theta0 - pi/2;
    xl = x0 + l0(i)*cos(thetal + (i-1)*dlaser);
    yl = y0 + l0(i)*sin(thetal + (i-1)*dlaser);
    plot(xl,yl,'.b');
end