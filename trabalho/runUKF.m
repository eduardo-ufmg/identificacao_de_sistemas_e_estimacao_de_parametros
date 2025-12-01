clearvars; close all; clc;

% Config
datFile = 'data.dat';
mapFile = 'icex_2.pgm';
res = 0.0962;           % map resolution (m/pixel)
dt = 0.1;               % sampling period (s)
robotSize = 0.4;        % for collision checks (m)
beamDecim = 5;          % use every 5th beam to reduce cost
maxRange = 8.0;         % LIDAR max range (m)

% Load data
sensorData = loadData(datFile);

% Extract signals
x_odo = sensorData(:,1);
y_odo = sensorData(:,2);
theta_odo = sensorData(:,3);
x_ref = sensorData(:,4);
y_ref = sensorData(:,5);
theta_ref = sensorData(:,6);
laserAll = sensorData(:,7:end);

% Compute velocities from odometry deltas
N = size(sensorData,1);
dx = [x_odo(1); diff(x_odo)];
dy = [y_odo(1); diff(y_odo)];
dth = [theta_odo(1); diff(theta_odo)];
v = sqrt(dx.^2 + dy.^2)/dt;
w = dth/dt;

% Map and laser helpers
ml = laserMap(mapFile, res);
dlaser = pi/360;           % 0.5 deg resolution
beamIdx = 1:beamDecim:361; % decimated beams

% Initial pose
x0 = 38.07; y0 = -19.52; theta0 = 0;

% UKF setup (state: [x;y;theta])
stateFcn = @(x,u) [
    x(1) + u(1)*dt*cos(x(3) + 0.5*u(2)*dt);
    x(2) + u(1)*dt*sin(x(3) + 0.5*u(2)*dt);
    wrapToPi(x(3) + u(2)*dt)
];

measFcn = @(x) simulateBeams(x, ml, maxRange, res, beamIdx);

ukf = unscentedKalmanFilter(stateFcn, measFcn, [x0; y0; theta0]);
ukf.Alpha = 1e-2;   % spread of sigma points
ukf.Beta = 2;       % optimal for Gaussian
ukf.Kappa = 0;      % secondary scaling

% Noise tuning
% Process noise: uncertainty on v and w mapped to state; approximate
q_pos = 0.05;   % m-level drift per step
q_ang = deg2rad(1.0);
ukf.ProcessNoise = diag([q_pos^2, q_pos^2, q_ang^2]);

% Measurement noise: SICK LMS typical ~0.01-0.02 m; use conservative 0.05 m
r_laser = 0.08; % m
ukf.MeasurementNoise = (r_laser^2)*eye(numel(beamIdx));

% Run filter
Xest = zeros(N,3);
Xest(1,:) = [x0, y0, theta0];

for k = 2:N
    u = [v(k); w(k)];
    % Predict
    predict(ukf, u);
    % Measurement from dataset (decimated beams)
    z = laserAll(k, beamIdx);
    % Some rows may have NaNs (padded); replace with max range
    z(~isfinite(z)) = maxRange;
    % Correct
    correct(ukf, z);
    % Save
    Xest(k,:) = ukf.State';
end

% Plot results
figure(1); ml.draw; hold on;
plot(x_ref, y_ref, 'r', 'LineWidth',1.5);
plot(Xest(:,1), Xest(:,2), 'b', 'LineWidth',1.2);
plot(x0,y0,'ko','MarkerFaceColor','y');
legend('Map','Reference','UKF','Start'); title('Trajectory Comparison'); hold off;

% Helper: simulate decimated beams
function z = simulateBeams(x, ml, maxRange, r_res, beamIdx)
    theta0 = x(3);
    radMin = -pi/2; radMax = pi/2;
    npt = 361;
    beams = ml.simLaser([x(1); x(2); theta0], maxRange, r_res, radMin, radMax, npt);
    z = beams(beamIdx);
end
