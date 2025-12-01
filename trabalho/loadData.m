function sensorData = loadData(datFile)
%LOADDATA Load dataset from data.dat, skipping header lines.
% Returns matrix with rows: [x_odo, y_odo, theta_odo, x_est, y_est, theta_est, 361 laser]
%
% Usage:
%   sensorData = loadData('data.dat');
%
% Notes:
% - Skips first 5 lines (header and initial prints)
% - Handles variable whitespace and line lengths

fid = fopen(datFile,'r');
assert(fid>0, 'Could not open %s', datFile);
cleanup = onCleanup(@() fclose(fid));

% Skip first 5 lines
for i=1:5
    fgetl(fid);
end

rows = {};
while ~feof(fid)
    ln = fgetl(fid);
    if ~ischar(ln)
        break;
    end
    ln = strtrim(ln);
    if isempty(ln)
        continue;
    end
    % Split by whitespace
    parts = regexp(ln,'\s+','split');
    nums = str2double(parts);
    if any(isnan(nums))
        % Ignore non-numeric lines
        continue;
    end
    % Expect 367 columns (3 odo + 3 estimate + 361 laser)
    if numel(nums) < 6
        continue;
    end
    % Some lines may include only first 6 columns; pad missing lasers
    if numel(nums) < 367
        nums = [nums, nan(1, 367-numel(nums))];
    end
    rows{end+1,1} = nums; %#ok<AGROW>
end

% Stack into matrix
sensorData = cell2mat(rows);
end
