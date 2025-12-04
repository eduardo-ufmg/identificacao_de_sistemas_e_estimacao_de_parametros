classdef obstaclesImg < handle

    properties
        map;
        xlimits, ylimits;
        ncol, nrow;
        obstacles;
        points;
        order;
    end

    methods

        function this = obstaclesImg(image, xlimits, ylimits)

            this.xlimits = xlimits;
            this.ylimits = ylimits;

            I = double(imread(image));

            if size(I,3) > 1
                I = rgb2gray(I);
            end
            I = I - min(I(:));
            I = I/max(I(:));

            I([1 end],:) = 1;
            I(:,[1 end]) = 1;

            [this.nrow, this.ncol] = size(I);

            I(I >= 0.5) = 1;
            I(I < 0.5) = 0;

            this.map = flipud(I);
        end

        function c = colision(this, p, robotSize)

            c = false;

            [col, lin] = this.mts2px(p(1), p(2));

            dx = this.xlimits(2) - this.xlimits(1);
            dy = this.ylimits(2) - this.ylimits(1);
            robCol = floor(robotSize*(this.ncol/dx));
            robRow = floor(robotSize*(this.nrow/dy));

            % Check bounds before accessing map
            if (lin-robRow) < 1 || (lin+robRow) > this.nrow || (col-robCol) < 1 || (col+robCol) > this.ncol
                c = true;
                return;
            end

            try

                robot = this.map(lin-robRow:lin+robRow, col-robCol:col+robCol);

                robot = robot < 1;
            catch

                c = true;
                return;
            end

            if any(robot(:))
                c = true;
                return;
            end
        end

        function [xm, ym] = px2mts(this, xp, yp)

            dx = this.xlimits(2) - this.xlimits(1);
            dy = this.ylimits(2) - this.ylimits(1);

            xm = xp*(dx/this.ncol) + this.xlimits(1);
            ym = yp*(dy/this.nrow) + this.ylimits(1);
        end

        function [xp, yp] = mts2px(this, xm, ym)

            dx = this.xlimits(2) - this.xlimits(1);
            dy = this.ylimits(2) - this.ylimits(1);

            xp = (xm - this.xlimits(1))*(this.ncol/dx);
            xp = round(xp);
            yp = (ym - this.ylimits(1))*(this.nrow/dy);
            yp = round(yp);
        end

        function draw(this)
            hold on;

            colormap gray;
            grid on;
            imagesc(this.xlimits, this.ylimits, this.map);

            xlabel('x [m]');
            ylabel('y [m]');
            axis equal;
            xlim(this.xlimits);
            ylim(this.ylimits);

            hold off;
        end

        function obstaclePoints(this,decimation,max_dist)

            map_aux = this.map;
            siz_map = size(this.map);
            visited = zeros(size(map_aux));
            wave_front = zeros(size(map_aux));
            boundary = zeros(size(map_aux));

            numObstacles = 0; handle = figure;

            while (nnz((visited(this.map == 0) == 0)) > 0)
                numObstacles = numObstacles + 1;

                [i,j] = find(map_aux == 0,1);
                map_aux(i,j) = -numObstacles;
                wave_front(i,j) = 1; front_counter = 1; movesleft = true;
                while movesleft
                    movesleft = false;

                    [i,j] = find(wave_front == front_counter);

                    for ell = 1:numel(i)

                        neighbors_i = [-1, -1, -1, 0, 0, 0, 1, 1, 1];
                        neighbors_j = [-1, 0, 1, -1, 0, 1, -1, 0, 1];
                        for k = 1:9

                            if ((i(ell) + neighbors_i(k))>0)&&((j(ell) + neighbors_j(k)) > 0)&&((i(ell) + neighbors_i(k))<siz_map(1))&&((j(ell) + neighbors_j(k))<siz_map(2))
                                if this.map(i(ell) + neighbors_i(k),j(ell) + neighbors_j(k)) == 0
                                    if visited(i(ell) + neighbors_i(k),j(ell) + neighbors_j(k)) == 0
                                        if (wave_front(i(ell) + neighbors_i(k),j(ell) + neighbors_j(k)) == 0)||(wave_front(i(ell) + neighbors_i(k),j(ell) + neighbors_j(k)) > front_counter)
                                            wave_front(i(ell) + neighbors_i(k),j(ell) + neighbors_j(k)) = front_counter + 1;
                                        end
                                        map_aux(i(ell) + neighbors_i(k),j(ell) + neighbors_j(k)) = map_aux(i(ell),j(ell));
                                        movesleft = true;
                                    end
                                else
                                    boundary(i(ell),j(ell)) = 1;
                                end
                            end
                        end
                        visited(i(ell),j(ell)) = 1;
                    end
                    front_counter = front_counter + 1;
                end

                wave_front(wave_front > 0) = nan;
                figure(handle);
                subplot(2,2,1), imagesc(flipud(map_aux));
                subplot(2,2,2), imagesc(flipud(visited));
                subplot(2,2,3), imagesc(flipud(wave_front));
                subplot(2,2,4), imagesc(flipud(boundary));
            end

            [xp,yp] = find(boundary == 1);
            [ym, xm] = px2mts(this, xp, yp);
            this.obstacles = [xm(1:decimation:end), ym(1:decimation:end)];

            this.points = cell(numObstacles,1);
            this.order = cell(numObstacles,1);
            for i = 1:numObstacles
                [xp,yp] = find((map_aux == -i)&(boundary == 1));
                [ym, xm] = px2mts(this, xp, yp);
                this.points{i} = [xm(1:decimation:end),ym(1:decimation:end)];
                this.order{i} = zeros(size(this.points{i},1),2);

                dist = inf*ones(size(this.points{i},1),size(this.points{i},1));
                for k = 1:size(this.points{i},1)
                    for j = 1:size(this.points{i},1)
                        if (j ~= k)
                            dist(k,j) = norm(this.points{i}(k,:) - this.points{i}(j,:),2);
                        end
                    end
                end
                for k = 1:size(this.points{i},1)
                    [sd, ind] = min(dist(k,:));
                    if (sd(1) < max_dist)
                        this.order{i}(k,:) = [k ind(1)];
                        dist(k, ind(1)) = inf;
                        dist(ind(1), k) = inf;
                    end
                end
                this.order{i}(this.order{i}(:,1) == 0, :) = [];
            end
        end
    end
end
