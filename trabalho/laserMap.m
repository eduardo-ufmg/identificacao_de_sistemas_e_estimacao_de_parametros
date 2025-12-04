classdef laserMap < handle

    properties
        map;
        xlimits, ylimits;
        ncol, nrow;
        resolution;
    end

    methods

        function this = laserMap(image, resolution)

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

            this.xlimits = resolution*0.5*[-this.ncol, this.ncol];
            this.ylimits = resolution*0.5*[-this.nrow, this.nrow];
            this.resolution = resolution;
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

        function c = checkPoint(this, p)

            [col, lin] = this.mts2px(p(1), p(2));
            c = 1 - this.map(lin,col);
        end

        function laser = simLaser(this, p, r, r_res, radMin, radMax, npt)

            ang = linspace(radMin,radMax,npt) + p(3);
            laser = r*ones(1,npt);
            for i = 1:npt
                raio_teste = 0:r_res:r;
                for rd = raio_teste
                    xt = p(1) + rd*cos(ang(i));
                    yt = p(2) + rd*sin(ang(i));
                    [xp, yp] = this.mts2px(xt, yt);
                    if this.map(yp,xp) < 1
                        laser(i) = rd;
                        break;
                    end
                end
            end
        end
    end
end
