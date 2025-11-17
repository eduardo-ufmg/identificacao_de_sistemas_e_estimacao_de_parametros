classdef mapaLaser < handle
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    properties %(SetAccess = private, GetAccess = private)
        map; % mapa da imagem para exibicao (com o u invertido)
        xlimits, ylimits; % dimesoes em metros
        ncol, nrow; % numero de colunas e linhas da imagem
        resolution; %resolução do mapa
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % construtor
        function this = mapaLaser(image, resolution)
            % le a imagem
            I = double(imread(image));
            % se a imagem eh colorida, torna-a gray
            if size(I,3) > 1
                I = rgb2gray(I);
            end
            I = I - min(I(:));
            I = I/max(I(:));
            
            % elimina bordas para ficar melhor a apresentacao
            I([1 end],:) = 1;
            I(:,[1 end]) = 1;
            
            % linhas e colunas da imagem
            [this.nrow, this.ncol] = size(I);
            
            % binariza imagem
            %I = imbinarize(I, 0.5);
            I(I >= 0.5) = 1;
            I(I < 0.5) = 0;
            
            % inverte a imagem em y
            this.map = flipud(I);
            
            % salva o tamanho geometrico da imagem em metros
            this.xlimits = resolution*0.5*[-this.ncol, this.ncol];
            this.ylimits = resolution*0.5*[-this.nrow, this.nrow];
            this.resolution = resolution;
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % transforma pixels na imagem para pontos no mundo real
        function [xm, ym] = px2mts(this, xp, yp)
            
            dx = this.xlimits(2) - this.xlimits(1);
            dy = this.ylimits(2) - this.ylimits(1);
            
            xm = xp*(dx/this.ncol) + this.xlimits(1);
            ym = yp*(dy/this.nrow) + this.ylimits(1);
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % transforma pontos no mundo real para pixels na imagem
        function [xp, yp] = mts2px(this, xm, ym)
            
            dx = this.xlimits(2) - this.xlimits(1);
            dy = this.ylimits(2) - this.ylimits(1);
            
            xp = (xm - this.xlimits(1))*(this.ncol/dx);
            xp = round(xp);
            yp = (ym - this.ylimits(1))*(this.nrow/dy);
            yp = round(yp);
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % desenha a imagem distorcida em metros
        function draw(this)
            hold on;
            
            % versao antiga
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
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % verifica ponto no mapa
        function c = checkPoint(this, p)
            % posicao de colisao na imagem
            [col, lin] = this.mts2px(p(1), p(2));
            c = 1 - this.map(lin,col);
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % simula laser para uma dada posição no mapa
        function laser = simLaser(this, p, r, r_res, radMin, radMax, npt)
            %ângulos a se testar
            ang = linspace(radMin,radMax,npt) + p(3); %leva em consideração orientação do robô
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
