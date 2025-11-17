classdef obstaclesImg < handle
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    properties %(SetAccess = private, GetAccess = private)
        mapa; % mapa da imagem para exibicao (com o u invertido)
        xlimits, ylimits; % dimesoes em metros
        ncol, nrow; % numero de colunas e linhas da imagem
        obstacles; %pontos contendo as bordas dos obstáculos
        points; %pontos das bordas dos obstáculos (separados por obstáculos)
        order; %ordem dos pontos na borda (para a restrição da triangulação)
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % construtor
        function this = obstaclesImg(image, xlimits, ylimits)
            
            % salva o tamanho geometrico da imagem em metros
            this.xlimits = xlimits;
            this.ylimits = ylimits;
            
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
            this.mapa = flipud(I);
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % verifica colisao com os obstaculos
        function c = colision(this, p, robotSize)
            
            % status comeca nulo
            c = false;
            
            % posicao de colisao na imagem
            [col, lin] = this.mts2px(p(1), p(2));
            
            % dimensoes do robo em pixels
            dx = this.xlimits(2) - this.xlimits(1);
            dy = this.ylimits(2) - this.ylimits(1);
            robCol = floor(robotSize*(this.ncol/dx));
            robRow = floor(robotSize*(this.nrow/dy));
            
            % captura parte do mapa correspondente ao robo
            try
                % area do robo no mapa
                robot = this.mapa(lin-robRow:lin+robRow, col-robCol:col+robCol);
                % ponto de obstaculo
                robot = robot < 1;
            catch
                % colisao com as bordas
                c = true;
                return;
            end
            
            % se algum ponto tocou obstaculos
            if any(robot(:))
                c = true;
                return;
            end
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
            imagesc(this.xlimits, this.ylimits, this.mapa);
            
            xlabel('x [m]');
            ylabel('y [m]');
            axis equal;
            xlim(this.xlimits);
            ylim(this.ylimits);
            
            hold off;
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %obstaclePoints
        % Essa função tem por objetivo encontrar os obstáculos existentes no mapa e
        % retornar uma nuvem de pontos que represente a fronteira destes obstáculos
        
        function obstaclePoints(this,decimacao,max_dist)
            %vamos fazer uma cópia do mapa para trabalharmos em cima dele
            mapa_aux = this.mapa;
            siz_mapa = size(this.mapa);
            visitados = zeros(size(mapa_aux));
            frente_onda = zeros(size(mapa_aux));
            fronteira = zeros(size(mapa_aux));
            
            %vamos fazer algo tipo um brushfire para resolver - partindo de dentro
            %de cada obstáculo
            numObstacles = 0; handle = figure;
            
            while (nnz((visitados(this.mapa == 0) == 0)) > 0) %enquanto ainda tivermos obstáculos não visitados
                numObstacles = numObstacles + 1;
                %começamos de um obstáculo qq
                [i,j] = find(mapa_aux == 0,1);
                mapa_aux(i,j) = -numObstacles;
                frente_onda(i,j) = 1; frente = 1; movesleft = true;
                while movesleft
                    movesleft = false;
                    %pega todos os elementos da frente de onda atual
                    [i,j] = find(frente_onda == frente);
                    %percorre estes elementos
                    for ell = 1:numel(i)
                        %confere os vizinhos do elemento visitado
                        vizinhos_i = [-1, -1, -1, 0, 0, 0, 1, 1, 1];
                        vizinhos_j = [-1, 0, 1, -1, 0, 1, -1, 0, 1];
                        for k = 1:9 %sempre são 9 vizinhos
                            %vizinho (vizinhos_i(k),vizinhos_j(k))
                            if ((i(ell) + vizinhos_i(k))>0)&&((j(ell) + vizinhos_j(k)) > 0)&&((i(ell) + vizinhos_i(k))<siz_mapa(1))&&((j(ell) + vizinhos_j(k))<siz_mapa(2)) %se o vizinho existe
                                if this.mapa(i(ell) + vizinhos_i(k),j(ell) + vizinhos_j(k)) == 0 %se ele é um obstáculo
                                    if visitados(i(ell) + vizinhos_i(k),j(ell) + vizinhos_j(k)) == 0
                                        if (frente_onda(i(ell) + vizinhos_i(k),j(ell) + vizinhos_j(k)) == 0)||(frente_onda(i(ell) + vizinhos_i(k),j(ell) + vizinhos_j(k)) > frente)
                                            frente_onda(i(ell) + vizinhos_i(k),j(ell) + vizinhos_j(k)) = frente + 1;
                                        end
                                        mapa_aux(i(ell) + vizinhos_i(k),j(ell) + vizinhos_j(k)) = mapa_aux(i(ell),j(ell));
                                        movesleft = true;
                                    end
                                else %se o vizinho é um espaço vazio
                                    fronteira(i(ell),j(ell)) = 1;
                                end
                            end
                        end
                        visitados(i(ell),j(ell)) = 1;
                    end
                    frente = frente + 1;
                end
                %para que não fique pegando um obstáculo que já passou, atribui nan
                %para a frentes de onda com valor positivo
                frente_onda(frente_onda > 0) = nan;
                figure(handle);
                subplot(2,2,1), imagesc(flipud(mapa_aux));
                subplot(2,2,2), imagesc(flipud(visitados));
                subplot(2,2,3), imagesc(flipud(frente_onda));
                subplot(2,2,4), imagesc(flipud(fronteira));
            end
                        
            %pega os pontos de fronteira e decima eles (para não manter
            %pontos demais)
            [xp,yp] = find(fronteira == 1);
            [ym, xm] = px2mts(this, xp, yp);
            this.obstacles = [xm(1:decimacao:end), ym(1:decimacao:end)];
            
            %Vamos tentar pegar uma ordem nos pontos da fronteira
            this.points = cell(numObstacles,1);
            this.order = cell(numObstacles,1);
            for i = 1:numObstacles
                [xp,yp] = find((mapa_aux == -i)&(fronteira == 1));
                [ym, xm] = px2mts(this, xp, yp);
                this.points{i} = [xm(1:decimacao:end),ym(1:decimacao:end)];
                this.order{i} = zeros(size(this.points{i},1),2);
                %para cada ponto, tentamos encontrar os pontos mais
                %próximos para descrever as restrições da malha
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
                this.order{i}(this.order{i}(:,1) == 0, :) = []; %remove as colunas que não tem vizinho
            end
        end
    end
end
