clearvars; close all; clc;

load dados.mat; %a taxa de amostragem é de 0.1s

%pelo arquivo de configuração, sabemos que a posição inicial do robô é
x0 = 38.07; y0 = -19.52; theta0 = 0;

%sabemos também que o mapa que temos é do tamanho 810x1654
%e que a resolução é de 0.0962
ytam = 0.0962*810; xtam = 0.0962*1654;
mapa = obstaclesImg('icex_2.pgm',[-xtam/2, xtam/2],[-ytam/2,ytam/2]);
figure(1); mapa.draw; hold on; plot(x0,y0,'r*'); hold off;

%as 3 primeiras colunas são a odometria (incrementos)
x_odo = dadosMod(:,1);
y_odo = dadosMod(:,2);
theta_odo = dadosMod(:,3);

%acumula a odometria para obter trajetória
x_odo_acum = x0 + cumsum(x_odo);
y_odo_acum = y0 + cumsum(y_odo);

%as 3 colunas seguintes são a localização estimada via AMCL (deve ser usada
%como "ground truth" para comparação)
x_amcl = dadosMod(:,4);
y_amcl = dadosMod(:,5);
theta_amcl = dadosMod(:,6);
figure(2); mapa.draw; hold on; plot(x_odo_acum,y_odo_acum,'b',x_amcl,y_amcl,'r'); hold off;

%as 361 colunas seguintes representam a leitura do laser (SICK LMS200)
%as leituras do laser variam de 0.5 em 0.5 grau - resolução = pi/360
%o laser lê 361 pontos, de -pi/2 a pi/2 relativo à orientação do robô
laser = dadosMod(:,7:end); dlaser = pi/360;
figure(3); mapa.draw; hold on;
for i = 1:size(laser,2)
    thetal = theta0 - pi/2;
    xl = x0 + laser(1,i)*cos(thetal + (i-1)*dlaser);
    yl = y0 + laser(1,i)*sin(thetal + (i-1)*dlaser);
    plot(xl,yl,'.r');
end

% %plota a leitura do laser sobre a trajetória, considerando a localização
% %"corrigida"
% for k = 1:50:size(laser,1)
%     figure(4); mapa.draw; hold on; plot(x_amcl(k),y_amcl(k),'b.');
%     for i = 1:size(laser,2)
%         thetal = theta_amcl(k) - pi/2;
%         xl = x_amcl(k) + laser(k,i)*cos(thetal + (i-1)*dlaser);
%         yl = y_amcl(k) + laser(k,i)*sin(thetal + (i-1)*dlaser);
%         plot(xl,yl,'.r');
%     end
%     hold off;
%     drawnow; 
%     pause(0.1);
% end

%beleza... Como o gráfico plotado fez sentdo... Então acredito que os
%parâmetros são estes mesmos (tanto para o mapa, quanto para a leitura do
%laser)

%define o objeto que pode ser usado para simular o laser
ml = mapaLaser('icex_2.pgm',0.0962);
%o código abaixo é como ele pode ser chamado, para os parâmetros do nosso
%problema
l0 = ml.simLaser([x0; y0; theta0],max(max(laser)),0.0962,-pi/2,pi/2,361);
%plota junto com os dados reais só para ter uma noção
figure(3); hold on;
for i = 1:size(l0,2)
    thetal = theta0 - pi/2;
    xl = x0 + l0(i)*cos(thetal + (i-1)*dlaser);
    yl = y0 + l0(i)*sin(thetal + (i-1)*dlaser);
    plot(xl,yl,'.b');
end