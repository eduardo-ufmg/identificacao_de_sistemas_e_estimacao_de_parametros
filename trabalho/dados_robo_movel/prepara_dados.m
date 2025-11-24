clearvars; close all; clc;

load dados.mat; %a taxa de amostragem é de 0.1s

%pelo arquivo de configuração, sabemos que a posição inicial do robô é
x0 = 38.07; y0 = -19.52; theta0 = 0;

%sabemos também que o mapa que temos é do tamanho 810x1654
%e que a resolução é de 0.0962
res = 0.0962;
ml = mapaLaser('icex_2.pgm',res);

%as 3 primeiras colunas são a odometria (incrementos/deslocamentos)
x_odo = dadosMod(:,1);
y_odo = dadosMod(:,2);
theta_odo = dadosMod(:,3);
%transforma a odometria em "velocidades"
% os dados de odometria já são incrementos, não posições acumuladas
delta_odo = [x_odo, y_odo, theta_odo];
w = delta_odo(:,3)/0.1; %velocidadade angular é só pegar e dividir a variação do theta
%a velocidade tangencial, acredito que o robô não utilizou velocidades negativas em nenhum momento, mas vamos testar
v = zeros(size(w,1),1);
for i = 1:numel(v)
    v(i) = sqrt(delta_odo(i,1)*delta_odo(i,1) + delta_odo(i,2)*delta_odo(i,2))/0.1;
end


%rode o comando abaixo para conferir - reconstrói a trajetória a partir das velocidades
%%confere se reconstrói a odometria acumulada
%xt = zeros(size(v,1),1);
%yt = zeros(size(v,1),1);
%thetat = zeros(size(v,1),1);
%
%xt(1) = x0 + 0.1*v(1)*cos(theta0);
%yt(1) = y0 + 0.1*v(1)*sin(theta0);
%thetat(1) = theta0 + 0.1*w(1);
%
%for i = 2:size(v,1)
%    xt(i) = xt(i-1) + 0.1*v(i)*cos(thetat(i-1));
%    yt(i) = yt(i-1) + 0.1*v(i)*sin(thetat(i-1));
%    thetat(i) = thetat(i-1) + 0.1*w(i);
%end
%
% reconstrói odometria acumulada para comparar
%x_odo_acum = x0 + cumsum(x_odo);
%y_odo_acum = y0 + cumsum(y_odo);
%%figure(1); ml.draw; hold on; plot(x_odo_acum,y_odo_acum,'b',xt,yt,'r'); hold off;

%as 3 colunas seguintes são a localização estimada via AMCL (deve ser usada
%como "ground truth" para comparação)
x_amcl = dadosMod(:,4);
y_amcl = dadosMod(:,5);
theta_amcl = dadosMod(:,6);

%as 361 colunas seguintes representam a leitura do laser (SICK LMS200)
%as leituras do laser variam de 0.5 em 0.5 grau - resolução = pi/360
%o laser lê 361 pontos, de -pi/2 a pi/2 relativo à orientação do robô
laser = dadosMod(:,7:end); dlaser = pi/360;