%% 标准Griewank函数结果
x = rand(1,1000);
x=x*10-5;
y = rand(1,1000);
y=y*10-5;
z=zeros(1,1000);
[X,Y]=meshgrid(x,y);
[row,col]=size(X);

for l=1:col
   for h=1:row
      Z(h,l)=Griewank([X(h,l),Y(h,l)]);
   end
end

for i=1:1000
    z(i)=Z(i,i);
end
scatter3(x,y,z,'.');
hold on
    xlay=xlabel('Dimension x');
    ylay=ylabel('Dimension y');
    zlay=zlabel('Target Value');
    set(xlay,'Rotation',20);
    set(ylay,'Rotation',-30);   
% 
% surf(X,Y,z);
%% 训练
global trainX trainY trainZ testX testY testZ
    trainX=x(1:800);
    trainY=y(1:800);
    trainZ=z(1:800);
    testX=x(801:1000);
    testY=y(801:1000);
    testZ=z(801:1000);
    % 直接GRNN建模
    GRNN=newgrnn([trainX; trainY],trainZ,0.0001);
    Tz=sim(GRNN,[testX;testY]);  % 计算；
    PerfGrnn=perform(GRNN,Tz,testZ);
    % SA优化GRNN建模
    ObjectiveFunction = @SAGRNN;
    lb = 0.0001;
    ub =3;
    start =1 ;
    options = saoptimset('MaxIter',1000, 'StallIterLim', 300, 'TolFun',1e-9,'AnnealingFcn',...
        @annealingboltz,'InitialTemperature',300,'TemperatureFcn',@temperatureboltz,'ReannealInterval',300);
    % ... 'PlotFcns', {@saplotbestx,@saplotbestf,@saplotx,@saplotf});
    [xsagrnn,fsagrnn] = simulannealbnd(ObjectiveFunction,start,lb,ub,options);
    PerfsaGrnn=fsagrnn;
    
    %% 绘图
    spd=xsagrnn;
    NewGRNNNet=newgrnn([trainX; trainY],trainZ,spd);
    Tzsa=sim(NewGRNNNet,[testX;testY]);  % 计算；
    PerfSAGrnn=perform(NewGRNNNet,Tzsa,testZ);
    
    GRNN=newgrnn([trainX; trainY],trainZ,0.0001);
    Tz=sim(GRNN,[testX;testY]);  % 计算；
    PerfGrnn=perform(GRNN,Tz,testZ);
    
    for j=1:200
        lateral(j)=j;
    end
    
    figure;
    plot(lateral,testZ,'ko-');
    hold on
    plot(lateral,Tzsa,'b*');
    plot(lateral,Tz,'rx');
    axis([0 200 -2.5 0.5]);
    
    xlabel('Test Sample No.');
    ylabel('Target Value');
    
     legend('Calculated','SA-GRNN','GRNN');
     hold off
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    