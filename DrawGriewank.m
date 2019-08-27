function DrawGriewank()
%Draw Griewank function graphic
x=[-5:0.1:5];
y=x;
[X,Y]=meshgrid(x,y);
[row,col]=size(X);
for l=1:col
   for h=1:row
       z(h,l)=Griewank([X(h,l),Y(h,l)]);
   end
end
surf(X,Y,z);
hold on

    xlay=xlabel('Dimension x');
    ylay=ylabel('Dimension y');
    zlay=zlabel('Target Value');
    title('');
    set(xlay,'Rotation',20);
    set(ylay,'Rotation',-30);
shading interp
