a=1;
b=1;
c=1;
d=1;
e=1;
f=1;
for x=-100:100
 for y=-100:100
  z=a*x^2+b*y^2+c*x*y+d*x+e*y+f;
  if(x==-100)&&(y==-100)
   data=[x,y,z];
  else
   data=[data;x,y,z];
  end
 end
end
%plot3(data(:,1:1),data(:,2:2),data(:,3:3))
scatter3(data(:,1:1),data(:,2:2),data(:,3:3),'.')