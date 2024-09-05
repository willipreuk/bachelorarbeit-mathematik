function fawert =  awert ( y, freq );


hsa       =0.3;
hza       =1.3;
hfs       =0.27;
hss       =0.105;
ms        =570;
Js        =6000;
ma        =5500;
Ja        =500;
bshalb    =0.2;
bahalb    =0.6;
shalb     =0.9;
cr        =1.0e6; 
cqf       =6.3e3; 
clf       =87000;
dr        =0.05;
dqf       =0.45;
dlf       =0.45;
% v         =0.5;
l         =2;
n         =4; 
g         =9.81;
ampl      =0.1;

v = l*freq;

time                = 0.0;
fahrbahnrechts      = ampl * sin(2*pi*((v/l)*time+0.1));
fahrbahnlinks       = ampl * sin(2*pi*((v/l)*time+0.4));
fahrbahnrechtspunkt = ampl * 2*pi*(v/l)*cos(2*pi*((v/l)*time+0.1));
fahrbahnlinkspunkt  = ampl * 2*pi*(v/l)*cos(2*pi*((v/l)*time+0.4));

% fahrbahnrechts      = 2*pi*(v/l)*sin(2*pi*(v/l)*time+2);
% fahrbahnlinks       = 2*pi*(v/l)*sin(2*pi*(v/l)*time+1.1);
% fahrbahnrechtspunkt = (2*pi*(v/l))^2*cos(2*pi*(v/l)*time+2);
% fahrbahnlinkspunkt  = (2*pi*(v/l))^2*cos(2*pi*(v/l)*time+1.1);

kr = 2*dr*sqrt(cr*ma);

zal=  y(1)-hza*cos(y(3))+shalb*sin(y(3));
zar=  y(1)-hza*cos(y(3))-shalb*sin(y(3));
zalpunkt=0.0;
zarpunkt=0.0;
rRly=-shalb*cos(y(3))-hza*sin(y(3));
rRlz=-shalb*sin(y(3))+hza*cos(y(3));
rRry=shalb*cos(y(3))-hza*sin(y(3));
rRrz=shalb*sin(y(3))+hza*cos(y(3));
rsaly=-bahalb*cos(y(3))-hsa*sin(y(3));
rsalz=-bahalb*sin(y(3))+hsa*cos(y(3));
rsary=bahalb*cos(y(3))-hsa*sin(y(3));
% rsarz=-bahalb*sin(y(3))+hsa*cos(y(3)),
rsarz=bahalb*sin(y(3))+hsa*cos(y(3));
rssly=-bshalb*cos(y(4))-(hfs+hss)*sin(y(4));                    
rsslz=-bshalb*sin(y(4))+(hfs+hss)*cos(y(4));                    
rssry=bshalb*cos(y(4))-(hfs+hss)*sin(y(4));                    
rssrz=bshalb*sin(y(4))+(hfs+hss)*cos(y(4));                    
ergfahrbahnlinks=fahrbahnlinks;
ergfahrbahnrechts=fahrbahnrechts;
ergfahrbahnlinkspunkt=fahrbahnlinkspunkt;
ergfahrbahnrechtspunkt=fahrbahnrechtspunkt;
Ff=clf*(y(1)+y(2)*cos(y(3))-(y(1)-0.195)-hfs*cos(y(4)));      


yasl=bahalb*cos(y(3))-bshalb*cos(y(4))+hsa*sin(y(3))-(hss+hfs)*sin(y(4));
% zasl=bahalb*sin(y(3))-bshalb*sin(y(4))-hsa*cos(y(3))+(hss+hfs)*cos(y(4)),
zasl=y(1)-(y(1)-0.195+hfs)+bahalb*sin(y(3))-bshalb*sin(y(4))-hsa*cos(y(3))+(hss+hfs)*cos(y(4));
yasr=-bahalb*cos(y(3))+bshalb*cos(y(4))+hsa*sin(y(3))-(hss+hfs)*sin(y(4));
% zasr=-bahalb*sin(y(3))+bshalb*sin(y(4))-hsa*cos(y(3))+(hss+hfs)*cos(y(4)),
zasr=y(1)-(y(1)-0.195+hfs)-bahalb*sin(y(3))+bshalb*sin(y(4))-hsa*cos(y(3))+(hss+hfs)*cos(y(4));
FAszr=cqf*zasr;
FAszl=cqf*zasl;
FAsyr=cqf*yasr;
FAsyl=cqf*yasl;

MRl=rRly*(cr*(zal-ergfahrbahnlinks)+kr*(zalpunkt-ergfahrbahnlinkspunkt));
MRr=rRry*(cr*(zar-ergfahrbahnrechts)+kr*(zarpunkt-ergfahrbahnrechtspunkt)); 
% Mrsal=rsaly*(-FAszl)-rsalz*(-FAsyl);                     
% Mrsar=rsary*(-FAszr)-rsarz*(-FAsyr);                              
% Mrssl=rssly*FAszl-rsslz*FAsyl;                     
% Mrssr=rssry*FAszr-rssrz*FAsyr;                              
Mrsal=-rsaly*(-FAszl)+rsalz*(-FAsyl);                     
Mrsar=-rsary*(-FAszr)+rsarz*(-FAsyr);                              
Mrssl=-rssly*FAszl+rsslz*FAsyl;                     
Mrssr=-rssry*FAszr+rssrz*FAsyr;                              

fawert = zeros ( 4, 1 );

fawert(3)=MRl+MRr+Mrsal+Mrsar;
fawert(4)=Mrssl+Mrssr;
fawert(2)= ms*g-Ff-FAszr-FAszl;
fawert(1)=(cr*(zal-ergfahrbahnlinks)+kr*(zalpunkt-ergfahrbahnlinkspunkt)) ...
      +(cr*(zar-ergfahrbahnrechts)+kr*(zarpunkt-ergfahrbahnrechtspunkt)) ...
      +FAszl+FAszr+Ff;
