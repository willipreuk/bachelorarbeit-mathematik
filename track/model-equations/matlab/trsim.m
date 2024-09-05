function [ t, y ] = trsim,
% TrSim
%
% Dynamische Simulation eines landwirtschaftlichen Nutzfahrzeugs
%
% Martin-Luther-Universitaet Halle-Wittenberg
% FB Mathematik und Informatik
% Institut fuer Numerische Mathematik
%
% Mathematisches Praktikum, Sommersemester 2003
%
% Bearbeiter :   T. Hertig
% Betreuer :     M. Arnold   (arnold@mathematik.uni-halle.de)
% Version vom :  01. Juli 2003

% -> Anregungsfrequenz
freq = 1.0;

% -> Gleichgewichtslage als Anfangswert der dynamischen Simulation
%      y0(1) ... z_a
%      y0(2) ... hfa
%      y0(3) ... \phi_a
%      y0(4) ... \phi_s

tol0 = 1.0e-8;

it   = 0;
yggw = [ 1.3,  0.0,  0.03,  -0.01 ]';

del  = zeros ( size ( yggw ) );
while ( (norm(del)>tol0) | (it==0) ),
    del  = jac(yggw,freq) \ awert(yggw,freq);
    yggw = yggw - del;
    it   = it + 1;
end;  

'Gleichgewichtslage fuer', yggw,

% -> dynamische Simulation
te   = 10.0;
tdel = 0.01;

y0 = zeros ( 8, 1 );
y0(1) = yggw(1);
y0(3) = y0(1) - 0.195;
y0(5) = yggw(3);
y0(7) = yggw(4);

hfa = yggw(2);

res  = trrhs ( 0, y0, hfa, freq );
resv = res(2:2:8);
'Residuum fuer Anfangswerte', resv,

options = odeset ( 'AbsTol', 1e-8, 'RelTol', 1e-8 );
[ t, y ] = ode45 ( @trrhs, 0:tdel:te, y0, options, hfa, freq );

% -> Ergebnisplot
ystr = [ 'Anhaenger  z_a [m]    '
         'Spritze z_s [m]       '
         'Anhaenger \phi_a [deg]'
         'Spritze \phi_s [deg]  ' ];

zsmin  =  1.0;
zsmax  =  1.5;
zamin  =  1.0;
zamax  =  1.5;
degmin = -6.0;
degmax =  6.0;
     
scal = [ 1.0 1.0 180/pi 180/pi ]; 
ymin = [ zsmin zamin degmin degmin ];
ymax = [ zsmax zamax degmax degmax ];
     
figure ( 1 );
for iplot=1:4,
    subplot ( 2, 2, iplot );
    plot ( t, y(:,2*iplot-1)*scal(iplot) );
    xlabel ( 't' );
    ylabel ( ystr(iplot,:) );
    title ( sprintf ( 'Anregungsfrequenz  %4.2f Hz', freq ) );
    set ( gca, 'YLim', [ ymin(iplot) ymax(iplot) ] );
end;

orient landscape,

hza   = 1.3;
ampl  = 0.1;
l     = 2.0;
shalb = 0.9;

v    = l*freq;

phase = [ 0.4  0.1 ];
sgn   = [ 1.0 -1.0 ];
lrstr = [ 'Rad links '
          'Rad rechts' ];

figure ( 2 );
for iplot=1:2,
    subplot ( 1, 2, iplot );
    fahrbahn = ampl * sin(2*pi*((v/l)*t+phase(iplot)));
    rad      = y(:,1) - hza*cos(y(:,5)) + sgn(iplot) * shalb*sin(y(:,5));
    plot ( t, fahrbahn, '--', t, rad, ':' );
    legend ( 'Anregung  u(t)', 'Auslenkung Rad  z(t)' );
    xlabel ( 't' );
    ylabel ( lrstr(iplot,:) );
    title ( sprintf ( 'Anregungsfrequenz  %4.2f Hz', freq ) );
    set ( gca, 'YLim', [ -0.15 0.15 ] );
end;

orient landscape,