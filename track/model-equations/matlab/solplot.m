function func = solplot
% SolPlot
%
% Plot typical solution for an agricultural Device
%
% Joint project of the Departments of Agriculture, Prof. Dr. P. Pickel,
% and Mathematics / Computer Science, Institute of Numerical Mathematics
%
% Author :      Prof. Dr. M. Arnold, martin.arnold@mathematik.uni-halle.de
% Version of :  Nov 18, 2008
%
% Parameters
%   see also devplt.m in ../matlab/landw
%
% Example:
%   solplot;  %   print -deps ../ps/solplot.eps

graphwidth  = 12.2;
graphheight = 10.0;

mygraph.facv	    =  1.0;
mygraph.shiftv	    =  0.0;

mygraph.fach	    =  1.2;
mygraph.shifth	    =  0.5;

papershift  =  0.0;

mygraph.ActFontSize      =  8;
mygraph.ActFontSizeSmall =  8;
mygraph.ActLineWidth     =  1;
mygraph.ActLineWidthMark =  2;
mygraph.ActMarkerSize    =  6;
RatioFlag     =  0;

border      = 1.0;

%
% -> Bestimme Groesse des Bildschirms und Skalierungsfaktor fuer die Ausgabe
root = 0;
set ( root, 'Units', 'centimeters' );
screensize = get ( root, 'ScreenSize' );
left   = screensize(1);
bottom = screensize(2);
width  = screensize(3);
height = screensize(4);
set ( root, 'Units', 'pixels' );
if RatioFlag==0,
  screenscale = 1.0;
elseif RatioFlag==1,
  screenscale = (width-2*border) / graphwidth;
end;

%
% -> Definiere den Bildschirmausschnitt
if isempty ( get ( root, 'Children' ) ),
  h = figure;
else
  h = gcf;
  clf;
end;
set ( h, 'Units',    'centimeters' );
set ( h, 'Position', ...
  [ left+width-screenscale*graphwidth-border height-2*border-screenscale*graphheight ...
    screenscale*graphwidth                   screenscale*graphheight ] );
set ( h, 'Units',    'pixels' );

%
% -> Skalierung der Ausgabe in PostScript
%
set ( gcf, 'PaperUnits',    'centimeters' );
set ( gcf, 'PaperPosition', [ papershift papershift graphwidth graphheight ] );

% -> get parameters and data
[ param, upar ] = modini;

t  = load ( '../../../2008/../EUROMECH500/proc/dat/sol_t.dat' );
xx = load ( '../../../2008/../EUROMECH500/proc/dat/sol_xx.dat' );

% -> evaluate frequency
freq = param.v / upar.wavelen;

% -> scaling of graphical output etc.
zsmin  =  1.0;
zsmax  =  1.5;
zamin  =  1.0;
zamax  =  1.5;
degmin = -6.0;
degmax =  6.0;
     
scal = [ 1.0 1.0 180/pi 180/pi ]; 
ymin = [ zsmin zamin degmin degmin ];
ymax = [ zsmax zamax degmax degmax ];

for iplot=1:4,
    subplot ( 2, 2, iplot );
    plot ( t, xx(:,iplot)*scal(iplot), 'k' );
    switch iplot
      case 1
        title ( 'Central body' );
        ylabel ( 'Vertical displacement  z [m]' );
      case 2
        title ( 'Horizontal bar' );
        ylabel ( ' ' );
        set ( gca, 'YAxisLocation', 'right' );
      case 3
        xlabel ( 't' );
        ylabel ( 'Angle of rotation  \phi [deg]' );
      case 4
        xlabel ( 't' );
        ylabel ( ' ' );
        set ( gca, 'YAxisLocation', 'right' );
    end
    set ( gca, 'YLim', [ ymin(iplot) ymax(iplot) ] );
end;

%
% -> Korrektur der Groesse und Lage des Koordinatensystems
allch = get ( gcf, 'Children' );
for al=allch',
    rect    = get ( al, 'Position' );
    rect(2) = rect(2) + mygraph.shiftv * (1-mygraph.facv) * rect(4);
    rect(4) = mygraph.facv * rect(4);
    rect(1) = rect(1) + mygraph.shifth * (1-mygraph.fach) * rect(3);
    rect(3) = mygraph.fach * rect(3);
    set ( al, 'Position', rect );
    set ( al, 'FontSize', mygraph.ActFontSize );
    set ( get ( al, 'XLabel' ), 'FontSize', mygraph.ActFontSize );
    set ( get ( al, 'YLabel' ), 'FontSize', mygraph.ActFontSize );
    set ( get ( al, 'Title' ), 'FontSize', mygraph.ActFontSize );
end
