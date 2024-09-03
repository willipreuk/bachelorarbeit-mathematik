function func = devplot
% DevPlot
%
% Plot typical configuration of an agricultural Device
%
% Joint project of the Departments of Agriculture, Prof. Dr. P. Pickel,
% and Mathematics / Computer Science, Institute of Numerical Mathematics
%
% Author :      Prof. Dr. M. Arnold, martin.arnold@mathematik.uni-halle.de
% Version of :  Nov 17, 2008
%
% Parameters
%   see also devani.m in ../matlab/landw
%
% Example:
%   devplot;  %   print -deps ../ps/devplot.eps

graphwidth  = 12.2;
graphheight =  6.0;

mygraph.facv	    =  1.0;
mygraph.shiftv	    =  0.0;

mygraph.fach	    =  1.2;
mygraph.shifth	    =  0.5;

papershift  =  0.0;

mygraph.ActFontSize      =  7;
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

t  = 1.2;
xx = load ( '../../../2008/../EUROMECH500/proc/dat/endstate.dat' );

% -> scaling of graphical output etc.
zmin =  - 1.0;
zmax =    3.5;
ymin = - 10.0;
ymax =   10.0;

fscal = 1 / 1.6e+4;

% -> evaluate marker positions etc.
q  = xx(         (1:param.nq))';
qp = xx(param.nq+(1:param.nq))';

z_a   = q(1);
z_s   = q(2);
phi_a = q(3);
phi_s = q(4);
[ qpp, uact, qc, fact ] = evaleom ( t, q, qp, param, upar );

crot_a  = [ 0, z_a ]';
[ qua_l, aux ] = rotvec ( crot_a, 0.0, phi_a, 0.0                ...
                        , [  param.shalb  param.hha-param.hza ]' );
[ qua_r, aux ] = rotvec ( crot_a, 0.0, phi_a, 0.0                ...
                        , [ -param.shalb  param.hha-param.hza ]' );
    
ccen_s  = [ 0, z_s ]';
[ qus_l, aux ] = rotvec ( ccen_s, 0.0, phi_s, 0.0       ...
                        , [  param.sshalb  param.hfs ]' );
[ qus_r, aux ] = rotvec ( ccen_s, 0.0, phi_s, 0.0       ...
                        , [ -param.sshalb  param.hfs ]' );
[ qls_l, aux ] = rotvec ( ccen_s, 0.0, phi_s, 0.0       ...
                        , [  param.sshalb -param.hfs ]' );
[ qls_r, aux ] = rotvec ( ccen_s, 0.0, phi_s, 0.0       ...
                        , [ -param.sshalb -param.hfs ]' );
    
% -> plot
fill ( ...
       [ qc.qr_l(1) qua_l(1) qua_r(1) qc.qr_r(1) qc.qr_l(1) ]  ...
     , [ qc.qr_l(2) qua_l(2) qua_r(2) qc.qr_r(2) qc.qr_l(2) ]  ...
     , 'b'                                                     ...
     , 'EdgeColor', 'b'                                        ...
     , 'FaceColor', [ 0.95 0.95 1 ]                            ...
     , 'LineWidth', 2                                          ...
     );
hold on;
fill ( ...
       [ qls_l(1) qus_l(1) qus_r(1) qls_r(1) qls_l(1) ]  ...
     , [ qls_l(2) qus_l(2) qus_r(2) qls_r(2) qls_l(2) ]  ...
     , 'g'                                               ...
     , 'EdgeColor', 'g'                                  ...
     , 'FaceColor', [ 0.95 1 0.95 ]                      ...
     , 'LineWidth', 2                                    ...
     );

plot ( 0,           z_a,         'ob', 'MarkerSize', 8, 'MarkerFaceColor', 'b' );
plot ( qc.qsa_l(1), qc.qsa_l(2), 'ob', 'MarkerSize', 4, 'MarkerFaceColor', 'b' );
plot ( qc.qsa_r(1), qc.qsa_r(2), 'ob', 'MarkerSize', 4, 'MarkerFaceColor', 'b' );
plot ( qc.qva(1),   qc.qva(2),   'ob', 'MarkerSize', 4, 'MarkerFaceColor', 'b' );
    
plot ( 0,           z_s,         'og', 'MarkerSize', 8, 'MarkerFaceColor', 'g' );
plot ( qc.qss_l(1), qc.qss_l(2), 'og', 'MarkerSize', 4, 'MarkerFaceColor', 'g' );
plot ( qc.qss_r(1), qc.qss_r(2), 'og', 'MarkerSize', 4, 'MarkerFaceColor', 'g' );
plot ( qc.qvs(1),   qc.qvs(2),   'og', 'MarkerSize', 4, 'MarkerFaceColor', 'g' );
    
plot ( ...
       qc.qss_l(1) + fact.fs_l(1)*[0.2 1]*fscal  ...
     , qc.qss_l(2) + fact.fs_l(2)*[0.2 1]*fscal  ...
     , 'Color',    'r'                           ...
     , 'LineWidth', 4                            ...
     );
plot ( ...
       qc.qss_r(1) + fact.fs_r(1)*[0.2 1]*fscal  ...
     , qc.qss_r(2) + fact.fs_r(2)*[0.2 1]*fscal  ...
     , 'Color',    'r'                           ...
     , 'LineWidth', 4                            ...
     );
    
hold off;
    
% -> scaling etc.
axis ( [ ymin ymax zmin zmax ] );
set ( gca, 'PlotBoxAspectRatio', [ 1 1 1 ] );
set ( gca, 'DataAspectRatio',    [ 1 1 1 ] );

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
