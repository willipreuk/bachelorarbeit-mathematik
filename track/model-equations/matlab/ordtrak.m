function func = ordtrak ( ideg, idamp, icomp );
% OrdTrak
%
% Order plots, Benchmark "Traktor", b/w version
%
% Parameter: 
%   ideg   (input)  : control flag "spline degree": 'Cubic', 'Linear'
%   idamp  (input)  : control flag "damping": 'damped', 'undamped'
%   icomp  (input)  : control flag "comparison": 'order', 'del'
%
% Example
%   ordtrak ( 'Linear', 'damped',   'del' );     % print -deps ../ps/ord_linear_damped_del.eps;
%   ordtrak ( 'Linear', 'undamped', 'del' );     % print -deps ../ps/ord_linear_undamped_del.eps;
%   ordtrak ( 'Cubic',  'damped',   'del' );     % print -deps ../ps/ord_cubic_damped_del.eps;
%   ordtrak ( 'Cubic',  'undamped', 'del' );     % print -deps ../ps/ord_cubic_undamped_del.eps;
%   ordtrak ( 'Linear', 'damped',   'order' );   % print -deps ../ps/ord_linear_damped_order.eps;
%   ordtrak ( 'Linear', 'undamped', 'order' );   % print -deps ../ps/ord_linear_undamped_order.eps;
%
%
% Author :      Martin Arnold, martin.arnold@mathematik.uni-halle.de
% Version of :  Nov 18, 2008

graphwidth  = 12.2;
graphheight =  8.0;
facv	    =  1.0;
shiftv	    =  0.5;
fach	    =  1.0;
shifth	    =  0.0;

papershift  =  0.0;

ActFontSize     =   8;
ActLineWidth    = 1.5;
SmallLineWidth  = 0.5;
ActMarkerSize   =   6;
RatioFlag       =   0;

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
if length ( get ( root, 'Children' ) ) == 0,
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

%
% -> data
if strcmp(icomp,'del'),
    eval ( [ 'ttr = load(''../../../../2005/hro05/dat/', ideg, '_', idamp, '/ord5_dense_del010000.dat'');' ] );
    eval ( [ 'ttm = load(''../../../../2005/hro05/dat/', ideg, '_', idamp, '/ord5_dense_del005000.dat'');' ] );
%     if strcmp(ideg,'Linear'),
%         eval ( [ 'ttb = load(''../../../../2005/hro05/dat/', ideg, '_', idamp, '/ord5_dense_del002500.dat'');' ] );
%         eval ( [ 'ttc = load(''../../../../2005/hro05/dat/', ideg, '_', idamp, '/ord5_dense_del001250.dat'');' ] );
%     end;
elseif strcmp(icomp,'order'),
    eval ( [ 'ttr = load(''../../../../2005/hro05/dat/', ideg, '_', idamp, '/ord2_dense_del002500.dat'');' ] );
    eval ( [ 'ttm = load(''../../../../2005/hro05/dat/', ideg, '_', idamp, '/ord4_dense_del002500.dat'');' ] );
    eval ( [ 'ttb = load(''../../../../2005/hro05/dat/', ideg, '_', idamp, '/ord5_dense_del002500.dat'');' ] );
end;

%
% -> plot
loglog ( ...
         ttr(:,1), ttr(:,2), '--k'   ...
       , ttm(:,1), ttm(:,2), ':k'    ...
       );

hold on;

if ( ( strcmp(icomp,'del') ) & ( strcmp(ideg,'Linear') ) ),
elseif strcmp(icomp,'order'),
    loglog ( ttb(:,1), ttb(:,2), '-.k' );
end;

hold off;

ylim = get ( gca, 'YLim' );
ylim(2) = min ( ylim(2), 1 );
set ( gca, 'YLim', ylim );

title ( [ 'u(t) : ', ideg, ' interpolation (', idamp, ' tyre force)' ] );
xlabel ( 'h' );
ylabel ( 'Error' );

%
% -> Korrektur der Groesse und Lage der Koordinatensysteme
allax = get ( gcf, 'Children' )';
for ax = allax,
  rect    = get ( ax, 'Position' );
  rect(2) = rect(2) + shiftv * (1-facv) * rect(4);
  rect(4) = facv * rect(4);
  rect(1) = rect(1) + shifth * (1-fach) * rect(3);
  rect(3) = fach * rect(3);
  set ( ax, 'Position', rect );
  set ( ax, 'FontSize', ActFontSize );
  set ( get ( ax, 'XLabel' ), 'FontSize', ActFontSize );
  set ( get ( ax, 'YLabel' ), 'FontSize', ActFontSize );
  set ( get ( ax, 'Title' ), 'FontSize', ActFontSize );
  allch = get ( gca, 'Children' );
  for al=allch',
    if strcmp(get(al,'Type'),'line'),
      set ( al, 'LineWidth', ActLineWidth );
    end;
  end;
end;

% -> legend
if strcmp(icomp,'del'),
    if strcmp(ideg,'Cubic'),
        legend ( ...
                 ' \Delta = 0.01'          ...
               , ' \Delta = 0.005'         ...
               , 'Location', 'NorthWest'   ...
               );
    elseif strcmp(ideg,'Linear'),
        legend ( ...
                 ' \Delta = 0.01'          ...
               , ' \Delta = 0.005'         ...
               , 'Location', 'NorthWest'   ...
               );
    end;
elseif strcmp(icomp,'order'),
   legend ( ...
            ' p = 2  (Euler-Heun)'          ...
          , ' p = 4  (Runge-Kutta)'         ...
          , ' p = 5  (Dormand-Prince)'      ...
          , 'Location', 'NorthWest'   ...
          );
end;

hold on;
if strcmp(icomp,'del'),
    if strcmp(ideg,'Cubic'),
        if strcmp(idamp,'damped'),
            loglog ( [1.1e-4 2e-3], [1.1e-4 2e-3].^3*0.04, '-k', 'LineWidth', SmallLineWidth );
            text ( 1.65e-3, 5.5e-11, '~ h^3', 'FontSize', ActFontSize );
            loglog ( [3.0e-3 9.5e-3], [3.0e-3 9.5e-3].^5*2.0e4, '-k', 'LineWidth', SmallLineWidth );
            text ( 6.6e-3, 6.8e-8, '~ h^5', 'FontSize', ActFontSize );
        elseif strcmp(idamp,'undamped'),
            loglog ( [1.0e-3 9.5e-3], [1.0e-3 9.5e-3].^5*2.0e5, '-k', 'LineWidth', SmallLineWidth );
            text ( 4.5e-3, 1.0e-7, '~ h^5', 'FontSize', ActFontSize );
        end;
    elseif strcmp(ideg,'Linear'),
        if strcmp(idamp,'damped'),
            loglog ( [5.0e-4 9.0e-3], [5.0e-4 9.0e-3].^1*8.0e-2, '-k', 'LineWidth', SmallLineWidth );
            text ( 4.0e-3, 2.0e-4, '~ h', 'FontSize', ActFontSize );
        elseif strcmp(idamp,'undamped'),
            loglog ( [5.0e-4 9.5e-3], [5.0e-4 9.5e-3].^2*0.2, '-k', 'LineWidth', SmallLineWidth );
            text ( 3.6e-3, 1.5e-6, '~ h^2', 'FontSize', ActFontSize );
        end;
    end;
elseif strcmp(icomp,'order'),
    if strcmp(idamp,'damped'),
        loglog ( [3.0e-4 8.0e-3], [3.0e-4 8.0e-3].^1*0.08, '-k', 'LineWidth', SmallLineWidth );
        text ( 5.6e-3, 3.2e-4, '~ h', 'FontSize', ActFontSize );
        loglog ( [2.0e-3 9.8e-3], [2.0e-3 9.8e-3].^2*3.5e2, '-k', 'LineWidth', SmallLineWidth );
        text ( 4.4e-3, 2.5e-2, '~ h^2', 'FontSize', ActFontSize );
    elseif strcmp(idamp,'undamped'),
        loglog ( [3.0e-4 9.5e-3], [3.0e-4 9.5e-3].^2*4.0e-1, '-k', 'LineWidth', SmallLineWidth );
        text ( 3.0e-3, 2.0e-6, '~ h^2', 'FontSize', ActFontSize );
        loglog ( [4.0e-3 9.8e-3], [4.0e-3 9.8e-3].^4*2.0e5, '-k', 'LineWidth', SmallLineWidth );
        text ( 7.0e-3, 3.2e-4, '~ h^4', 'FontSize', ActFontSize );
        loglog ( [1.5e-4 6.0e-3], [1.5e-4 6.0e-3].^2*1.0e3, '-k', 'LineWidth', SmallLineWidth );
        text ( 2.6e-4, 3.0e-5, '~ h^2', 'FontSize', ActFontSize );
    end;
end;
hold off;
