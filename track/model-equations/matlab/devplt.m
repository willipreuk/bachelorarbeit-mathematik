function func = devplt ( t, xx, param, upar );
% DevPlt
%
% Animation of simulation data for an agricultural Device
%
% Joint project of the Departments of Agriculture, Prof. Dr. P. Pickel,
% and Mathematics / Computer Science, Institute of Numerical Mathematics
%
% Author :      Prof. Dr. M. Arnold, arnold@mathematik.uni-halle.de
% Version of :  Jul 10, 2003
%
% Parameters:
%   t      (input)  : vector of time instances
%   xx     (input)  : vector of simulation data, xx(i,:) is the solution at time t(i)
%                       [ z_s z_a phi_s phi_a zp_s zp_a phip_s phip_a ]
%   param  (input)  : structure with system parameters
%   upar   (input)  : parameters of external excitations
%
% Example:
%   see devsim.m

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

ystr = [ 'Anhaenger  z_a [m]    '
         'Spritze z_s [m]       '
         'Anhaenger \phi_a [deg]'
         'Spritze \phi_s [deg]  ' ];

% -> plot
clf,
for iplot=1:4,
    subplot ( 2, 2, iplot );
    plot ( t, xx(:,iplot)*scal(iplot) );
    xlabel ( 't' );
    ylabel ( ystr(iplot,:) );
    title ( sprintf ( 'Anregungsfrequenz  %4.2f Hz', freq ) );
    set ( gca, 'YLim', [ ymin(iplot) ymax(iplot) ] );
end;

orient landscape,
