function func = devsim ( iplot, iprint );
% DevSim
%
% Dynamical Simulation of an agricultural Device
%
% Joint project of the Departments of Agriculture, Prof. Dr. P. Pickel,
% and Mathematics / Computer Science, Institute of Numerical Mathematics
%
% Author :      Prof. Dr. M. Arnold, arnold@mathematik.uni-halle.de
% Version of :  Jul 10, 2003
%
% Parameters:
%   iplot  (input)  : control flag "output"
%                       =1 .. standard output in 2D plots
%                       =2 .. animated output
%   iprint (input)  : control flag "print messages"
%                       0 .. no messages
%                       1 .. print equilibrium position
%                       2 .. "iprint=1" and report progress of computation
%
% Example:
%   devsim ( 1, 1 );

% -> read system parameters
[ param, upar ] = modini;

% -> compute equilibrium position as initial values for dynamical simulation
[ time, q, param ] = equini ( param, upar, iprint );

% -> time integration
tol   = 1.0e-8;
tspan = 0:(1.0e-2):(param.te);
options = odeset ( 'AbsTol', 1e-8, 'RelTol', 1e-8 );

xx0 = zeros ( 2*param.nq, 1 ); 
xx0(1:param.nq) = q;

[ t, xx ] = ode45 ( @evalrhs, tspan, xx0, options, param, upar );

% -> output
if iplot==1,
    devplt ( t, xx, param, upar );
elseif iplot==2,
    devani ( t, xx, param, upar );
end;
