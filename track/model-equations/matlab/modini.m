function [ param, upar ] = modini,
% ModIni
%
% Agricultural device: Initialization of Model data
%
% Joint project of the Departments of Agriculture, Prof. Dr. P. Pickel,
% and Mathematics / Computer Science, Institute of Numerical Mathematics
%
% Author :      Prof. Dr. M. Arnold, arnold@mathematik.uni-halle.de
% Version of :  Jul 10, 2003
%
% Parameters:
%   param  (output) : structure with system parameters
%   upar   (output) : parameters of external excitations
%
%
% Example:
%   see devsim.
% -> general data
param.nq        = 4;               %           number of position coordinates

param.g         = 9.81;            % [m/s^2]   gravitational constant
param.v         = 2.0;             % [m/s]     velocity
param.te        = 10.0;            % [s]       end time

% -> geometrical data
param.hha       = 2.8;             % [m]       overall height of car
param.hsa       = 0.3;             % [m]       vertical displacement couple markers car
param.hza       = 1.3;             % [m]       vertical displacement of wheels
param.hfa       = 0.0;             % [m]       end point vertical spring at the car (to be determined)
param.hss       = 0.105;           % [m]       vertical displacement couple markers device
param.hfs       = 0.27;            % [m]       end point vertical spring at the device

param.sshalb    = 9.0;             % [m]       halfwidth of device
param.shalb     = 0.9;             % [m]       horizontal displacement of wheels
param.bahalb    = 0.6;             % [m]       horizontal displacement couple markers car
param.bshalb    = 0.2;             % [m]       horizontal displacement couple markers device

% -> mass and inertia data
param.ms        = 570.0;           % [kg]      mass car
param.Js        = 6000.0;          % [kg*m^2]  moment of inertia car
param.ma        = 5500.0;          % [kg]      mass device
param.Ja        = 500.0;           % [kg*m^2]  moment of inertia device

% -> parameters of force elements
param.cr        = 1.0e6;           % [N/m]     spring constant of tyres
param.dr        = 0.05;            % [-]       damping rate of tyres
param.clf       = 87000.0;         % [N/m]     spring constant of vertical spring
param.dlf       = 0.45;            % [-]       damping rate of vertical spring
param.cqf       = 6300.0;          % [N/m]     spring constant of horizontal springs
param.dqf       = 0.45;            % [-]       damping rate of horizontal springs

% -> parameters of external excitations
upar.type       = 'harmonic';      %           [ 'harmonic' 'jump' 'user data' ]
upar.ampl       = 0.1;             % [m]       amplitude
upar.wavelen    = 2.0;             % [m]       wavelength
upar.phas_l     = 0.4;             % [-]       phase shift harmonic excitation left wheel
upar.phas_r     = 0.1;             % [-]       phase shift harmonic excitation left wheel
upar.jumppos    = 1.0 * param.v;   % [m]       position of jump event
