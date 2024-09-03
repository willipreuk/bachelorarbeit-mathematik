function uact = timeex ( time, param, upar );
% TimeEx
%
% Evaluation of the Time Excitations
%
% Joint project of the Departments of Agriculture, Prof. Dr. P. Pickel,
% and Mathematics / Computer Science, Institute of Numerical Mathematics
%
% Author :      Prof. Dr. M. Arnold, arnold@mathematik.uni-halle.de
% Version of :  Jul 8, 2003
%
% Parameters:
%   time   (input)  : actual time
%   param  (input)  : structure with system parameters
%   upar   (input)  : parameters of external excitations
%   uact   (output) : structure of actual time excitations
%                       [ ur_l(time); ur_r(time); urp_l(time); urp_r(time) ]
%
% Example:
%   [ param, upar ] = modini;
%   uact = timeex ( 0.0, param, upar );

% -> time excitations left and right wheel
if strcmp ( upar.type, 'harmonic' ),
    uact.ur_l  = upar.ampl * sin ( 2*pi*( (param.v/upar.wavelen)*time + upar.phas_l ) );
    uact.ur_r  = upar.ampl * sin ( 2*pi*( (param.v/upar.wavelen)*time + upar.phas_r ) );
    uact.urp_l = 2*pi*(param.v/upar.wavelen) * upar.ampl * cos ( 2*pi*( (param.v/upar.wavelen)*time + upar.phas_l ) );
    uact.urp_r = 2*pi*(param.v/upar.wavelen) * upar.ampl * cos ( 2*pi*( (param.v/upar.wavelen)*time + upar.phas_r ) );
else
    error ( [ 'TimeEx: upar.type = ' upar.type 'not yet implemented' ] );
end;   
 

