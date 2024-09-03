function moment = f2mom ( crot, cforce, force );
% F2Mom
%
% Conversion of Forces TWO Moments
%
% Joint project of the Departments of Agriculture, Prof. Dr. P. Pickel,
% and Mathematics / Computer Science, Institute of Numerical Mathematics
%
% Author :      Prof. Dr. M. Arnold, arnold@mathematik.uni-halle.de
% Version of :  Jul 8, 2003
%
% Parameters:
%   crot   (input)  : position of the centre of rotation
%   cforce (input)  : position of force marker
%   force  (input)  : force vector
%   moment (output) : moment
%
% Example:
%   see evaleom.m

% -> vector from centre of rotation to force marker
rvec = cforce - crot;

% -> moment in x-direction
moment = rvec(1) * force(2) - rvec(2) * force(1);
