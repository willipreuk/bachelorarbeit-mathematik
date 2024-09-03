function force = fspring ( c, d, m, cfrom, cfromp, cto, ctop );
% FSpring
%
% Force of a linear Spring
%
% Joint project of the Departments of Agriculture, Prof. Dr. P. Pickel,
% and Mathematics / Computer Science, Institute of Numerical Mathematics
%
% Author :      Prof. Dr. M. Arnold, arnold@mathematik.uni-halle.de
% Version of :  Jul 8, 2003
%
% Parameters:
%   c      (input)  : spring constant
%   d      (input)  : damping rate
%   m      (input)  : mass
%   cfrom  (input)  : position of from marker
%   cfromp (input)  : velocity of from marker
%   cto    (input)  : position of to marker
%   ctop   (input)  : velocity of to marker
%   force  (output) : force vector
%
% Example:
%   see evaleom.m

% -> damping constant
k = 2 * d * sqrt ( c*m );

% -> absolute value of the spring force
slen = norm ( cto - cfrom );
f = c * slen + k * sum ( (cto-cfrom).*(ctop-cfromp) ) / slen;

% -> force vector
force = f * ( cto - cfrom ) / slen;
