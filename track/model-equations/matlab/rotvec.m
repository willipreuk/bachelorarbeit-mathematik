function [ vec, vecp ] = rotvec ( crot, crotp, phi, phip, vec0 );
% RotVec
%
% Planar Rotation and shift of Vectors
%
% Joint project of the Departments of Agriculture, Prof. Dr. P. Pickel,
% and Mathematics / Computer Science, Institute of Numerical Mathematics
%
% Author :      Prof. Dr. M. Arnold, arnold@mathematik.uni-halle.de
% Version of :  Jul 8, 2003
%
% Parameters:
%   crot   (input)  : position of the centre of rotation
%   crotp  (input)  : velocity of the centre of rotation
%   phi    (input)  : angle of rotation
%   phip   (input)  : time derivative of the angle of rotation
%   vec0   (input)  : input vector
%   vec    (output) : position vector in Cartesian coordinates
%   vecp   (output) : velocity vector in Cartesian coordinates
%
% Example:
%   see evaleom.m

% -> rotation matrix
Arot = [ cos(phi) -sin(phi)
         sin(phi)  cos(phi) ];

Arotp = phip * [ -sin(phi) -cos(phi)
                  cos(phi) -sin(phi) ];
     
% -> vector transformation
vec  = crot  + Arot  * vec0;
vecp = crotp + Arotp * vec0;
