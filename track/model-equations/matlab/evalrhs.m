function xxp = evalrhs ( time, xx, param, upar );
% EvalRHS
%
% Evaluation of the Right Hand Side for the motion of an agricultural device
%
% Joint project of the Departments of Agriculture, Prof. Dr. P. Pickel,
% and Mathematics / Computer Science, Institute of Numerical Mathematics
%
% Author :      Prof. Dr. M. Arnold, arnold@mathematik.uni-halle.de
% Version of :  Jul 8, 2003
%
% Parameters:
%   time   (input)  : actual time
%   xx     (input)  : actual state vector  
%   param  (input)  : structure with system parameters
%   upar   (input)  : parameters of external excitations
%   xxp    (output) : actual derivative vector  
%
% Example:
%   see ode45.m

% -> read position and velocity coordinates 
q  = xx(1:param.nq);
qp = xx(param.nq+(1:param.nq));

% -> evaluate the equations of motion calling EvalEoM
[ qpp, uact, qc, fact ] = evaleom ( time, q, qp, param, upar );

% -> save velocity and acceleration coordinates
xxp = zeros ( 2*param.nq, 1 );

xxp(1:param.nq)            = qp;
xxp(param.nq+(1:param.nq)) = qpp;
