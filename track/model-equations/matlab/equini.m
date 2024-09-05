function [ time, q, parout ] = equini ( parin, upar, iprint );
% EquiIni
%
% Equilibrium of agricultural device, used for Initialization
%
% Joint project of the Departments of Agriculture, Prof. Dr. P. Pickel,
% and Mathematics / Computer Science, Institute of Numerical Mathematics
%
% Author :      Prof. Dr. M. Arnold, arnold@mathematik.uni-halle.de
% Version of :  Jul 8, 2003
%
% Parameters:
%   parin  (input)  : structure with system parameters on input
%   upar   (input)  : parameters of external excitations
%   iprint (input)  : control flag "print messages"
%                       0 .. no messages
%                       1 .. print equilibrium position
%                       2 .. print equilibrium position and report iteration progress
%   time   (output) : initial time
%   q      (output) : vector of initial position coordinates  
%                       [ z_a; z_s; \phi_a; \phi_s ]
%   parout (output) : structure with system parameters on output ( param.hfa may be modified )
%
% Example:
%   see devsim.m

% -> initializations
param = parin;                              % initialize system parameters

time = 0.0;                                 % initial time

qact  = zeros ( param.nq, 1 );
qactp = zeros ( param.nq, 1 );

qact(1) = param.hza;                        % vertical position z_a of the center of gravity car
qact(2) = qact(1) - param.hsa + param.hss;  % vertical position z_s of the center of gravity device
qact(3) = 0.0;                              % roll angle \phi_a of the car
qact(4) = 0.0;                              % roll angle \phi_s of the device

% -> numerical parameters
tol = 1e-10;                                % error tolerance in Newton's method
maxit = 20;                                 % maximum number of Newton steps
maxnorm = 1.0e8;                            % upper bound for valid Newton corrections

% -> Newton's method
it = 0;
actnorm = 0.0;

while ( actnorm > tol*(1+norm(qact)) )  | ( it==0 ),
    
    % -> function call
    [ qactpp0, uact, qc, fact ] = evaleom ( time, qact, qactp, param, upar );

    % -> difference approximation of Jacobian
    jac = zeros ( param.nq, param.nq );
    for j=1:param.nq,   
        if j==2,
            del = sqrt(eps) * max([abs(param.hfa);sqrt(sqrt(eps))]);
            qsave     = param.hfa;
            param.hfa = param.hfa + del;
        else
            del = sqrt(eps) * max([abs(qact(j));sqrt(sqrt(eps))]);
            qsave   = qact(j);
            qact(j) = qact(j) + del;
        end;
        [ qactpp, uact, qc, fact ] = evaleom ( time, qact, qactp, param, upar );
        jac(:,j) = ( qactpp - qactpp0 ) / del;
        if j==2,  param.hfa = qsave;  else  qact(j) = qsave;  end;
    end;    
    
    % -> Newton step
    qnewt = jac \ qactpp0;
    actnorm = norm ( qnewt );
    if actnorm>maxnorm,  error ( 'EquIni: Divergence of Newton''s method detected' );  end;
    
    qact((1:param.nq)~=2) = qact((1:param.nq)~=2) - qnewt((1:param.nq)~=2);
    qact(2) = qact(1) - param.hsa + param.hss;
    param.hfa = param.hfa - qnewt(2); 

    it = it + 1;
    if it>maxit,  error ( 'EquIni: Maximum number of Newton steps exceeded' );  end;

    if iprint>=2,
        fprintf ( '\nNorm of Newton increment in step %3i : %12.4e', it, actnorm );
    end;    
    
end;

% -> save data for output
q      = qact;
q(2)   = q(1) - param.hsa + param.hss;        % vertical position z_s of the center of gravity device
parout = param;

% -> print equilibrium position (if applicable)
if iprint>=1,
    fprintf ( '\n\nEquiIni: Newton iteration completed successfully' );
    fprintf ( '\n\nEquilibrium position' ); q,
    fprintf ( '\nParameter hfa = %12.4e', param.hfa );
end;