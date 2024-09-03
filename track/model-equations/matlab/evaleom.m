function [ qpp, uact, qc, fact ] = evaleom ( time, q, qp, param, upar ),
% EvalEoM
%
% Evaluation of the Equations of Motion for an agricultural device
%
% Joint project of the Departments of Agriculture, Prof. Dr. P. Pickel,
% and Mathematics / Computer Science, Institute of Numerical Mathematics
%
% Author :      Prof. Dr. M. Arnold, arnold@mathematik.uni-halle.de
% Version of :  Jul 8, 2003
%
% Parameters:
%   time   (input)  : actual time
%   q      (input)  : vector of actual position coordinates  
%                       [ z_a; z_s; \phi_a; \phi_s ]
%   qp     (input)  : vector of actual velocity coordinates  
%                       [ \dot{z}_a; \dot{z}_s; \dot{\phi}_a; \dot{\phi}_s ]
%   param  (input)  : structure with system parameters
%   upar   (input)  : parameters of external excitations
%   qpp    (output) : vector of actual acceleration coordinates  
%                       [ \ddot{z}_a; \ddot{z}_s; \ddot{\phi}_a; \ddot{\phi}_s ]
%   uact   (output) : structure of actual time excitations
%                       [ ur_l(time); ur_r(time); urp_l(time); urp_r(time) ]
%   qc     (output) : structure with actual positions of all couple markers
%                       qc.qr_l    ... position of left wheel
%                       qc.qr_r    ... position of right wheel
%                       qc.qsa_l   ... end point car of left horizontal spring
%                       qc.qsa_r   ... end point car of right horizontal spring
%                       qc.qss_l   ... end point device of left horizontal spring
%                       qc.qss_r   ... end point device of right horizontal spring
%                       qc.qva     ... end point car of vertical spring
%                       qc.qvs     ... end point device of vertical spring
%   fact   (output) : structure with actual forces and momenta
%                       fact.fr_l  ... tyre force left wheel
%                       fact.fr_r  ... tyre force right wheel
%                       fact.mr_l  ... moment resulting from tyre force left wheel
%                       fact.mr_r  ... moment resulting from tyre force right wheel
%                       fact.fs_l  ... force of left horizontal spring
%                       fact.fs_r  ... force of right horizontal spring
%                       fact.msa_l ... moment (car) resulting from left horizontal spring
%                       fact.msa_r ... moment (car) resulting from right horizontal spring
%                       fact.mss_l ... moment (device) resulting from left horizontal spring
%                       fact.mss_r ... moment (device) resulting from right horizontal spring
%                       fact.fv    ... force of vertical spring
%                       fact.mva   ... moment (car) resulting from vertical spring
%                       fact.mvs   ... moment (device) resulting from vertical spring
%
% Example:
%   see evalrhs.m

% -> read physical data from input
z_a    = q(1);
z_s    = q(2);
phi_a  = q(3);
phi_s  = q(4);
zp_a   = qp(1);
zp_s   = qp(2);
phip_a = qp(3);
phip_s = qp(4);

% -> evaluate external excitations
uact = timeex ( time, param, upar );

% -> compute all couple markers and their time derivatives
crot_a  = [ 0, z_a ]';
crotp_a = [ 0, zp_a ]';
crot_s  = [ 0, z_s + param.hfs ]';
crotp_s = [ 0, zp_s ]';

[ qc.qr_l, qrp_l ]   = rotvec ( crot_a, crotp_a, phi_a, phip_a  ...
                              , [  param.shalb  -param.hza ]'   );
[ qc.qr_r, qrp_r ]   = rotvec ( crot_a, crotp_a, phi_a, phip_a  ...
                              , [ -param.shalb  -param.hza ]'   );

[ qc.qsa_l, qsap_l ] = rotvec ( crot_a, crotp_a, phi_a, phip_a  ...
                              , [  param.bahalb -param.hsa ]'   );
[ qc.qsa_r, qsap_r ] = rotvec ( crot_a, crotp_a, phi_a, phip_a  ...
                              , [ -param.bahalb -param.hsa ]'   );
[ qc.qss_l, qssp_l ] = rotvec ( crot_s, crotp_s, phi_s, phip_s          ...
                              , [  param.bshalb -param.hfs-param.hss ]' );
[ qc.qss_r, qssp_r ] = rotvec ( crot_s, crotp_s, phi_s, phip_s          ...
                              , [ -param.bshalb -param.hfs-param.hss ]' );

[ qc.qva, qvap ]     = rotvec ( crot_a, crotp_a, phi_a, phip_a  ...
                              , [ 0.0  param.hfa ]'             );
[ qc.qvs, qvsp ]     = rotvec ( crot_s, crotp_s, phi_s, phip_s  ...
                              , [ 0.0  0.0 ]'                   );

% -> contact points wheel in the inertial system (set to be exactly below wheel)
qrini_l  = [ qc.qr_l(1), uact.ur_l  ]';
qrinip_l = [ qrp_l(1),   uact.urp_l ]';
qrini_r  = [ qc.qr_r(1), uact.ur_r  ]';
qrinip_r = [ qrp_r(1),   uact.urp_r ]';

% -> spring forces and momenta
fact.fr_l = fspring ( param.cr, param.dr, param.ma      ...
                    , qc.qr_l, qrp_l, qrini_l, qrinip_l );
fact.fr_r = fspring ( param.cr, param.dr, param.ma      ...
                    , qc.qr_r, qrp_r, qrini_r, qrinip_r );

fact.fs_l = fspring ( param.cqf, param.dqf, (param.ms+param.ma) ...
                    , qc.qss_l, qssp_l, qc.qsa_l, qsap_l );
fact.fs_r = fspring ( param.cqf, param.dqf, (param.ms+param.ma) ...
                    , qc.qss_r, qssp_r, qc.qsa_r, qsap_r );

% fact.fv   = fspring ( param.clf, param.dlf, param.ms ...
%                     , qc.qvs, qvsp, qc.qva, qvap     );

cfrom  = [ qc.qva(1), z_s  + cos(phi_s) * param.hfs ]';
cfromp = [ qvap(1),   zp_s - sin(phi_s) * param.hfs * phip_s ]';
                    
fact.fv   = fspring ( param.clf, param.dlf, param.ms ... 
                    , cfrom, cfromp, qc.qva, qvap    );  % for backward compatibility

fact.fv(1) = 0;                                          % for backward compatibility
                    
fact.mr_l  = f2mom ( crot_a, qc.qr_l,   fact.fr_l );
fact.mr_r  = f2mom ( crot_a, qc.qr_r,   fact.fr_r );

fact.msa_l = f2mom ( crot_a, qc.qsa_l, -fact.fs_l );
fact.msa_r = f2mom ( crot_a, qc.qsa_r, -fact.fs_r );
fact.mss_l = f2mom ( crot_s, qc.qss_l,  fact.fs_l );
fact.mss_r = f2mom ( crot_s, qc.qss_r,  fact.fs_r );

fact.mva   = f2mom ( crot_a, qc.qva,   -fact.fv   );
fact.mvs   = f2mom ( crot_s, qc.qvs,    fact.fv   );    % this moment vanishes identically

% -> set right hand side
zpp_a   = ( fact.fr_l(2) + fact.fr_r(2) - fact.fs_l(2) - fact.fs_r(2)  ...
            - fact.fv(2) ) / param.ma;
zpp_s   = ( fact.fs_l(2) + fact.fs_r(2) + fact.fv(2) ) / param.ms - param.g;
phipp_a = ( fact.mr_l + fact.mr_r + fact.msa_l + fact.msa_r ...
            + fact.mva ) / param.Ja;
phipp_s = ( fact.mss_l + fact.mss_r + fact.mvs ) / ...
            ( param.Js + param.ms*param.hfs^2 );

qpp = [ zpp_a, zpp_s, phipp_a, phipp_s ]';
