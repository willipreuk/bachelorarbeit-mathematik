function func = devani ( t, xx, param, upar );
% DevAni
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

% -> scaling of graphical output etc.
zmin = -0.5;
zmax =  3.0;
ymin = -2.0;
ymax =  2.0;

fscal = 1 / 1.6e+4;

% -> animation
clf,

nt = length ( t );
for it=1:nt,

    % -> read data and evaluate marker positions etc.
    q  = xx(it,1:param.nq)';
    z_a   = q(1);
    z_s   = q(2);
    phi_a = q(3);
    phi_s = q(4);
    qp = xx(it,param.nq+(1:param.nq))';
    [ qpp, uact, qc, fact ] = evaleom ( t(it), q, qp, param, upar );

    crot_a  = [ 0, z_a ]';
    [ qua_l, aux ] = rotvec ( crot_a, 0.0, phi_a, 0.0                ...
                            , [  param.shalb  param.hha-param.hza ]' );
    [ qua_r, aux ] = rotvec ( crot_a, 0.0, phi_a, 0.0                ...
                            , [ -param.shalb  param.hha-param.hza ]' );
    
    ccen_s  = [ 0, z_s ]';
    [ qus_l, aux ] = rotvec ( ccen_s, 0.0, phi_s, 0.0       ...
                            , [  param.sshalb  param.hfs ]' );
    [ qus_r, aux ] = rotvec ( ccen_s, 0.0, phi_s, 0.0       ...
                            , [ -param.sshalb  param.hfs ]' );
    [ qls_l, aux ] = rotvec ( ccen_s, 0.0, phi_s, 0.0       ...
                            , [  param.sshalb -param.hfs ]' );
    [ qls_r, aux ] = rotvec ( ccen_s, 0.0, phi_s, 0.0       ...
                            , [ -param.sshalb -param.hfs ]' );
    
    % -> plot
    fill ( ...
           [ qc.qr_l(1) qua_l(1) qua_r(1) qc.qr_r(1) qc.qr_l(1) ]  ...
         , [ qc.qr_l(2) qua_l(2) qua_r(2) qc.qr_r(2) qc.qr_l(2) ]  ...
         , 'b'                                                     ...
         , 'EdgeColor', 'b'                                        ...
         , 'FaceColor', [ 0.95 0.95 1 ]                            ...
         , 'LineWidth', 2                                          ...
         );
    hold on;
    fill ( ...
           [ qls_l(1) qus_l(1) qus_r(1) qls_r(1) qls_l(1) ]  ...
         , [ qls_l(2) qus_l(2) qus_r(2) qls_r(2) qls_l(2) ]  ...
         , 'g'                                               ...
         , 'EdgeColor', 'g'                                  ...
         , 'FaceColor', [ 0.95 1 0.95 ]                      ...
         , 'LineWidth', 2                                    ...
         );

    plot ( 0,           z_a,         'ob', 'MarkerSize', 8, 'MarkerFaceColor', 'b' );
    plot ( qc.qsa_l(1), qc.qsa_l(2), 'ob', 'MarkerSize', 4, 'MarkerFaceColor', 'b' );
    plot ( qc.qsa_r(1), qc.qsa_r(2), 'ob', 'MarkerSize', 4, 'MarkerFaceColor', 'b' );
    plot ( qc.qva(1),   qc.qva(2),   'ob', 'MarkerSize', 4, 'MarkerFaceColor', 'b' );
    
    plot ( 0,           z_s,         'og', 'MarkerSize', 8, 'MarkerFaceColor', 'g' );
    plot ( qc.qss_l(1), qc.qss_l(2), 'og', 'MarkerSize', 4, 'MarkerFaceColor', 'g' );
    plot ( qc.qss_r(1), qc.qss_r(2), 'og', 'MarkerSize', 4, 'MarkerFaceColor', 'g' );
    plot ( qc.qvs(1),   qc.qvs(2),   'og', 'MarkerSize', 4, 'MarkerFaceColor', 'g' );
    
    plot ( ...
           qc.qss_l(1) + fact.fs_l(1)*[0.2 1]*fscal  ...
         , qc.qss_l(2) + fact.fs_l(2)*[0.2 1]*fscal  ...
         , 'Color',    'r'                           ...
         , 'LineWidth', 4                            ...
         );
    plot ( ...
           qc.qss_r(1) + fact.fs_r(1)*[0.2 1]*fscal  ...
         , qc.qss_r(2) + fact.fs_r(2)*[0.2 1]*fscal  ...
         , 'Color',    'r'                           ...
         , 'LineWidth', 4                            ...
         );
    
    hold off;
    
    % -> scaling etc.
    axis ( [ ymin ymax zmin zmax ] );
    set ( gcf, 'Name', sprintf ( 'Time  t = %8.3f', t(it) ) );
    set ( gca, 'PlotBoxAspectRatio', [ 1 1 1 ] );
    set ( gca, 'DataAspectRatio',    [ 1 1 1 ] );
    drawnow,
    
end;
