function jacobian = jac ( y0, freq );

f0 = awert ( y0, freq );

for i=1:4,
    y = y0;
    delta = sqrt(eps) * max ( abs(y0(i)), sqrt(sqrt(eps)) );
    y(i) = y(i) + delta;
    f = awert ( y, freq );
    jacobian(:,i) = (f-f0)/delta;
end;