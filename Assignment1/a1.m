% Question 1.(c) solution


m = ones(100, 200);
m(10, 20) = 2;
m(20, 40) = 2;
m(30, 60) = 2;
m(40, 80) = 2;
m(50, 100) = 2; 
m(60, 80) = 2;
m(70, 60) = 2;
m(80, 40) = 2;
m(90, 20) = 2;
% surf(m)


% Question 2.(c) solution

F1 = [10, 40, 8; 5, 3, 5; 12, 5, 12];
F2 = [6, 3, 6; 2, 1, 2; 6, 3, 6];

[u1, s1, v1] = svd(F1);
[u2, s2, v2] = svd(F2);

disp(s1)
disp(s2)

v2_trans = (v2)';
vertical_filter = sqrt(s2(1,1))*u2(:,1);
horizontal_filter = sqrt(s2(1,1))*(v2_trans(:,1));

disp(vertical_filter)
disp(horizontal_filter)

% COMMENT: F1 is not separable, but F2 is separable. 
% The vertical filter is [-2.4888, -0.8296, -2.4888]' 
% and the horizontal filter is [-2.4108, 2.6953, 0]'.



