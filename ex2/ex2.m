%% a. prove exp(-x) >= (1-x)
%plot it
x = linspace(-5,5,100);
y1 = exp(-x);
y2 = 1-x;
plot(x,y1)
hold on
plot(x,y2);
legend('e^-^x','1-x');
title('e^-^x VS 1-x ')
hold off