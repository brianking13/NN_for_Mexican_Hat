clear;clc;

[M, N] = meshgrid(0:1:10,0:1:10);
x(1,:) = reshape(M,1,[]);
x(2,:) = reshape(N,1,[]);
a = 1;
b = 0;
Z = (0.0431/0.3614)+(1/0.3614)*(1/pi).*(1-((M-5).^2 + (N-5).^2)/2).*exp(-((M-5).^2 + (N-5).^2)/2);
output = reshape(Z,1,[]);

figure(1)
surf(M,N,Z)
hold on
title('Mexican Hat')
xlabel('x')
ylabel('y')
zlabel('z')


% Neural Network Architecture Variables
ins = 2; % Number of input nodes
hids = 10; % Number of hidden nodes
outs = 1; % Number of output nodes
examples = length(x(1,:)); % Number of examples

% Learning Parameters
kappa = 0.1;
phi = 0.5;
theta = 0.7;
mu = .7;

% Weights and Derivatives
a(1:(ins+1), 1:hids) = 0.0; % Hidden weights
b(1:(hids+1), 1:outs) = 0.0; % Output weights
cHid(1:(ins+1), 1:hids) = 0.0; % Weight changes
cOut(1:(hids+ins+1), 1:outs) = 0.0;
dHid(1:(ins+1), 1:hids) = 0.0; % Derivatives
dOut(1:(hids+ins+1), 1:outs) = 0.0;
eHid(1:(ins+1), 1:hids) = 0.0; % Adaptive learning rates
eOut(1:(hids+ins+1), 1:outs) = 0.0;
fHid(1:(ins+1), 1:hids) = 0.0; % Recent average of derivatives
fOut(1:(hids+ins+1), 1:outs) = 0.0;
u = 0.0; % Weighted sum for hidden node 
y(1:hids) = 0.0; % Hidden node outputs
v = 0.0; % Weighted sum for output node
z(1:outs) = 0.0; % Output node outputs
p(1:outs) = 0.0; % dE/dv
epoch = 0; % Current epoch number


%initWeights
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for j = 1:hids,
    for i = 1:(ins+1),
        a(i,j) = 0.2 * (rand - 0.5);
        eHid(i,j) = kappa;
    end % i
end % j
for k = 1:outs,
    for j = 1:(hids+1),
        if mod(j,2) == 0
            b(j,k) = 1;
        else
            b(j,k) = -1;
        end % if
    eOut(j,k) = kappa;
   end % j
end % k


% The main program
while (epoch < 10000)
    epoch = epoch + 1;
    error_sum(epoch) = 0;
    for n = 1:examples % Cycling through examples for epoch-based update



         %Forward Evaluation %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        for j = 1:hids,
            u = a(ins+1,j); % Bias weight
            for i = 1:ins,
                u = u + (a(i,j) * x(i,n)); % Weighted inputs
            end % i
            y(j) = logistic(u);
        end % j
        for k = 1:outs,
            v = b(hids+1,k); % Bias weight
            for j = 1:hids,
                v = v + (b(j,k) * y(j)); % Hidden outputs, wtd.
            end % j
            z(k) = logistic(v);
           
         
            
        end % k
        s(1,n) = z; % For recording results
        error_sum(epoch) = error_sum(epoch) + abs(z(1)-output(1,n));% Sum of errors



        % Backpropagation %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        q(1:hids) = 0.0; % dE/du (reset to 0 each time)
        for k = 1:outs,% Calculate error derivatives on output node weights
            p(k) = (z(k) - output(k,n)) * z(k) * (1 - z(k));
            dOut(hids+1,k) = dOut(hids+1,k) + p(k); % Bias weight
            for j = 1:hids, % Weights on hids
                dOut(j,k) = dOut(j,k) + p(k) * y(j);
                q(j) = q(j) + p(k) * b(j,k); % Used below
            end % j
        end % k
        for j = 1:hids,
            q(j) = q(j) * y(j) * (1 - y(j));
            dHid(ins+1,j) = dHid(ins+1,j) + q(j); % Bias weight
            for i = 1:ins, % Weights on ins
                dHid(i,j) = dHid(i,j) + q(j) * x(i,n);
            end % i
        end % j


    end % n - examples


   % Change Weights %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for j = 1:hids, % Change weights on hidden nodes
        for i = 1:(ins+1),
            if dHid(i,j)*fHid(i,j) > 0.0 
                eHid(i,j) = eHid(i,j) + kappa;
            else
                eHid(i,j) = eHid(i,j) * phi;
            end
            fHid(i,j) = theta * fHid(i,j) + (1 - theta) * dHid(i,j);
            cHid(i,j) = mu *cHid(i,j) - (1 - mu) * eHid(i,j) * dHid(i,j);
            a(i,j) = a(i,j) + cHid(i,j);
        end % i
    end % j
    for k = 1:outs, % Change weights on output nodes
        for j = 1:(hids+1),
            if dOut(j,k)*fOut(j,k) > 0.0 
                eOut(j,k) = eOut(j,k) + kappa;
            else
                eOut(j,k) = eOut(j,k) * phi;
            end
            fOut(j,k) = theta * fOut(j,k) + (1 - theta) * dOut(j,k);
            cOut(j,k) = mu *cOut(j,k) - (1 - mu) * eOut(j,k) * dOut(j,k);
            b(j,k) = b(j,k) + cOut(j,k);
        end % j
    end % k
    dHid(1:(ins+1), 1:hids) = 0; % Reset to 0
    dOut(1:(hids+1), 1:outs) = 0; % Reset to 0

    dHid(1:(ins+1), 1:hids) = 0; % Reset to 0
    dOut(1:(hids+1), 1:outs) = 0; % Reset to 0
end % while



output3d = reshape(s,length(M),length(N));
plot3(M,N,output3d,'ro','MarkerSize',10)


figure(2)
plot(error_sum);
xlabel('Epoch'); ylabel('Error');

