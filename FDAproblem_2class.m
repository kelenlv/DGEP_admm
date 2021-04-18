function [A,B,X_class1,X_class2,RandSeed] = FDAproblem_2class(p,RandSeed)

%% The code is associated with the paper "FIRST-ORDER ALGORITHMS FOR A CLASS OF FRACTIONAL OPTIMIZATION PROBLEM"
%% SFDA simulation

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Input: 	p: dimension of the problem, a positive integer can be devided by 5 %%%
%%%			RandSeed: random number seed 										%%%
%%%																				%%%
%%% Output: RandSeed: the same in input											%%%
%%%			A and B: matrices of the problem                                   	%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

n1 = 500;                % the number of Class 1 samples
n2 = 500;                % the number of Class 2 samples
if(nargin == 1)
	RandSeed = ceil(rand*1000);
end

% Blocks of the matrix
d_Block = p/5;
Block = zeros(d_Block);
t = 0.8.^(0:d_Block-1);
for i = 1:d_Block
    Block(i,:) = [t(i:-1:2),t(1:d_Block-i+1)];
end


% covariance matrix 
Zero = zeros(d_Block);
Sigma = [ Block,Zero,Zero,Zero,Zero;
          Zero,Block,Zero,Zero,Zero;
          Zero,Zero,Block,Zero,Zero;
          Zero,Zero,Zero,Block,Zero;
          Zero,Zero,Zero,Zero,Block];
      
% Class 1
miu1 = zeros(1,p);
[X_class1,RandSeed]= multivrandn(miu1,Sigma,n1,RandSeed);
% Class 2
miu2 = zeros(1,p);
miu2(2:2:40) = 0.5;
[X_class2,RandSeed]= multivrandn(miu2,Sigma,n2,RandSeed);

hat_miu1 = mean(X_class1);
hat_miu2 = mean(X_class2);

B = zeros(p);
for i = 1:n1
    B = B+(X_class1(i,:)-hat_miu1)'*(X_class1(i,:)-hat_miu1);
end
for i = 1:n2
    B = B+(X_class2(i,:)-hat_miu2)'*(X_class2(i,:)-hat_miu2);
end
B = B./(n1+n2);

A = (n1*hat_miu1'*hat_miu1+n2*hat_miu2'*hat_miu2)./(n1+n2);
A=A+eye(size(A));
end

function [Y,RandSeed] = multivrandn(u,R,M,RandSeed)
% this function draws M samples from N(u,R)
% where u is the mean vector(row) and R is the covariance matrix which must be positive definite

n = length(u);              % get the dimension
C = chol(R);                % perform cholesky decomp R = C'C
rng(RandSeed);
X = randn(M,n);             % draw M samples from N(0,I)

Y = X*C + ones(M,1)*u;     
RandSeed = RandSeed+1; 
end
