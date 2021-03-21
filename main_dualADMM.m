clc
clear all
close all
warning off
%% graph settings
N=4;
%  G=gen_graph(N);
%   G=[0     1     0     0     1;
%      1     0     1     0     0;
%      0     1     0     1     0;
%      0     0     1     0     1;
%      1     0     0     1     0];%circle
G=[0 1 0 1;1 0 1 0 ;0 1 0 1;1 0 1 0];%N=4 circle
%  G_r=G(:);
G_=triu(G);
 index=find(G_(:));
for i=1:N
   n(:,i)=sum(G(:,i));% number of neighbor
end
%% sample distributed
d=100;
load('data.mat');
% chol(A);
% chol(B);
sumA=A/100*N;
sumB=B/100*N;
clear A B
%% ground truth
[V,D]=eig(sumA,sumB);%AV=BVD norm(sumA*V-sumB*V*D)
global VV
VV=V(:,1)/sqrt(norm(V(:,1)'*sumB*V(:,1)));% constraint norm(VV(:,1)'*sumA*VV(:,1)),norm(VV(:,1)'*sumB*VV(:,1))
F_true=-norm(VV(:,1)'*sumA*VV(:,1));
% VV=V/sqrt(V(1,:)*sumB*V(1,:)');% constraint norm(VV(:,1)'*sumA*VV(:,1)),norm(VV(:,1)'*sumB*VV(:,1))
% F_true=-norm(VV(1,:)*sumA*VV(1,:)');
%% parameter initialization
rho1=100;
rho2=100;

% w_init=randn(d,1);
% w_init=w_init/sqrt(w_init'*sumB*w_init);%must
% load('wronginit.mat');
l_init=zeros(d,1);
% z_init=w_init+l_init/rho2;
%% local data preparation
w_b=0;
for i=1:N
    A(:,:,i)=sumA/N;
    B(:,:,i)=sumB/N;
    l(:,:,i)=l_init;
    w(:,:,i)=randn(d,1);
    w(:,:,i)=w(:,:,i)/sqrt(w(:,:,i)'*sumB*w(:,:,i));%must
    w_b=w_b+w(:,:,i);
    c(:,i)=w(:,:,i)'*B(:,:,i)*w(:,:,i);
end
z_init=w_b/N+l_init/rho2;
z=z_init;
w_b=w_b/N;
% w_b=w_init;
fprintf('init_outerL: %0.5f\n',outerL(N,w,A,l,z,rho2));
iter=0;
% ii=[1 2  3  4];
% jj=[2  3  4 1 ];
%% outer ADMM
while 1
    %% w_i update: select edge and inner loop
    w_b_old=w_b;
    iter=iter+1;
    flag=zeros(N,1);
    while 1
         r=randperm(size(index,1));
         s_rp=r(1:size(r,2)/2);%
         for k=1:size(s_rp,2)%3N
             ii=floor((index(s_rp(k))-1)/N)+1;
             jj=index(s_rp(k))-(ii-1)*N;
             sumc=cal_globalc(c,N);%(k)
             fprintf('!! Node %d and Node %d are updating!\n',ii,jj);
             [w(:,:,ii),w(:,:,jj),c(:,ii),c(:,jj)]=inner_loop(ii,jj,A,B,rho1,d,sumc,w(:,:,ii),w(:,:,jj));
%              fprintf('!! Node %d and Node %d are updating!\n',ii(k),jj(k));
%              [w(:,:,ii(k)),w(:,:,jj(k)),c(:,ii(k)),c(:,jj(k))]=inner_loop(ii(k),jj(k),A,B,rho1,d,sumc,w(:,:,ii(k)),w(:,:,jj(k)));
             flag(ii)=1;
             flag(jj)=1;
         end
         if isempty(find(~flag))
             break;
         end
    end
    fprintf('outerL after w_i: %0.5f\n',outerL(N,w,A,l,z,rho2));
    %% z update in the FC
    w_m=0;
    l_m=0;
    for i=1:N
        w_m=w_m+w(:,:,i);
        l_m=l_m+l(:,:,i);
    end
    w_m=w_m/N;
    l_m=l_m/N;
    z=w_m+l_m/rho2;
    fprintf('outerL after z: %0.5f\n',outerL(N,w,A,l,z,rho2));
    %% lambda update
    for i=1:N
        l(:,:,i)=l(:,:,i)+rho2*(w(:,:,i)-z);
    end
    fprintf('outerL after l_i: %0.5f\n',outerL(N,w,A,l,z,rho2));
    %% stop criteria
    w_b=0;
    for i=1:N
        w_b=w_b+w(:,:,i);
    end
    w_b=w_b/N; %output
    res1=0;
    for i=1:N
        res1=res1+norm(w(:,:,i)-w_b);
    end
    res2=norm(w_b-w_b_old);
    if res1<1e-04 && res2<1e-04%iter>300 
            fprintf('#complete outer iter=%d, res1=%0.5f, res2=%0.5f\n',iter,res1,res2);
            fprintf('\n')
            break;
    else
            fprintf('#complete outer iter=%d, res1=%0.5f, res2=%0.5f\n',iter,res1,res2);
%             fprintf('\n')
    end
end
%%


%% functions
function  [L]=outerL(N,w,A,l,z,rho2)%,F
    L=0;
%     F=0;
    for i=1:N
        L=L-w(:,:,i)'*A(:,:,i)*w(:,:,i)+l(:,:,i)'*(w(:,:,i)-z)+rho2/2*(norm(w(:,:,i)-z)^2);
%         F=F-w(:,:,i)'*A(:,:,i)*w(:,:,i)
    end
end
function sumc=cal_globalc(c,N)
sumc=0;
for i=1:N
    sumc=sumc+c(:,i);
end
end
function L=innerL(wi,wj,Ai,Aj,Bi,Bj,zi,zj,ai,bi,bj,rho1,c)
    L=-wi'*Ai*wi-wj'*Aj*wj+ai*(c-wi'*Bi*wi-wj'*Bj*wj)+bi'*(wi-zi)+bj'*(wj-zj)+rho1/2*(norm(wi-zi)^2+norm(wj-zj)^2);
end
function p=inv_ill(A)
    [u,d,v]=svd(A);%A=udv'
    dd=size(d,1);
    for i=1:dd
        if d(i,i)==0
            s(i)=0;
        else
            s(i)=1/d(i,i);
        end
    end
    S=diag(s);
    p=v*S*u';
end

function [wi,wj,ci,cj]=inner_loop(i,j,A,B,rho1,d,sumc,wi,wj)
%% initialization
    global VV
    ai=0;
    aj=0;
    bi=zeros(d,1);
    bj=zeros(d,1);
    zi=1/2*(wi+wj);
    zj=zi;
    Ai=A(:,:,i);
    Aj=A(:,:,j);
    Bi=B(:,:,i);
    Bj=B(:,:,j);
%     cjti=wi'*Bj*wi;
    cj=wj'*Bj*wj;
    ci=wi'*Bi*wi;
    sumk=sumc-ci-cj;
    c=1-sumk;
    hj=rho1*wj+bj;
    clear A B
    iter=0;
%     fprintf('init_L: %0.5f\n',innerL(wi,wj,Ai,Aj,Bi,Bj,zi,zj,ai,bi,bj,rho1,c));
    while 1
        iter=iter+1;
        wi_old=wi;
        wj_old=wj;
        %% node i,receive cjti hj
%         wi=inv(2*(-ai*Bi-Ai+rho1/2*eye(d)))*(rho1*zi-bi);     
        wi=inv_ill(2*(-ai*Bi-Ai+rho1/2*eye(d)))*(rho1*zi-bi);   %pinv
%         citj= wi'*Bi*wi;        
%         wi=wi/sqrt(citj);
%         wi=wi/norm(wi);
%         fprintf('L after wi: %0.5f\n',innerL(wi,wj,Ai,Aj,Bi,Bj,zi,zj,ai,bi,bj,rho1,c));
        bi=bi+rho1*(wi-zi);
%         fprintf('L after bi: %0.5f\n',innerL(wi,wj,Ai,Aj,Bi,Bj,zi,zj,ai,bi,bj,rho1,c));
        hi=rho1*wi+bi;
        
%         zi=(hi+hj)/(2*rho1);
%         fprintf('L after zi: %0.5f\n',innerL(wi,wj,Ai,Aj,Bi,Bj,zi,zj,ai,bi,bj,rho1,c));   
        ci= wi'*Bi*wi;  
        ai=ai+rho1*(c-ci);%-cj
%         fprintf('L after ai: %0.5f\n',innerL(wi,wj,Ai,Aj,Bi,Bj,zi,zj,ai,bi,bj,rho1,c));
        %% node j, receive citj,hi
        aj=ai;
        wj=inv_ill(2*(-aj*Bj-Aj+rho1/2*eye(d)))*(rho1*zj-bj);   
%         cjti= wj'*Bj*wj;
%         wj=wj/norm(cjti);
%         wj=wj/norm(wj);
%         fprintf('L after wj: %0.5f\n',innerL(wi,wj,Ai,Aj,Bi,Bj,zi,zj,aj,bi,bj,rho1,c));
        bj=bj+rho1*(wj-zj);
%         fprintf('L after bj: %0.5f\n',innerL(wi,wj,Ai,Aj,Bi,Bj,zi,zj,aj,bi,bj,rho1,c));
        hj=rho1*wj+bj;
        zj=(hi+hj)/(2*rho1);
%         fprintf('L after zj: %0.5f\n',innerL(wi,wj,Ai,Aj,Bi,Bj,zi,zj,aj,bi,bj,rho1,c));        
        cj= wj'*Bj*wj;
%         aj=aj+rho1*(c-cj-citj);
%         aj=aj+rho1*(c-ci-cj);
        aj=aj+rho1*(-cj);
%         fprintf('L after aj: %0.5f\n',innerL(wi,wj,Ai,Aj,Bi,Bj,zi,zj,aj,bi,bj,rho1,c));
        ai=aj;
        zi=zj;
        res1=norm(wi_old-wi);
        res2=norm(wj_old-wj);
        %% stop criteria
        if res1<1e-05 && res2<1e-05%iter>300 
            fprintf('#complete inner iter=%d, res_wi=%0.5f, res_wj=%0.5f\n',iter,res1,res2);
            break;
        else
%             fprintf('#complete inner iter=%d, res_wi=%0.5f, res_wj=%0.5f\n',iter,res1,res2);
%             fprintf('\n')
        end
    end
end



