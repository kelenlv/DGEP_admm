clc
clear all
% close all
warning off
%% graph settings
N=5;
sgy = 'diverse-F';% 'diverse-z''random'
randn('seed',1)
%  G=gen_graph(N);
  G=[0     1     0     0     1;
     1     0     1     0     0;
     0     1     0     1     0;
     0     0     1     0     1;
     1     0     0     1     0];%circle
% G=[0 1 0 1;1 0 1 0 ;0 1 0 1;1 0 1 0];%N=4 circle
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
rho1=15;%外50 >10
rho2=10;%内100

w_init=randn(d,1);
w_init=w_init/sqrt(w_init'*sumB*w_init);%must
% load('wronginit.mat');
l_init=zeros(d,1);
% z_init=w_init+l_init/rho2;
E_list=[];
L_list=[];
%% local data preparation
w_b=0;
for i=1:N
    A(:,:,i)=sumA/N;
    B(:,:,i)=sumB/N;
    l(:,:,i)=l_init;
    w(:,:,i)=w_init;%randn(d,1);
    w_b=w_b+w(:,:,i);
    c(:,i)=w(:,:,i)'*B(:,:,i)*w(:,:,i);
end
if w_b'*VV<0
    w_b=-w_b;
end
z_init=w_b/N+l_init/rho2;
z=z_init;
w_b=w_b/N;
% w_b=w_init;
fprintf('init_outerL: %0.5f\n',outerL(N,w,A,l,z,rho2));
iter=0;
%% outer ADMM
while 1
    %% w_i update: select edge and inner loop
    w_b_old=w_b;
    iter=iter+1;
    flag=zeros(N,1);
    coun=0;
    %% w-update
    while 1
         if strcmp(sgy ,'diverse-z')%new added
             nor=[];
             for ss = 1:N
                nor(ss)=norm(w(:,:,ss)-z,2);
             end
%              nor
             ii=find(nor==max(nor));
             ii=ii(1);
             jj=find(nor==min(nor));
             jj=jj(1);
             sumc=cal_globalc(c,N);%(k)
             if ii ~= jj
                fprintf('!! Node %d and Node %d are updating!\n',ii,jj);
                [w(:,:,ii),w(:,:,jj),c(:,ii),c(:,jj)]=inner_loop(ii,jj,A,B,rho1,d,sumc,w(:,:,ii),w(:,:,jj),l(:,:,ii),l(:,:,jj),z,rho2);
                coun = coun+1;
                 if coun >10
                     break;
                 end
             else 
                 jj=jj+1;
                 fprintf('!! Node %d and Node %d are updating!\n',ii,jj);
                 [w(:,:,ii),w(:,:,jj),c(:,ii),c(:,jj)]=inner_loop(ii,jj,A,B,rho1,d,sumc,w(:,:,ii),w(:,:,jj),l(:,:,ii),l(:,:,jj),z,rho2);

             end
             flag(ii)=1;
             flag(jj)=1;
             [w(:,:,ii),w(:,:,jj)]=check_allign(VV,w(:,:,ii),w(:,:,jj));
         elseif strcmp(sgy ,'diverse-F') %new added
             F_choose=[];
             for ss = 1:N
                F_choose(ss) = w(:,:,ss)'*A(:,:,ss)*w(:,:,ss);
             end
             F_choose
             ii=find(F_choose==max(F_choose));
             ii=ii(1);
             jj=find(F_choose==min(F_choose));
             jj=jj(1);
             sumc=cal_globalc(c,N);%(k)
             if ii ~= jj
                fprintf('!! Node %d and Node %d are updating!\n',ii,jj);
                [w(:,:,ii),w(:,:,jj),c(:,ii),c(:,jj)]=inner_loop(ii,jj,A,B,rho1,d,sumc,w(:,:,ii),w(:,:,jj),l(:,:,ii),l(:,:,jj),z,rho2);
                coun = coun+1;
                 if coun >10
                     break;
                 end
             else 
                 jj=jj+1;
                 fprintf('!! Node %d and Node %d are updating!\n',ii,jj);
                 [w(:,:,ii),w(:,:,jj),c(:,ii),c(:,jj)]=inner_loop(ii,jj,A,B,rho1,d,sumc,w(:,:,ii),w(:,:,jj),l(:,:,ii),l(:,:,jj),z,rho2);
             end
             flag(ii)=1;
             flag(jj)=1;
             [w(:,:,ii),w(:,:,jj)]=check_allign(VV,w(:,:,ii),w(:,:,jj));
         elseif strcmp(sgy ,'random')
             r=randperm(size(index,1));
             s_rp=r(1:size(r,2)/2);%    
    %          resz=0;
    %          zm=0;
             for k=1:size(s_rp,2)%3N
                 ii=floor((index(s_rp(k))-1)/N)+1;
                 jj=index(s_rp(k))-(ii-1)*N;
                 sumc=cal_globalc(c,N);%(k)
                 fprintf('!! Node %d and Node %d are updating!\n',ii,jj);
    %              [w(:,:,ii),w(:,:,jj),c(:,ii),c(:,jj),zz(:,ii),zz(:,jj)]=inner_loop(ii,jj,A,B,rho1,d,sumc,w(:,:,ii),w(:,:,jj));
                 [w(:,:,ii),w(:,:,jj),c(:,ii),c(:,jj)]=inner_loop(ii,jj,A,B,rho1,d,sumc,w(:,:,ii),w(:,:,jj),l(:,:,ii),l(:,:,jj),z,rho2);
    %              fprintf('!! Node %d and Node %d are updating!\n',ii(k),jj(k));
    %              [w(:,:,ii(k)),w(:,:,jj(k)),c(:,ii(k)),c(:,jj(k))]=inner_loop(ii(k),jj(k),A,B,rho1,d,sumc,w(:,:,ii(k)),w(:,:,jj(k)));
                 flag(ii)=1;
                 flag(jj)=1;
                 [w(:,:,ii),w(:,:,jj)]=check_allign(VV,w(:,:,ii),w(:,:,jj));
             end
         end
         if isempty(find(~flag))%&& resz<1e-3
                break;
%          else
%                  flag=zeros(N,1);
         end
    end
    L=outerL(N,w,A,l,z,rho1);
    fprintf('outerL after w_i: %0.5f\n',L);
    L_list=[L_list L];
     %% z update in the FC
    w_m=0;
    l_m=0;
    for i=1:N
        w_m=w_m+w(:,:,i);
        l_m=l_m+l(:,:,i);
    end
    w_m=w_m/N;
    l_m=l_m/N;
    z=w_m+l_m/rho1;
    fprintf('outerL after z: %0.5f\n',outerL(N,w,A,l,z,rho1));
    %% lambda update
    for i=1:N
        temp=l(:,:,i)+rho2*(w(:,:,i)-z);
        if iter==1||norm(temp)>1e-3
            l(:,:,i)=temp;
        end
    end
    fprintf('outerL after l_i: %0.5f\n',outerL(N,w,A,l,z,rho1));
    %% stop criteria
    w_b=0;
    for i=1:N
        w_b=w_b+w(:,:,i);
    end
    w_b=w_b/N; %output
    res1=0; %
    for i=1:N
        res1=res1+norm(w(:,:,i)-w_b);
    end
    res2=norm(w_b-w_b_old);%z-residual
    if  res1<1e-04 &&res2<1e-02%iter>300 
            fprintf('#complete outer iter=%d, res1=%0.5f, res2=%0.5f\n',iter,res1,res2);% 
            sin(subspace(VV,w_b))
            fprintf('\n')
            break;
    else
            fprintf('#complete outer iter=%d, res1=%0.5f, res2=%0.5f\n',iter,res1,res2);% 
%             fprintf('\n')
    end
    if iter>500
        break;
    end
    sin(subspace(VV,w_b))
    E_list=[E_list norm(VV-w_b)];
   
end
%%
figure; yyaxis left; 
plot(E_list,'LineWidth',2);
% title('Convergence performance of Alg.1','interpreter','latex', 'FontSize', 18);
xlabel('iterations','interpreter','latex', 'FontSize', 18);
ylabel('distance of subspaces','interpreter','latex', 'FontSize', 18); 
yyaxis right; 
plot(L_list,'LineWidth',2);
ylabel('The Lagrangian fuction value','interpreter','latex', 'FontSize', 18); 

%% functions
function [wi,wj]=check_allign(w_b,wi,wj)
    if w_b'*wi<0
        wi=-wi;
    end
    if w_b'*wj<0
        wj=-wj;
    end
end
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
function L=innerLGD(wi,wj,Ai,Aj,Bi,Bj,ai,li,lj,rho1,c,z,rho2)
    L=-wi'*Ai*wi-wj'*Aj*wj+ai*(wi'*Bi*wi+wj'*Bj*wj-c)+li'*(wi-z)+lj'*(wj-z)+rho1/2*(norm(wi-z)^2+norm(wj-z)^2)+rho2/2*norm(wi'*Bi*wi+wj'*Bj*wj-c)^2;
end
% function L=innerL(wi,wj,Ai,Aj,Bi,Bj,ai,li,lj,rho1,c,z)
%     L=-wi'*Ai*wi-wj'*Aj*wj+ai*(wi'*Bi*wi+wj'*Bj*wj-c)+li'*(wi-z)+lj'*(wj-z)+rho1/2*(norm(wi-z)^2+norm(wj-z)^2);
% end
function p=inv_ill(A)
    [u,d,v]=svd(A);%A=udv'
    dd=size(d,1);
    for i=1:dd
        if d(i,i)<=1e-3
            s(i)=0;
        else
            s(i)=1/d(i,i);
        end
    end
    S=diag(s);
    p=v*S*u';
end
function [wi,wj,ci,cj]=inner_loop(i,j,A,B,rho1,d,sumc,wi,wj,li,lj,z,rho2)
%% initialization
    global VV
    a=0;
    Ai=A(:,:,i);
    Aj=A(:,:,j);
    Bi=B(:,:,i);
    Bj=B(:,:,j);
    cj=wj'*Bj*wj;
    ci=wi'*Bi*wi;
    sumk=sumc-ci-cj;
    c=1-sumk; %c=ci+cj
    clear A B
    iter=0;
%     a_old=a;
%     fprintf('init_L: %0.5f\n',innerLGD(wi,wj,Ai,Aj,Bi,Bj,a,li,lj,rho1,c,z,rho2));
%     fprintf('init_L: %0.5f\n',innerL(wi,wj,Ai,Aj,Bi,Bj,a,li,lj,rho1,c,z));

    while 1
        iter=iter+1;
        wi_old=wi;
        wj_old=wj;
%         cj+ci-c
%         flag=0;
%         wi=inv_ill(2*(a*Bi-Ai+rho1/2*eye(d)))*(rho1*z-li);   %pinv
        wi=w_GD(wi,Ai,Bi,rho1,rho2,z,a,li,cj,c);
%         fprintf('L after wi: %0.5f\n',innerLGD(wi,wj,Ai,Aj,Bi,Bj,a,li,lj,rho1,c,z,rho2));   
%         fprintf('L after wi: %0.5f\n',innerL(wi,wj,Ai,Aj,Bi,Bj,a,li,lj,rho1,c,z)); 
        ci= wi'*Bi*wi;     
        wj=w_GD(wj,Ai,Bi,rho1,rho2,z,a,lj,ci,c);
%         wj=inv_ill(2*(a*Bj-Aj+rho1/2*eye(d)))*(rho1*z-lj);   
%         fprintf('L after wj: %0.5f\n',innerLGD(wi,wj,Ai,Aj,Bi,Bj,a,li,lj,rho1,c,z,rho2));  
%         fprintf('L after wj: %0.5f\n',innerL(wi,wj,Ai,Aj,Bi,Bj,a,li,lj,rho1,c,z));   
        cj= wj'*Bj*wj; 
%         aj=aj+rho1*(ci+cj-c);
        if norm(wi_old-wi)<1e-3&&norm(wj_old-wj)<1e-3
                a=a+rho1*(cj+ci-c);%
%                 fprintf('L after a: %0.5f\n',innerL(wi,wj,Ai,Aj,Bi,Bj,a,li,lj,rho1,c,z));
%                 fprintf('L after a: %0.5f\n',innerLGD(wi,wj,Ai,Aj,Bi,Bj,a,li,lj,rho1,c,z,rho2));
%                 flag=1;
        end
%         if iter==1||temp>1e-3
%              a=temp;
%         else
%             a=0;
%         end   
%         fprintf('L after a: %0.5f\n',innerL(wi,wj,Ai,Aj,Bi,Bj,a,li,lj,rho1,c,z,rho1)); 
        res1=norm(wi_old-wi);
        res2=norm(wj_old-wj);
        %% stop criteria
        if res1<1e-03 && res2<1e-03 %(ci+cj-c)<1e-3&&norm(a_old-a)<1e-3&&flag==1
%             fprintf('#complete inner iter=%d, res_wi=%0.5f, res_wj=%0.5f\n',iter,res1,res2);
            break;
        else
%             fprintf('#complete inner iter=%d, res_wi=%0.5f, res_wj=%0.5f\n',iter,res1,res2);
%             fprintf('\n')
        end
%         a_old=a;      
    end
end
function [w]=w_GD(w,A,B,rho1,rho2,z,a,l,ci,c)
    iter=0;
    r=0.001;
%     w=w/norm(w);
    L_old=wL_func(w,A,B,a,l,z,rho1,rho2,ci,c);
    while 1
        iter=iter+1;
        g=2*(a*B-A)*w+l+rho1*w-rho1*z+2*rho2*w'*B*w*B*w+2*rho2*(ci-c)*B*w;
        g=g/norm(g);
        w=w-r*g;
         L=wL_func(w,A,B,a,l,z,rho1,rho2,ci,c);
%         if  norm(w_old-w)<1e-3
%             break;
%         end
        if norm(L_old-L)<1e-2
%             w=w/norm(w);
            break;
        else
%             fprintf('#iter=%d,norm_gradient:%f\n',iter,norm(g))
        end
        L_old=L;
    end
end
function [L]=wL_func(wi,Ai,Bi,ai,li,z,rho1,rho2,cj,c)
    L=-wi'*Ai*wi+ai*(wi'*Bi*wi+cj-c)+li'*wi+rho1/2*wi'*wi-rho1*wi'*z+rho2/2*norm(wi'*Bi*wi+cj-c)^2;
end

