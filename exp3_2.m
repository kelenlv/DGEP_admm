clc
clear all
%% FDA
N=5;
d=100;
% [A,B]=load_phishing(N );
% load('phishingdata.mat');
% load_wine(N);
% load('winedata_full.mat');
[sumA,sumB,X_class1,X_class2,RandSeed] = FDAproblem_2class(d,666);
  G=[0     1     0     0     1;
     1     0     1     0     0;
     0     1     0     1     0;
     0     0     1     0     1;
     1     0     0     1     0];%circle
G_=triu(G);
 index=find(G_(:));
for i=1:N
   n(:,i)=sum(G(:,i));% number of neighbor
end
%% sample distributed
% d=12;
% ground truth
% sumA=0;
% sumB=0;
% for i=1:N
% %     A(:,:,i)=A(:,:,i)+eye(d);
%     sumA=sumA+A(:,:,i);
% %     B(:,:,i)=B(:,:,i)+eye(d);
%     sumB=sumB+B(:,:,i);
% end
for i=1:N
    A(:,:,i)=sumA/N;
    B(:,:,i)=sumB/N;
end
[V,D]=eig(sumA,sumB);%AV=BVD norm(sumA*V-sumB*V*D)
global VV
VV=V(:,1)/sqrt(norm(V(:,1)'*sumB*V(:,1)));% constraint norm(VV(:,1)'*sumA*VV(:,1)),norm(VV(:,1)'*sumB*VV(:,1))
F_true=-norm(VV(:,1)'*sumA*VV(:,1));
% correct_label(2000,w_test,r_test,VV)
corr=correct_label(500,X_class1,X_class2,VV)/1000 
% VV=V/sqrt(V(1,:)*sumB*V(1,:)');% constraint norm(VV(:,1)'*sumA*VV(:,1)),norm(VV(:,1)'*sumB*VV(:,1))
% F_true=-norm(VV(1,:)*sumA*VV(1,:)');
%% parameter initialization
rho1=100;%Íâ1000
rho2=100;%ÄÚ2000
E_list=[];
L_list=[];
Corr_list=[];
w_init=randn(d,1);
w_init=w_init/sqrt(w_init'*sumB*w_init);%must
% load('wronginit.mat');
l_init=zeros(d,1);
% z_init=w_init+l_init/rho2;
%% local data preparation
w_b=0;
for i=1:N
%     A(:,:,i)=sumA/N;
%     B(:,:,i)=sumB/N;
    l(:,:,i)=l_init;
    w(:,:,i)=w_init;%randn(d,1);
%     w(:,:,i)=w(:,:,i)/sqrt(w(:,:,i)'*sumB*w(:,:,i));%must
    w_b=w_b+w(:,:,i);
    c(:,i)=w(:,:,i)'*B(:,:,i)*w(:,:,i);
end
% if w_b'*VV<0
%     w_b=-w_b;
% end
w_b=w_b/N;

iter=0;
%% outer ADMM
while 1
    %% w_i update: select edge and inner loop
    w_b_old=w_b;
    iter=iter+1;
    flag=zeros(N,1);
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
    %% w-update
    while 1
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
         if isempty(find(~flag))%&& resz<1e-3
                break;
%          else
%                  flag=zeros(N,1);
         end
    end
%     fprintf('outerL after w_i: %0.5f\n',outerL(N,w,A,l,z,rho2));
    L=outerL(N,w,A,l,z,rho1);
    fprintf('outerL after w_i: %0.5f\n',L);
    L_list=[L_list L];
    %% lambda update
%     l(:,:,i)=l(:,:,i)+rho1*(w(:,:,i)-z);
    for i=1:N
        temp=l(:,:,i)+rho1*(w(:,:,i)-z);
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
    res1=0;
    for i=1:N
        res1=res1+norm(w(:,:,i)-w_b);
    end
    res2=norm(w_b-w_b_old);
    if  res1<1e-04 &&res2<1e-02%iter>300 
            fprintf('#complete outer iter=%d, res1=%0.5f, res2=%0.5f\n',iter,res1,res2);% 
            sin(subspace(VV,w_b))
            fprintf('\n')
            break;
    else
            fprintf('#complete outer iter=%d, res1=%0.5f, res2=%0.5f\n',iter,res1,res2);% 
%             fprintf('\n')
    end
    if  sin(subspace(VV,w_b))<1e-03
        break;
    end
%     norm(VV-w_b)
    sin(subspace(VV,w_b))
    E_list=[E_list sin(subspace(VV,w_b))];
    Corr_list=[Corr_list correct_label(500,X_class1,X_class2,w_b)/1000];
%     
end
%%
figure; yyaxis left; 
plot(E_list,'LineWidth',1);
title('Convergence performance of Alg.1','interpreter','latex', 'FontSize', 18);
xlabel('iterations','interpreter','latex', 'FontSize', 18);
ylabel('distance of subspaces','interpreter','latex', 'FontSize', 18); 
yyaxis right; 
plot(L_list,'LineWidth',1);
ylabel('The Lagrangian fuction value'); 

figure;
plot(Corr_list,'LineWidth',1);
title('Accuracy performance of Alg.1','interpreter','latex', 'FontSize', 18);
xlabel('iterations','interpreter','latex', 'FontSize', 18);
ylabel('Accuracy of classification ','interpreter','latex', 'FontSize', 18); 

function [correct]=correct_label(n,X_class1,X_class2,w_b)
    y=0;
    correct=0;
    for i=1:n
        y=y+X_class1(i,:)*w_b - X_class2(i,:)*w_b;
    end
    if y<0
        for i=1:n
              if X_class1(i,:)*w_b+0.5*y/n<0%
                  correct=correct+1;
              end
        end
        for i=1:n
              if X_class2(i,:)*w_b+0.5*y/n>0%
                  correct=correct+1;
              end
        end
     else if y>0
                for i=1:n
                      if X_class1(i,:)*w_b+0.5*y/n>0%
                          correct=correct+1;
                      end
                end
                for i=1:n
                      if X_class2(i,:)*w_b+0.5*y/n<0%
                          correct=correct+1;
                      end
                end
            end
    end
end
% tr_label = [ones(2898,1); -1*ones(2898,1)];
% train= [w_train;r_train];
% ts_label= [ones(2000,1); -1*ones(2000,1)];
% test=[w_test;r_test];
% projed_train_data=train*w_b;
% projed_test_data= test*w_b;
% svmModel = fitcsvm(projed_train_data, tr_label);
% [test_pre,~] = predict(svmModel, projed_test_data);
% (4000-sum(ts_label==test_pre))/4000
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
                a=a+rho2*(cj+ci-c);%
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
            fprintf('#complete inner iter=%d, res_wi=%0.5f, res_wj=%0.5f\n',iter,res1,res2);
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
        r=r/(iter^2);
    end
end
function [L]=wL_func(wi,Ai,Bi,ai,li,z,rho1,rho2,cj,c)
    L=-wi'*Ai*wi+ai*(wi'*Bi*wi+cj-c)+li'*wi+rho1/2*wi'*wi-rho1*wi'*z+rho2/2*norm(wi'*Bi*wi+cj-c)^2;
end
function [A,B]=load_phishing(N )

    da= csvread('C:\Users\Kelen\Downloads\Phishing-Dataset-master\dataset_small.csv');
    d_test=da(1:20000,:);
    d_train=da(20000+1:end,:);
    p=floor(size(d_train,1)/N);
    for i=1:N
        X(:,:,i)=d_train(p*(i-1)+1:p*i,:);
    end
    for i=1:N
        A_list=[];
        B_list=[];
        temp=size(X(:,:,i),1);
        for j=1:temp
            if X(j,end,i)==0
                A_list=[A_list ;X(j,1:end-1,i)];
            else
                B_list=[B_list ;X(j,1:end-1,i)];
            end
        end
        [A(:,:,i),B(:,:,i)]=prepare_FDA(A_list,B_list);
    end
end
function [A,B,w_train,r_train,w_test,r_test]=load_wine(N)
     white= csvread('C:\Users\Kelen\Downloads\white.csv');
     red=csvread('C:\Users\Kelen\Downloads\red.csv');
%      w_test=white(1:2000,:);
%      r_test=red(1:2000,:);
%      w_train=white(2000+1:end,:);
%      r_train=red(2000+1:end,:);
     w_train=white;
     r_train=red;
     p1=floor(size(w_train,1)/N);
     p2=floor(size(r_train,1)/N);
     u1=mean(w_train);
     u2=mean(r_train);
     for i=1:N
         [A(:,:,i),B(:,:,i)]=prepare_FDA(w_train(p1*(i-1)+1:p1*i,:),r_train(p2*(i-1)+1:p2*i,:));
     end
     save winedata_full.mat A B w_train r_train u1 u2%w_test r_test
end


function [Sb,Sw]=prepare_FDA(A,B)%max w'*Sb*w s.t. w'*Sw*w=1
u1=mean(A);
p1=size(A,1);
u2=mean(B);
p2=size(B,1);
Sb=(p1*u1'*u1+p2*u2'*u2)/(p1+p2);
% u=(u1+u2)/2;
% Sb=((u1-u)'*(u1-u)+(u2-u)'*(u2-u))/2;
% Sb=(u1-u2)'*(u1-u2);
S1=0;S2=0;
    for i=1:size(A,1)
        S1=S1+(A(i,:)-u1)'*(A(i,:)-u1);
    end
    for i=1:size(B,1)
        S2=S2+(B(i,:)-u2)'*(B(i,:)-u2);
    end
Sw=(S1+S2)/(p1+p2);
end