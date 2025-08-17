%Made with MATLAB2022a
function [genome_r, sdist, avf, histo1,histo2,avw,fsi] = recomb_2023_epi(r,s0,ac,bc,M,L,N,tf,f0,run,run2,epi,NUbs2)

% file locations 
%homedir = '~/Desktop/Recombination/figs';
% file titles for saving figures
%filename = sprintf('_%s_%g_%g_%g_%g_%g_%g_%g_%g_%g_%g',...
    %r,s0,ac,bc, M,L,N,tf,f0,run);

% Output used in external programs
global ts Nlineage T tcoal Ksample tsec1 tsec2 tsec3 mu robust

%% Arguments:
% distribution_s is uniform distribuition from ac to bc
% r recombination rate per genome
% s0 average selection coefficient
% a index in the power distribution_s (not used)
% L number of loci
% N population number
% M crossover number
% tf full time in genertions
% run the number of the run
% ac - left border of the s distribution
% bc - right border of the s distribution

%% random seed, so can average over runs.
rng(run2)




yessave=0; % save figure or not?
yesplot=0; % plotting anything but Fig 1?

%% Mutations
                     % N*Ub*s^2
mu=NUbs2/N/s0^2/L;
%%

%mu=0;% 3e-5;                    % recurrent beneficial mutation rate 
tesc=0;                         % time interval for a kind of initial conditions
T= 0:tf ;                       % times
N_sample=30;                    % Sampling segment for w2 and LD
muesc=0;                        % standing variation f0(s): deleterious mutation rate during escape

%% Distributions of s and parameter titles of fig 1 


       % s = s0*(ac + (bc-ac)*(rand(1,L)));
       s = s0*(ones(1,L));
       % disp(s)
 ts=sprintf('minus N=%g,r=%g,L=%g,s0=%g,aс=%g,bc=%g,\n f0=%g, muesc=%g,mu=%g,tesc=%g,\n T=%g,M=%g',...
            N,r,L,s0,ac,bc,f0, muesc,mu,tesc,max(T),M);

rng(run^run2)

%% Initial settings
 
fsite=zeros(length(T),L); Plinkav=fsite; %fanc=fsite;

fav=zeros(length(T),1);  Cav=fav; w2av=fav; 
nlineages=fav; Closs=fav; fpol=fav; Cpol=fav; Vark=fav;
w2=zeros(1,N_sample); Plink=zeros(N_sample,L);  C=zeros(1,L); 
nprog=zeros(N,1);
 
Knew=zeros(N,L);  % binary DNA sequences
Anew=zeros(N,L);  % ancestor labels
Pnew=zeros(N,L);  % parent labels

tint=tf/5; % plot time interval 
col='rgbmkrgbmkrgbmkrgbmk';

%% Initial population

% Case 1: randomly distributed good alleles with fixed frequency
if f0~=0
    K=(rand(N,L) < f0); 
    %disp(mean(mean(K)));
    % Matrix K: Each row is a genome, sequence of 0 and 1
    
% Case 2: dynamic distribution based on the antigenic escape compensation model: 
% Alleles acumulate during tesc as deleterious, then change the sign of s. 
% Model valid if 1-site deterministic: f0(s) > 1/Ns <=> mu*N > 1 for s > 1/tesc)
elseif muesc~=0
	f0=muesc*s.^(-1).*(1-exp(-s*tesc));       % row of initial frequencies 
	K=(rand(N,L) < ones(N,1)*f0);
    
% Case 3: no alleles at start
else
    K=zeros(N,L);
end

A=(1:N)'*ones(1,L);  % Initial ancestor labels used for C(t)

P1=zeros(N,length(T)); Pm=P1; PL=P1; % Initial parent labels for 3 sites in time, for phylogeny 
W=P1;                                % Initial fitness values
% Epistasis
E = ones(L,L)*epi; % Constant E value
%Te = Tmatrix;
Te = zeros(L,L);
% 2 arcs form
%for ie = 1:2:L-1
%Te(ie, ie+1) = 1;
%Te(ie+1,ie) = 1;
%end
% triple arcs form
%for ie = 1:3:L-1
%Te(ie, ie+1) = 1;
%Te(ie+1,ie) = 1;
%Te(ie+1, ie+2) = 1;
%Te(ie+2,ie+1) = 1;
%end
% One star in the middle with 12 rays
for ie = 1:6
Te(round(10),round(10)+ie) = 1;
Te(round(10),round(10)-ie) = 1;
Te(round(10)+ie,round(10)) = 1;
Te(round(10)-ie,round(10)) = 1;
end

%disp(size(Te));
%disp(round(L/2));
%disp(Te);

Se = zeros(L,L);

for ie2 = 1:L
    for je2 = 1:L
        if ie2 ~= je2
        Se(ie2,je2) = E(ie2,je2)*(s(ie2)+s(je2))*Te(ie2,je2);
        end
    end
end
disp(Se)
%% Evolution starts...
for t=T
    %ftd = mean(K,'all');
%disp("t is " + t + "fbeg: ")
%disp(ftd)

% beneficial mutation if any
%disp(mu)
    if mu > 0
      %  disp(mu)
      %  K = K | sprand(N,L,mu);
     %   K= ~K & sprand(N,L,mu);
    % srav = sprand(N,L,mu);
    % disp(srav);
    % raz = size(K);
     % only matrix
     %K = xor(K,srav);
    % disp(K)
     %%
    %1 for kk2 = 1:raz(1)
     %1for kk = 1:raz(2)
%if s(kk)>=0 
     %1K(kk2,kk) = xor(K(kk2,kk),srav(kk2,kk));
%else
    %K(kk2,kk) = ~(~K(kk2,kk) | srav(kk2,kk));
    
%end     
    %1 end
     %1end
     %%
     K = xor(K, rand(N,L) < mu);
    end
    
% Initial parent labels for one-time population 
    P=(1:N)'*ones(1,L);  
   
%% Recombination of randomly chosen pairs with one-parent replacement
    npairs=round(r*N/2);
    ii=ceil(rand(npairs,2)*N); i1=ii(:,1); i2=ii(:,2);  % 2 columns of random indices of parents
    for i=1:npairs
        % generating random 1 or 0 for each site, with probabilities M/L and 1-M/L,
        % respectively, to mark crossovers with 1
        % Even/odd xx shows site segments copied from 1st parent, 2nd, 1st, 2nd etc
        xx=cumsum(rand(1,L) < M/L); 
        first=(round(xx/2)==xx/2);                          %  sites copied from 1st parent are marked as 1
        prog1=K(i1(i),:).*first+K(i2(i),:).*(1-first);      % recombinant DNA sequence
        progA=A(i1(i),:).*first+A(i2(i),:).*(1-first);      % recombinant ancestor labels
        progP=P(i1(i),:).*first+P(i2(i),:).*(1-first);      % recombinant parent labels
%% replacing a parent FIX
        if rand > 0.5  
            K(i1(i),:)=prog1; % 1st parent's DNA replaced
            A(i1(i),:)=progA; % 1st parent's ancestor label replaced
            P(i1(i),:)=progP; % 1st parent's ancestor label replaced
        else
            K(i2(i),:)=prog1; % 2nd parent's DNA replaced
            A(i2(i),:)=progA; % 2nd parent's ancestor label replaced
            P(i2(i),:)=progP; % 2nd parent's ancestor label replaced
        end


    end
   
    
%% Random sampling of progeny and natural selection with a broken stick

    % column of N log-fitnesses ; state with 0s only has w=0 (fitness 1) by definition
    %w=K*(s'); standart
    %disp(size(w));
    % Epistasis term start
    wE = zeros(1,N);
    %disp(size(wE'));
    %disp(size(K*(s')));
    %disp(t);
   % disp(K);
    for ne = 1:N
        for ie3 = 1:L-1
            for je3 = ie3+1:L
            wE(ne) =wE(ne) + Se(ie3,je3)*K(ne,ie3)*K(ne,je3);
            
            %disp("N "+ ne + " Se " + Se(ie3,je3)+" K*K "+K(ne,ie3)*K(ne,je3))
          %  if K(ne,ie3)*K(ne,je3) == 1
           %     disp("Here is 1 "+"Se is " + Se);
           % end
         %  if Se(ie3,je3) ~= 0
             %  disp("Se is " + Se(ie3,je3) + "K*K is " + K(ne,ie3)*K(ne,je3));
          % end


           % disp ("i " + ie3 + " j "+je3)
            end
        end
    end
  % disp("It was " + t)
   % disp(sum(wE));
    %disp(sum(-K*(s')));
    %disp(Se)
    % Epistasit term end

     w=-K*(s') + wE';
    % disp(K*(s') - (-K*(s') + wE'));
    %disp(t);
    %disp(K);
    %disp(s');
    %disp(-K*(s'));
    %disp(w);
    %disp(wE');
    %disp(w);
     
   
    
    nprogav=exp(w)/mean(exp(w));    % average progeny number
    b2=cumsum(nprogav);
    b1=[0;b2(1:end-1)];             % broken stick
    X=rand(N,1)*N;
    for i=1:N
        nprog(i)=sum( X > b1(i) & X < b2(i)); % actual progeny number
    end
  
%% Updating population
    is=[0;cumsum(nprog(1:(N-1)))];
    for i=1:N
        if nprog(i)
            Knew(is(i)+1:is(i)+nprog(i),:)=ones(nprog(i),1)*K(i,:);  % DNA sequences
            Anew(is(i)+1:is(i)+nprog(i),:)=ones(nprog(i),1)*A(i,:);  % ancestor labels
            Pnew(is(i)+1:is(i)+nprog(i),:)=ones(nprog(i),1)*P(i,:);  % parent labels
        end
    end
    K=Knew;
    A=Anew; 
    P=Pnew;
    sK=size(Knew);sK=sK(1);
    if sK  ~= N, disp('N not conserved'),return;end
    
    % updating sequences done
    
    %% Memorizing observables 
    W(:,t+1)=-K*(s') + wE';                % fitness column
    P1(:,t+1)=P(:,1);               % parent labels for 3 sites  
    Pm(:,t+1)=P(:,L/2); 
    PL(:,t+1)=P(:,L);                 
    %ipol=find(~all(1-K,1));         % non-lost allele site  indices
    ipol=find(~all(K.*(1-K)));       % heterozygous site  indices
    Closs(t+1)=1-length(ipol)/L;    % fraction of sites with extinct (or non heterozygous) alleles
    fsite(t+1,:)=mean(K);           % 1-site allele frequencies at all sites
   wav(t+1) = mean(mean(W));
    fav(t+1)=mean(mean(K));         % average allele frequency per site per genome
    fpol(t+1)=mean(mean(K(:,ipol)));% average allele frequency per polymorphic site 
 %   Vark(t+1)=(std(w)/s0).^2;        % variance of the allele number between genomes
    
    % Sampling random pairs for w2
    ii=ceil(rand(N_sample,2)*N); i1=ii(:,1); i2=ii(:,2);  % 2 columns of N_sample random indices between 1 and N

    for i=1:N_sample
        w2(i)=sum(K(i1(i),:)~=K(i2(i),:))/2; % half-distance
        for j=0:(L-1)
            Plink(i,j+1)=mean(A(i1(i),1:(L-j))==A(i1(i),(j+1):L)); % for LD
            %  probability of 2 sites at distance j in one genome to have the same ancestor 
        end
    end
    % Average over the sample 
    w2av(t+1)= mean(w2); 
    nlineages(t+1)=length(unique(reshape(A,[1,N*L])));  % total dictinct lineage number
    Plinkav(t+1,:)= mean(Plink);  
    
   
    %allowed=fsite(t+1,:) > 0 && fsite(t+1,:) < 1; % binary mask of polymorphous sites

        
    % Ancestor spectrum and C
    nanc =[];nmax=zeros(1,L);
    for i=1:1:L
        %Au=unique(A(:,i));
        %spec_anc=sum(ones(N,1)*Au'== A(:,i)*ones(1,length(Au)))/N;
        spec_anc=hist(A(:,i),min(A(:,i)):max(A(:,i)));
        C(i)=sum(spec_anc.^2)/N^2;
        % clone sizes
        if ~all(1-K(:,i))
            nanc =[nanc  spec_anc];
            nmax(i)=max(spec_anc);
        end % non-lost sites only
        %fanc(t+1,1:length(spec_anc))=spec_anc;
    end
  
    Cav(t+1)=mean(C(C>0));
    Cpol(t+1)=mean(C(ipol));
    nmax=nmax(ipol);
    
    %% Plotting the wave and ancestral clone  spectrum at some time points
    %{
    if tint*round(t/tint)==t %&& yesplot
        c=col(round(t/tint)+1);
        figure(2)
     subplot( tf/tint +2,1,1)
        [nn,xx]=hist( w );                % histogram of fitness among genomes
        semilogy(xx,nn,[c '+'])
        hold on
     subplot( tf/tint +2,2, 2*t/tint+3)
        [nn,xx]=hist( nanc(nanc>0),min(30,length(nanc(nanc>0))) );                % histogram of fitness among genomes
        semilogy(xx/N,nn,[c 'o'])
        title(sprintf('t=%g',t))
        axi=axis;axi(1:2)=[0 1 ];axis(axi);
     subplot( tf/tint +2,2, 2*t/tint+4)
        [nn,xx]=hist( nmax(nmax>0),min(30,length(nmax(nmax>0))) );                % histogram of fitness among genomes
        semilogy(xx/N,nn,[c 'o'])
         histogram(nanc(nanc>0),30,'FaceColor', c )
        title(sprintf('t=%g',t))
        axi=axis;axi(1:2)=[0 1 ];axis(axi);
    end
    %}
    %% Plotting LD (average Pearson r^2) at some time points
    if tint*round(t/tint)==t && yesplot
        c=col(round(t/tint)+1);
        %figure(3)
        LD=zeros(1,L );
        for j=0:(L-1)
            yy=0; nsum=0;
            for i=1:(L-j)
                xx1=fsite(t+1,i)*(1-fsite(t+1,i));
                xx2=fsite(t+1,i+j)*(1-fsite(t+1,i+j));
                % if each site is more than 5% diverse, we add LD
                if xx1 > 0.05 && xx2 > 0.05  
                    nsum=nsum+1;
                    yy=yy+(mean(K(:,i).*K(:,i+j)) - fsite(t+1,i)*fsite(t+1,i+j))^2/xx1/xx2;
                end
            end
            LD(j+1)=yy/nsum;
        end

    end

%disp(t);
% 3 points
%if t == tsec1
%    genome_r{1,1} = K;
%
%end
%if t == tsec2
% genome_r{1,2} = K;  
%end
%if t == tsec3
%genome_r{1,3} = K;
%sdist = s;
%end
%

[histo1{1,t+1},histo2{1,t+1}] = hist( w );

 %   ftd = mean(K,'all');
%disp("t is " + t + "fend: ")
%disp(ftd)

%epi end conditions etc

if t == tf
genome_r = K;
sdist = s;
%disp(sum(wE));
end
%disp(t);
%disp(K);
end




% Evolution has ended
avf = fav;
avw = wav;
fsi = fsite;
end




