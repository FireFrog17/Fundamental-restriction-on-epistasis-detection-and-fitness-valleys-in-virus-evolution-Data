allr = [0,0.1,0.5,1];
allrstring = ["0","01","05","1"];
allL = [20,40,200];
allepi = [0,0.5,0.6,0.75];
allepistring = ["0","05","06","075"];
allrun = [6,20,50];
for zz = 1:1 % r
for ii = 1:1 % run
for jj = 1:1 % L
for kk = 1:1 % epi
%run=50;
run = allrun(ii);
%r=1;
r = allr(zz);
rs = allrstring(zz);
%L=200;
L = allL(jj);
M=3;
N=1000;
run2=3;
%NUbs2=0.25;
NUbs2=0.0007*1000;
tf=30;
s0=0.1;
    f0=0.45;
    ac=-1;
    bc=1;
    mu=NUbs2/N/s0^2/L;
    %mu = 0;
    tsec1=20;
    tsec2=40;
    tsec3=60;
    C=0;
    appr='test'; 
    apprR=4;
    %epi = 0.5;
    epi = allepi(kk);
    epis = allepistring(kk);
for i =1:run
[dat{i},~,ftest{i},~,~,~,~]=recomb_2023_epi(r,s0,ac,bc,M,L,N,tf,f0,i,run2,epi,NUbs2);
end
%i = 3;
%[endData,~,~,~,~,~,~]=recomb_2023_epi(r,s0,ac,bc,M,L,N,tf,f0,i,run2,epi);
disp(mu*L);
path = "DataChaos\"+"r"+rs+"run"+int2str(run)+"E"+epis+"L"+int2str(L)+"chaos";
save(path,"dat")
disp(kk)
clear dat ftest
end
end
end
end