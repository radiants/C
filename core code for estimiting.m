    xsa=0.0570;% XSA(i); the spread parameter 
    
    Ptrain=systemnakcl; % input solubility
    EstData=xlsread('estdata.xls',i); % not in training set
    
    xi=Ptrain(:,1); % read training set (3 colums)
    yi=Ptrain(:,2);
    zi=Ptrain(:,3);
    
    LenTraini=length(xi);
    
    ai=EstData(:,1); % read test set (3 colums)
    bi=EstData(:,2);
    ci=EstData(:,3);

    LenTesti=length(ai);
    
    ui=[xi;ai]; % combine for normalize
    vi=[yi;bi];
    wi=[zi;ci];
    
    [Ui,PSUi]=mapminmax(ui',0,1);  % normalize
    [Vi,PSVi]=mapminmax(vi',0,1);
    [Wi,PSWi]=mapminmax(wi',0,1);
    
    Xi=Ui(1:LenTraini);
    Yi=Vi(1:LenTraini);
    Zi=Wi(1:LenTraini);
    
    Ai=Ui(LenTraini+1:LenTraini+LenTesti);  % 
    Bi=Vi(LenTraini+1:LenTraini+LenTesti);
    Ci=Wi(LenTraini+1:LenTraini+LenTesti);
    
    NewNetCi=newgrnn([Xi; Yi],Zi,xsa); % construct grnn net
    TCi=sim(NewNetCi,[Ai;Bi]);  
    tci{i}= mapminmax('reverse',TCi,PSWi); % anti-normalized
    tciT{i}=tci{i}';    % est. result of colume III;
    
    NewNetBi=newgrnn([Xi; Zi],Yi,xsa); % construct grnn net
    TBi=sim(NewNetBi,[Ai;Ci]);  
    tbi{i}= mapminmax('reverse',TBi,PSVi); % anti-normalized
    tbiT{i}=tbi{i}';   % est. result of colume II;
