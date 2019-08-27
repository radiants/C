    sprd=0.0570;% XSA(i); the spread parameter 
    load systemnakcl;
    Ptrain=systemnakcl; % input solubility
    EstData=[35 35 35 35 35; 25.00 15.00 10.00 5.00 0.00; 2.90 15.53 18.99 23.34 27.66]; % not listed in training set
    
    xi=Ptrain(:,1); % read training set (3 colums)
    yi=Ptrain(:,2);
    zi=Ptrain(:,3);
    
    LenTraini=length(xi);
    
    ai=EstData(1,:); % read test set (3 lines)
    bi=EstData(2,:);
    ci=EstData(3,:);

    LenTesti=length(ai);
    
    ui=[xi;ai']; % combine for normalize
    vi=[yi;bi'];
    wi=[zi;ci'];
    
    [Ui,PSUi]=mapminmax(ui',0,1);  % normalize
    [Vi,PSVi]=mapminmax(vi',0,1);
    [Wi,PSWi]=mapminmax(wi',0,1);
    
    Xi=Ui(1:LenTraini); % split
    Yi=Vi(1:LenTraini);
    Zi=Wi(1:LenTraini);
    
    Ai=Ui(LenTraini+1:LenTraini+LenTesti);  % split
    Bi=Vi(LenTraini+1:LenTraini+LenTesti);
    Ci=Wi(LenTraini+1:LenTraini+LenTesti);
    
    NewNetCi=newgrnn([Xi; Yi],Zi,sprd); % construct grnn net
    TCi=sim(NewNetCi,[Ai;Bi]);  
    tci = mapminmax('reverse',TCi,PSWi); % anti-normalized（estimated result of line III (kcl))
   
