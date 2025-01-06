set(0,'defaulttextInterpreter','latex')
set(0,'defaultAxesTickLabelInterpreter','latex'); 
set(0,'defaultLegendInterpreter','latex');
set(0,'defaultLineLineWidth',3);
set(0,'defaultAxesFontSize',35)


clc
clf


a = 2.25;
[sx, sy, sz] = sphere(50);



% vid_name = 'Rhombus_Simulation';
% NAME ='suspension_rhombus_N_12_random';
% NAME ='suspension_rhombus_N_12_random_eq1';
% NAME ='suspension_rhombus_N_12_random_eq2';
% f_name = ['./data/' vid_name '.' NAME '.config'];

vid_name = 'Ladder_Simulation';
NAME ='suspension_ladder_N_6_random';
f_name = ['./data/' vid_name '.' NAME '.config'];


L = 128.0;

A = dlmread(f_name);


n_bods = A(1,1); 
B_z_history = A(1:(n_bods+1):end,2);

A(1:(n_bods+1):end,:) = [];
N = length(A)/n_bods;
dt = 80*0.000125;
skip = 4*1;




[X, Y] = meshgrid([-L/2:0.5:L/2],[-L/2:0.5:L/2]);

show_triad = 1;
show_beads = 1;


k = 0;

Ntime = length(A)/n_bods;
for i = Ntime 
    i
    k = k+1;
    clf
    
        

    
    x = A((i-1)*n_bods+1:i*n_bods,1);
    y = A((i-1)*n_bods+1:i*n_bods,2);
    z = A((i-1)*n_bods+1:i*n_bods,3);
    s = A((i-1)*n_bods+1:i*n_bods,4);
    p = A((i-1)*n_bods+1:i*n_bods,5:end);
    
    for d = 1:length(x)
    while x(d) > L/2
        x(d) = x(d) - L;
    end
    while x(d) < -L/2
        x(d) = x(d) + L;
    end
    while y(d) > L/2
        y(d) = y(d) - L;
    end
    while y(d) < -L/2 
        y(d) = y(d) + L;
    end
    end
    
    scaleax = 1.8;
    xlim(scaleax*[-10 10])
    ylim(scaleax*[-10 10])
    zlim([0 8])

    out_cfgs = A((i-1)*n_bods+1:i*n_bods,1:3);
    out_vs = 0*out_cfgs;

    for j = 1:length(x)
        if show_beads==1
        fcol = 0.3*[1 1 1];
        h = surface(x(j)+a*sx,y(j)+a*sy,z(j)+a*sz,'facecolor',fcol,'edgecolor','none');
        set(h,'FaceLighting','gouraud',...%'facealpha',0.2, ...
        'AmbientStrength',0.3, ...
        'DiffuseStrength',0.6, ... 
        'Clipping','off',...
        'BackFaceLighting','lit', ...
        'SpecularStrength',1, ...
        'SpecularColorReflectance',1, ...
        'SpecularExponent',7,'FaceAlpha',1)
        end
    
        daspect([1 1 1])
        grid on
        view([0 90]) %view([-20 35]) %view([-140 10])% 


        set(gca, 'linewidth',3)
        ax1 = gca;
        set(ax1,'XTick',get(ax1,'YTick'));
        hold all
        
        if show_triad==1
        R = Rot_From_Q(s(j),p(j,:));
        V = R*eye(3);
        v = V(:,3);
        tv = V(:,2);
        ev = V(:,1);
        hA1 = mArrow3([x(j); y(j); z(j)],[x(j); y(j); z(j)]+1.3*a*v,'color',[0.8 0.8 0.8],'stemWidth',0.1*a);
        hold all
        hA2 = mArrow3([x(j); y(j); z(j)],[x(j); y(j); z(j)]+1.3*a*tv,'color',[0.8 0.8 0.8],'stemWidth',0.1*a);
        hold all
        hA3 = mArrow3([x(j); y(j); z(j)],[x(j); y(j); z(j)]+1.4*a*ev,'color','m','stemWidth',0.2*a);
        hold all
        out_vs(j,:) = ev;

        end
        
    end



    camlight

    B_z = B_z_history(i);
    title(['t = ' num2str((i-1)*dt) ', $$B_z = $$' num2str(B_z)])
    


    drawnow

    hold off
    
end

