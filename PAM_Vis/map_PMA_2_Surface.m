clear
clc

addpath D:\note0719\matlab_path\Function_lin
addpath D:\note0719\matlab_path\npy-matlab-master\npy-matlab


T = readtable('PAM-nonsense_HCP_A.xlsx');

load('schaefer400_2_rsn7.mat');
load('hcp_64984_2_rsn7.mat');


data = table2array(T);


rsn_data =  zeros(7,size(data,2));

for j=1:size(data,2)
    for i =1:7
        rsn_data(i , j) = mean(data(find(schaefer400_2_rsn7==i),j));
    end
end




surface_2_rsn_index = zeros(64984,1);

for i=1:size(surface_2_rsn_index,1)
    if a2(i,1)==0
        continue
    end

    surface_2_rsn_index(i,1) = schaefer400_2_rsn7(a2(i,1),1);
end


surface_data = zeros(64984,size(data,2));
for j=1:size(data,2)
    for i=1:size(rsn_data,1)
        surface_data(find(surface_2_rsn_index==i),j) = rsn_data(i,j);
    end
end


surf_l = vtkSurfRead('S1200.L.white_GyralNet.surf.vtk');
surf_l = rmfield(surf_l, 'Pdata');
surf_r = vtkSurfRead('S1200.R.white_GyralNet.surf.vtk');
surf_r = rmfield(surf_r, 'Pdata');



for tt=1:size(data,2)
    surf_l.Pdata{1, tt}.val = surface_data(1:32492,tt);
    surf_l.Pdata{1, tt}.val(find(surf_l.Pdata{1, tt}.val==0)) = min(rsn_data(:,tt));
    surf_l.Pdata{1, tt} .name = ['pam', num2str(tt)];
    surf_r.Pdata{1, tt}.val = surface_data(32493:64984,tt);
    surf_r.Pdata{1, tt}.val(find(surf_r.Pdata{1, tt}.val==0)) = min(rsn_data(:,tt));
    surf_r.Pdata{1, tt} .name = ['pam', num2str(tt)];
end


surf_l.Face  = surf_l.Face-1;
vtkSurfWrite('surf_l_pam_HCP_A.vtk',surf_l);

surf_r.Face  = surf_r.Face-1;
vtkSurfWrite('surf_r_pam_HCP_A.vtk',surf_r);



%%


clear
clc

addpath D:\note0719\matlab_path\Function_lin
addpath D:\note0719\matlab_path\npy-matlab-master\npy-matlab


T = readtable('PAM-nonsense_HCP_D.xlsx');

load('schaefer400_2_rsn7.mat');
load('hcp_64984_2_rsn7.mat');


data = table2array(T);


rsn_data =  zeros(7,size(data,2));

for j=1:size(data,2)
    for i =1:7
        rsn_data(i , j) = mean(data(find(schaefer400_2_rsn7==i),j));
    end
end




surface_2_rsn_index = zeros(64984,1);

for i=1:size(surface_2_rsn_index,1)
    if a2(i,1)==0
        continue
    end

    surface_2_rsn_index(i,1) = schaefer400_2_rsn7(a2(i,1),1);
end


surface_data = zeros(64984,size(data,2));
for j=1:size(data,2)
    for i=1:size(rsn_data,1)
        surface_data(find(surface_2_rsn_index==i),j) = rsn_data(i,j);
    end
end


surf_l = vtkSurfRead('S1200.L.white_GyralNet.surf.vtk');
surf_l = rmfield(surf_l, 'Pdata');
surf_r = vtkSurfRead('S1200.R.white_GyralNet.surf.vtk');
surf_r = rmfield(surf_r, 'Pdata');



for tt=1:size(data,2)
    surf_l.Pdata{1, tt}.val = surface_data(1:32492,tt);
    surf_l.Pdata{1, tt}.val(find(surf_l.Pdata{1, tt}.val==0)) = min(rsn_data(:,tt));
    surf_l.Pdata{1, tt} .name = ['pam', num2str(tt)];
    surf_r.Pdata{1, tt}.val = surface_data(32493:64984,tt);
    surf_r.Pdata{1, tt}.val(find(surf_r.Pdata{1, tt}.val==0)) = min(rsn_data(:,tt));
    surf_r.Pdata{1, tt} .name = ['pam', num2str(tt)];
end


surf_l.Face  = surf_l.Face-1;
vtkSurfWrite('surf_l_pam_HCP_D.vtk',surf_l);

surf_r.Face  = surf_r.Face-1;
vtkSurfWrite('surf_r_pam_HCP_D.vtk',surf_r);

%%
clear
clc

addpath D:\note0719\matlab_path\Function_lin
addpath D:\note0719\matlab_path\npy-matlab-master\npy-matlab


T = readtable('PAM-nonsense_HCP_YA.xlsx');

load('schaefer400_2_rsn7.mat');
load('hcp_64984_2_rsn7.mat');


data = table2array(T);


rsn_data =  zeros(7,size(data,2));

for j=1:size(data,2)
    for i =1:7
        rsn_data(i , j) = mean(data(find(schaefer400_2_rsn7==i),j));
    end
end




surface_2_rsn_index = zeros(64984,1);

for i=1:size(surface_2_rsn_index,1)
    if a2(i,1)==0
        continue
    end

    surface_2_rsn_index(i,1) = schaefer400_2_rsn7(a2(i,1),1);
end


surface_data = zeros(64984,size(data,2));
for j=1:size(data,2)
    for i=1:size(rsn_data,1)
        surface_data(find(surface_2_rsn_index==i),j) = rsn_data(i,j);
    end
end


surf_l = vtkSurfRead('S1200.L.white_GyralNet.surf.vtk');
surf_l = rmfield(surf_l, 'Pdata');
surf_r = vtkSurfRead('S1200.R.white_GyralNet.surf.vtk');
surf_r = rmfield(surf_r, 'Pdata');



for tt=1:size(data,2)
    surf_l.Pdata{1, tt}.val = surface_data(1:32492,tt);
    surf_l.Pdata{1, tt}.val(find(surf_l.Pdata{1, tt}.val==0)) = min(rsn_data(:,tt));
    surf_l.Pdata{1, tt} .name = ['pam', num2str(tt)];
    surf_r.Pdata{1, tt}.val = surface_data(32493:64984,tt);
    surf_r.Pdata{1, tt}.val(find(surf_r.Pdata{1, tt}.val==0)) = min(rsn_data(:,tt));
    surf_r.Pdata{1, tt} .name = ['pam', num2str(tt)];
end


surf_l.Face  = surf_l.Face-1;
vtkSurfWrite('surf_l_pam_HCP_YA.vtk',surf_l);

surf_r.Face  = surf_r.Face-1;
vtkSurfWrite('surf_r_pam_HCP_YA.vtk',surf_r);



