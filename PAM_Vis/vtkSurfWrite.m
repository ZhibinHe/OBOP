%cojoc(at)hotmail.com
%2014.3.16

function vtkSurfWrite(fname_output, Patches)
%%inputs:
%%fname_output: file names 'example.vtk'
%%Patches
%%  Patches.Vtx 3*pNum
%%  Patches.Face 3*faceNum
%%  Patches.Pdata{i}.val  label values
%%  Patches.Pdata{i}.name  label name

    %open file
    fp=fopen(fname_output,'w');
    if fp<=0
        error(['cannot open file ' fname_output 'for read']);
    end
    
    fprintf(fp,'# vtk DataFile Version 3.0\nmesh surface\nASCII\nDATASET POLYDATA\n');
    fprintf(fp,'POINTS %d float\n',size(Patches.Vtx,2));
    fprintf(fp, '%f %f %f\n',Patches.Vtx);
    fprintf(fp,'POLYGONS %d %d\n',size(Patches.Face,2),size(Patches.Face,2)*4);
    fprintf(fp,'3 %d %d %d\n',Patches.Face);
    if(isfield(Patches,'Pdata'))
    fprintf(fp, 'POINT_DATA %d\n',size(Patches.Vtx,2));
        for i=1:length(Patches.Pdata)
            fprintf(fp, 'SCALARS %s float\n',Patches.Pdata{i}.name);
            fprintf(fp, 'LOOKUP_TABLE %s\n',Patches.Pdata{i}.name);
            fprintf(fp, '%f\n',Patches.Pdata{i}.val);
        end
    end
    fclose(fp);
end