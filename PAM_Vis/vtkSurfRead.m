function Patch = vtkSurfRead(fname)
fp = fopen(fname,'rt');

%get points
s = fscanf(fp,'%s',1);
while (~strcmp(s,'POINTS'))
    s = fscanf(fp,'%s',1);
end
VtxNum = fscanf(fp,'%d',1);
s = fscanf(fp,'%s',1);
Vtx = fscanf(fp,'%f',VtxNum*3);
Vtx = reshape(Vtx,3,VtxNum);
Patch.Vtx = Vtx;

%get faces
s = fscanf(fp,'%s',1);
while ((~strcmp(s,'POLYGONS'))&&(~strcmp(s,'TRIANGLE_STRIPS')))
    if(feof(fp))
        fclose(fp);
        error(['error: failed in searching POLYGONS || TRIANGLE_STRIPS:' ...
            fname]);
    end
    s = fscanf(fp,'%s',1);
end
FaceNum = fscanf(fp,'%d',1);
s = fscanf(fp,'%d',1);
Face = fscanf(fp,'%f',FaceNum*4);
Face = reshape(Face,4,FaceNum);
Face = Face+1;
Patch.Face = Face(2:4,:);

%get point label if there is any
s = fscanf(fp,'%s',1);
Plabel=[];
if(strcmp(s,'POINT_DATA'))
    s = fscanf(fp,'%s',1);
    idx=1;
    while ~feof(fp)
        s = fscanf(fp,'%s',1);
        if(strcmp(s,'SCALARS'))
            name = fscanf(fp,'%s',1);
            s = fscanf(fp,'%s',1); s = fscanf(fp,'%s',1);
            if(strcmp(s,'LOOKUP_TABLE'))
                s = fscanf(fp,'%s',1);
                label = fscanf(fp,'%f',VtxNum);
            else
                label = fscanf(fp,'%f',VtxNum-1);
                label = [str2num(s) label];
            end
            Plabel{idx}.val=label;
            Plabel{idx}.name=name;
            idx=idx+1;
        elseif (strcmp(s,'COLOR_SCALARS'))
            name = fscanf(fp,'%s',1);
            s = fscanf(fp,'%s',1);
            label = fscanf(fp,'%f',VtxNum*3);
            label = reshape(label,[3 VtxNum]);

            Plabel{idx}.val=label;
            Plabel{idx}.name=name;
            idx=idx+1;
        elseif (strcmp(s,'VECTORS'))
            name = fscanf(fp,'%s',1);
            s = fscanf(fp,'%s',1);
            label = fscanf(fp,'%f',VtxNum*3);
            label = reshape(label,[3 VtxNum]);
            Plabel{idx}.val=label;
            Plabel{idx}.name=name;
            idx=idx+1;
        elseif(strcmp(s,'NORMALS'))
            s=fgetl(fp);
            tmp=fscanf(fp,'%f',VtxNum*3);
        elseif(strcmp(s,'neighbors'))
            neighbor=cell(VtxNum,1);
            for i=1:VtxNum
                num=fscanf(fp,'%d',1);
                neighbor{i}=fscanf(fp,'%d',num)+1;
            end
            Patch.Pneighbor=neighbor;
        elseif(strcmp(s,'adjacentfaces'))
            neighbor=cell(VtxNum,1);
            for i=1:VtxNum
                num=fscanf(fp,'%d',1);
                neighbor{i}=fscanf(fp,'%d',num)+1;
            end
            Patch.Fadj=neighbor;
        elseif(strcmp(s,'CELL_DATA'))
            disp('current version do not suport CELL_DATA');
            break;
        else
            break;
        end
    end
end
Patch.Pdata=Plabel;
fclose(fp);
