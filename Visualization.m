%===========================================================================================
%Copyright (c) 2018 by Georgia Tech Research Corporation.
%All rights reserved.
%
%The files contain code and data associated with the paper titled
%"A Deep Learning Approach to Estimate Stress Distribution: A Fast and
%Accurate Surrogate of Finite Element Analysis".
%
%The paper is authored by Liang Liang, Minliang Liu, Caitlin Martin,
%and Wei Sun, and published at Journal of The Royal Society Interface, 2018.
%
%The file list: ShapeData.mat, StressData.mat, DLStress.py, im2patch.m,
%UnsupervisedLearning.m, ReadMeshFromVTKFile.m, ReadPolygonMeshFromVTKFile.m,
%WritePolygonMeshAsVTKFile.m, Visualization.m, TemplateMesh3D.vtk, TemplateMesh2D.vtk.
%Note: *.m and *.py files were converted to pdf files for documentation purpose.
%
%THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR IMPLIED WARRANTIES,
%INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
%FOR A PARTICULAR PURPOSE.
%===========================================================================================


%%
load('ShapeData.mat') %729 shapes
load('StressData.mat') %stress from FEA


%% show ground truth stress on 3D and 2D meshes of the 729 shapes

for ShapeIndex=1:729

    S11=StressData(1,:,ShapeIndex);
    S22=StressData(2,:,ShapeIndex);
    S12=StressData(4,:,ShapeIndex);
    Von=sqrt(S11.*S11+S22.*S22-S11.*S22+3*S12.*S12);
    Mesh3D = ReadPolygonMeshFromVTKFile('TemplateMesh3D.vtk');
    Mesh3D.Point=reshape(ShapeData(:,ShapeIndex), [3, 5000]);
    Mesh3D.PointData(1).Name='S11';
    Mesh3D.PointData(1).Data=S11(:);
    Mesh3D.PointData(2).Name='S22';
    Mesh3D.PointData(2).Data=S22(:);
    Mesh3D.PointData(3).Name='S12';
    Mesh3D.PointData(3).Data=S12(:);
    Mesh3D.PointData(4).Name='Von';
    Mesh3D.PointData(4).Data=Von(:);
    WritePolygonMeshAsVTKFile(Mesh3D, ['result/' num2str(ShapeIndex) '_Aorta_stress_FEA.vtk'])

    % show ground truth stress on 2D mesh
    S11=StressData(1,:,ShapeIndex);
    S22=StressData(2,:,ShapeIndex);
    S12=StressData(4,:,ShapeIndex);
    S11=reshape(S11, [50, 100]);
    S11(end+1,:)=S11(1,:);
    S22=reshape(S22, [50, 100]);
    S22(end+1,:)=S22(1,:);
    S12=reshape(S12, [50, 100]);
    S12(end+1,:)=S12(1,:);
    Von=sqrt(S11.*S11+S22.*S22-S11.*S22+3*S12.*S12);
    Mesh2D = ReadPolygonMeshFromVTKFile('TemplateMesh2D.vtk');
    Mesh2D.PointData(1).Name='S11';
    Mesh2D.PointData(1).Data=S11(:);
    Mesh2D.PointData(2).Name='S22';
    Mesh2D.PointData(2).Data=S22(:);
    Mesh2D.PointData(3).Name='S12';
    Mesh2D.PointData(3).Data=S12(:);
    Mesh2D.PointData(4).Name='Von';
    Mesh2D.PointData(4).Data=Von(:);
    WritePolygonMeshAsVTKFile(Mesh2D, ['result/' num2str(ShapeIndex) '_StressGrid_FEA.vtk'])
end



%% load predicted stress and idx_test
load('StressData_pred.mat')
for k=1:size(Sp, 2)
    StressData_pred(1,:,k)=Sp(1:5000,k);
    StressData_pred(2,:,k)=Sp(5001:10000,k);
    StressData_pred(3,:,k)=Sp(10001:15000,k);
end




%% show predicted stress on 3D and 2D meshes of the shapes in the testing set (idx_test)
for Index=1:length(idx_test)

    ShapeIndex=idx_test(Index);
    S11=StressData_pred(1,:,Index);
    S22=StressData_pred(2,:,Index);
    S12=StressData_pred(3,:,Index);
    Von=sqrt(S11.*S11+S22.*S22-S11.*S22+3*S12.*S12);
    Mesh3D = ReadPolygonMeshFromVTKFile('TemplateMesh3D.vtk');
    Mesh3D.Point=reshape(ShapeData(:,ShapeIndex), [3, 5000]);
    Mesh3D.PointData(1).Name='S11';
    Mesh3D.PointData(1).Data=S11(:);
    Mesh3D.PointData(2).Name='S22';
    Mesh3D.PointData(2).Data=S22(:);
    Mesh3D.PointData(3).Name='S12';
    Mesh3D.PointData(3).Data=S12(:);
    Mesh3D.PointData(4).Name='Von';
    Mesh3D.PointData(4).Data=Von(:);
    WritePolygonMeshAsVTKFile(Mesh3D, ['result2/' num2str(ShapeIndex) '_Aorta_stress_DL.vtk'])

    % show predicted stress on 2D mesh
    S11=StressData_pred(1,:,Index);
    S22=StressData_pred(2,:,Index);
    S12=StressData_pred(3,:,Index);
    S11=reshape(S11, [50, 100]);
    S11(end+1,:)=S11(1,:);
    S22=reshape(S22, [50, 100]);
    S22(end+1,:)=S22(1,:);
    S12=reshape(S12, [50, 100]);
    S12(end+1,:)=S12(1,:);
    Von=sqrt(S11.*S11+S22.*S22-S11.*S22+3*S12.*S12);
    Mesh2D = ReadPolygonMeshFromVTKFile('TemplateMesh2D.vtk');
    Mesh2D.PointData(1).Name='S11';
    Mesh2D.PointData(1).Data=S11(:);
    Mesh2D.PointData(2).Name='S22';
    Mesh2D.PointData(2).Data=S22(:);
    Mesh2D.PointData(3).Name='S12';
    Mesh2D.PointData(3).Data=S12(:);
    Mesh2D.PointData(4).Name='Von';
    Mesh2D.PointData(4).Data=Von(:);
    WritePolygonMeshAsVTKFile(Mesh2D, ['result2/' num2str(ShapeIndex) '_StressGrid_DL.vtk'])
end
