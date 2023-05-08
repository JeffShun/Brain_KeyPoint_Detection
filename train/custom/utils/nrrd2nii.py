import os
from glob import glob
import numpy as np
import vtk

def readnrrd(filename):
    """Read image in nrrd format."""
    reader = vtk.vtkNrrdReader()
    reader.SetFileName(filename)
    reader.Update()
    info = reader.GetInformation()
    return reader.GetOutput(), info


def writenifti(image,filename, info):
    """Write nifti file."""
    writer = vtk.vtkNIFTIImageWriter()
    writer.SetInputData(image)
    writer.SetFileName(filename)
    writer.SetInformation(info)
    writer.Write()


if __name__ == '__main__':
    baseDir = '../../data/AI试标注数据/seg_nrrd'
    saveDir = baseDir.replace("seg_nrrd","seg_nii")
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    files = glob(baseDir+'/*seg.nrrd')
    for file in files:
        m, info = readnrrd(file)
        writenifti(m,  file.replace('seg.nrrd', 'seg.nii.gz').replace(baseDir, saveDir), info)
