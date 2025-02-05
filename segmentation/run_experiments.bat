@echo off

@REM :: SAM2 with bounding box prompts
@REM "c:/Program Files/Python311/python.exe" D:/RoVF-meerkat-reidentification/segmentation/SAM2_BBOX.py -i polarbears_vid -dir polarbears -o D:/RoVF-meerkat-reidentification/segmentation/results/
@REM "c:/Program Files/Python311/python.exe" D:/RoVF-meerkat-reidentification/segmentation/SAM2_BBOX.py -i meerkat_vid -dir meerkat -o D:/RoVF-meerkat-reidentification/segmentation/results/

@REM :: Default settings for DINOv2-LDA and DINOv2-LDA-SAM
@REM "c:/Program Files/Python311/python.exe" D:/RoVF-meerkat-reidentification/segmentation/DINOv2_LDA_SAM2.py -i polarbears_DINOv2LDA -dir polarbears --use_LDA True --restart True -fp "[0,10,19]" -o D:/RoVF-meerkat-reidentification/segmentation/results/
@REM "c:/Program Files/Python311/python.exe" D:/RoVF-meerkat-reidentification/segmentation/DINOv2_LDA_SAM2.py -i meerkat_DINOv2LDA -dir meerkat --use_LDA True --restart True -fp "[0,10,19]" -o D:/RoVF-meerkat-reidentification/segmentation/results/

@REM :: DINOv2-PCA and DINOv2-PCA-SAM
@REM "c:/Program Files/Python311/python.exe" D:/RoVF-meerkat-reidentification/segmentation/DINOv2_LDA_SAM2.py -i meerkat_DINOv2PCA -dir meerkat --use_LDA False --restart True -fp "[0,10,19]" -o D:/RoVF-meerkat-reidentification/segmentation/results/
@REM "c:/Program Files/Python311/python.exe" D:/RoVF-meerkat-reidentification/segmentation/DINOv2_LDA_SAM2.py -i polarbears_DINOv2PCA -dir polarbears --use_LDA False --restart True -fp "[0,10,19]" -o D:/RoVF-meerkat-reidentification/segmentation/results/

@REM :: DINOv2-LDA and DINOv2-LDA-SAM with only the first frame prompt
@REM "c:/Program Files/Python311/python.exe" D:/RoVF-meerkat-reidentification/segmentation/DINOv2_LDA_SAM2.py -i meerkat_DINOv2LDA_f0 -dir meerkat --use_LDA True --restart True -fp "[0]" -o D:/RoVF-meerkat-reidentification/segmentation/results/
@REM "c:/Program Files/Python311/python.exe" D:/RoVF-meerkat-reidentification/segmentation/DINOv2_LDA_SAM2.py -i polarbears_DINOv2LDA_f0 --use_LDA True --restart True -fp "[0]" -o D:/RoVF-meerkat-reidentification/segmentation/results/

@REM :: DINOv2-PCA and DINOv2-PCA-SAM with only the first frame prompt
@REM "c:/Program Files/Python311/python.exe" D:/RoVF-meerkat-reidentification/segmentation/DINOv2_LDA_SAM2.py -i meerkat_DINOv2PCA_f0 -dir meerkat --use_LDA False --restart True -fp "[0]" -o D:/RoVF-meerkat-reidentification/segmentation/results/
@REM "c:/Program Files/Python311/python.exe" D:/RoVF-meerkat-reidentification/segmentation/DINOv2_LDA_SAM2.py -i polarbears_DINOv2PCA_f0 --use_LDA False --restart True -fp "[0]" -o D:/RoVF-meerkat-reidentification/segmentation/results/

:: DINOv2-LDA and DINOv2-LDA-SAM without masking boxes during post-processing
"c:/Program Files/Python311/python.exe" D:/RoVF-meerkat-reidentification/segmentation/DINOv2_LDA_SAM2.py -i meerkat_DINOv2LDA_nm -dir meerkat --use_LDA True --restart True -fp "[0,10,19]" -mb False -o D:/RoVF-meerkat-reidentification/segmentation/results/
"c:/Program Files/Python311/python.exe" D:/RoVF-meerkat-reidentification/segmentation/DINOv2_LDA_SAM2.py -i polarbears_DINOv2LDA_nm -dir polarbears --use_LDA True --restart True -fp "[0,10,19]" -mb False -o D:/RoVF-meerkat-reidentification/segmentation/results/

:: DINOv2-PCA and DINOv2-PCA-SAM without masking boxes during post-processing
"c:/Program Files/Python311/python.exe" D:/RoVF-meerkat-reidentification/segmentation/DINOv2_LDA_SAM2.py -i meerkat_DINOv2PCA_nm -dir meerkat --use_LDA False --restart True -fp "[0,10,19]" -mb False -o D:/RoVF-meerkat-reidentification/segmentation/results/
"c:/Program Files/Python311/python.exe" D:/RoVF-meerkat-reidentification/segmentation/DINOv2_LDA_SAM2.py -i polarbears_DINOv2PCA_nm -dir polarbears --use_LDA False --restart True -fp "[0,10,19]" -mb False -o D:/RoVF-meerkat-reidentification/segmentation/results/

pause