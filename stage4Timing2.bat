@ECHO OFF
set runs=%1
if "%1"=="" set runs=5
@ECHO ON
FastRelease\Stage4.exe -runs %runs% -blockSize 16 -size 1024 1024 -samples 1 -input Scenes/cornell.txt     
FastRelease\Stage4.exe -runs %runs% -blockSize 16 -size  720 1280 -samples 1 -input Scenes/juggler.txt 
FastRelease\Stage4.exe -runs %runs% -blockSize 16 -size  240  135 -samples 1 -input Scenes/5000spheres.txt 
FastRelease\Stage4.exe -runs %runs% -blockSize 16 -size 1024 1024 -samples 1 -input Scenes/all.txt 
FastRelease\Stage4.exe -runs %runs% -blockSize 16 -size 1024 1024 -samples 1 -input Scenes/spiral.txt
FastRelease\Stage4.exe -runs %runs% -blockSize 16 -size  256  256 -samples 1 -input Scenes/sponge.txt
