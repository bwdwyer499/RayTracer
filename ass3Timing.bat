@ECHO OFF
set runs=%1
if "%1"=="" set runs=1
@ECHO ON

@echo -------------------------------------------------------------
@echo CAUTION: THESE TIMING TESTS WILL TAKE QUITE A LONG TIME
@echo see the assignment spec for workarounds to speed things up
@echo -------------------------------------------------------------

Release\Stage5.exe -runs %runs% -size 1024 1024 -samples 1  -output Outputs/a03s00btiming01.bmp -input Scenes/all.txt     
Release\Stage5.exe -runs %runs% -size 1024 1024 -samples 4  -output Outputs/a03s00btiming02.bmp -input Scenes/all.txt     
Release\Stage5.exe -runs %runs% -size 1024 1024 -samples 16 -output Outputs/a03s00btiming03.bmp -input Scenes/all.txt     
Release\Stage5.exe -runs %runs% -size 1024 1024 -samples 1  -output Outputs/a03s00btiming04.bmp -input Scenes/cornell.txt     
Release\Stage5.exe -runs %runs% -size 1024 1024 -samples 1  -output Outputs/a03s00btiming05.bmp -input Scenes/spiral.txt
Release\Stage5.exe -runs %runs% -size  720 1280 -samples 1  -output Outputs/a03s00btiming06.bmp -input Scenes/juggler.txt 
Release\Stage5.exe -runs %runs% -size 1024 1024 -samples 1  -output Outputs/a03s00btiming07.bmp -input Scenes/sponge.txt
Release\Stage5.exe -runs %runs% -size 1280  720 -samples 1  -output Outputs/a03s00btiming08.bmp -input Scenes/5000spheres.txt 

