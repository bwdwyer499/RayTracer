Release\Stage5.exe -runs 1 -blockSize 8 -size 8 8 -samples 1 -threads 1 -output Outputs/stage5_01.bmp -input Scenes/cornell.txt           
Release\Stage5.exe -runs 1 -blockSize 8 -size 8 8 -samples 1 -threads 1 -output Outputs/stage5_02.bmp -input Scenes/juggler.txt 
Release\Stage5.exe -runs 1 -blockSize 8 -size 8 8 -samples 1 -threads 1 -output Outputs/stage5_03.bmp -input Scenes/5000spheres.txt 
Release\Stage5.exe -runs 1 -blockSize 8 -size 8 8 -samples 1 -threads 1 -output Outputs/stage5_04.bmp -input Scenes/all.txt 
Release\Stage5.exe -runs 1 -blockSize 8 -size 8 8 -samples 1 -threads 1 -output Outputs/stage5_05.bmp -input Scenes/spiral.txt
Release\Stage5.exe -runs 1 -blockSize 8 -size 8 8 -samples 1 -threads 1 -output Outputs/stage5_06.bmp -input Scenes/sponge.txt

magick compare -metric mae Outputs\stage5_01.bmp Outputs_REFERENCE\stage0_01.bmp Outputs\stage5diff_01.bmp
magick compare -metric mae Outputs\stage5_02.bmp Outputs_REFERENCE\stage0_02.bmp Outputs\stage5diff_02.bmp
magick compare -metric mae Outputs\stage5_03.bmp Outputs_REFERENCE\stage0_03.bmp Outputs\stage5diff_03.bmp
magick compare -metric mae Outputs\stage5_04.bmp Outputs_REFERENCE\stage0_04.bmp Outputs\stage5diff_04.bmp
magick compare -metric mae Outputs\stage5_05.bmp Outputs_REFERENCE\stage0_05.bmp Outputs\stage5diff_05.bmp
magick compare -metric mae Outputs\stage5_06.bmp Outputs_REFERENCE\stage0_06.bmp Outputs\stage5diff_06.bmp

Release\Stage5.exe -runs 1 -blockSize 16 -size 256 256 -samples 1 -output Outputs/stage5_11.bmp -input Scenes/cornell.txt     
Release\Stage5.exe -runs 1 -blockSize 16 -size 256 256 -samples 1 -output Outputs/stage5_12.bmp -input Scenes/juggler.txt 
Release\Stage5.exe -runs 1 -blockSize 16 -size 256 256 -samples 1 -output Outputs/stage5_13.bmp -input Scenes/5000spheres.txt 
Release\Stage5.exe -runs 1 -blockSize 16 -size 256 256 -samples 1 -output Outputs/stage5_14.bmp -input Scenes/all.txt 
Release\Stage5.exe -runs 1 -blockSize 16 -size 256 256 -samples 1 -output Outputs/stage5_15.bmp -input Scenes/spiral.txt
Release\Stage5.exe -runs 1 -blockSize 16 -size 256 256 -samples 1 -output Outputs/stage5_16.bmp -input Scenes/sponge.txt

magick compare -metric mae Outputs\stage5_11.bmp Outputs_REFERENCE\stage0_11.bmp Outputs\stage5diff_11.bmp
magick compare -metric mae Outputs\stage5_12.bmp Outputs_REFERENCE\stage0_12.bmp Outputs\stage5diff_12.bmp
magick compare -metric mae Outputs\stage5_13.bmp Outputs_REFERENCE\stage0_13.bmp Outputs\stage5diff_13.bmp
magick compare -metric mae Outputs\stage5_14.bmp Outputs_REFERENCE\stage0_14.bmp Outputs\stage5diff_14.bmp
magick compare -metric mae Outputs\stage5_15.bmp Outputs_REFERENCE\stage0_15.bmp Outputs\stage5diff_15.bmp
magick compare -metric mae Outputs\stage5_16.bmp Outputs_REFERENCE\stage0_16.bmp Outputs\stage5diff_16.bmp

Release\Stage5.exe -runs 1 -blockSize 16 -size 256 256 -samples 2 -output Outputs/stage5_21.bmp -input Scenes/cornell.txt      
Release\Stage5.exe -runs 1 -blockSize 16 -size 256 256 -samples 2 -output Outputs/stage5_22.bmp -input Scenes/juggler.txt 
Release\Stage5.exe -runs 1 -blockSize 16 -size 256 256 -samples 2 -output Outputs/stage5_23.bmp -input Scenes/5000spheres.txt 
Release\Stage5.exe -runs 1 -blockSize 16 -size 256 256 -samples 2 -output Outputs/stage5_24.bmp -input Scenes/all.txt 
Release\Stage5.exe -runs 1 -blockSize 16 -size 256 256 -samples 2 -output Outputs/stage5_25.bmp -input Scenes/spiral.txt
Release\Stage5.exe -runs 1 -blockSize 16 -size 256 256 -samples 2 -output Outputs/stage5_26.bmp -input Scenes/sponge.txt
                                                                                
magick compare -metric mae Outputs\stage5_21.bmp Outputs_REFERENCE\stage0_21.bmp Outputs\stage5diff_21.bmp
magick compare -metric mae Outputs\stage5_22.bmp Outputs_REFERENCE\stage0_22.bmp Outputs\stage5diff_22.bmp
magick compare -metric mae Outputs\stage5_23.bmp Outputs_REFERENCE\stage0_23.bmp Outputs\stage5diff_23.bmp
magick compare -metric mae Outputs\stage5_24.bmp Outputs_REFERENCE\stage0_24.bmp Outputs\stage5diff_24.bmp
magick compare -metric mae Outputs\stage5_25.bmp Outputs_REFERENCE\stage0_25.bmp Outputs\stage5diff_25.bmp
magick compare -metric mae Outputs\stage5_26.bmp Outputs_REFERENCE\stage0_26.bmp Outputs\stage5diff_26.bmp