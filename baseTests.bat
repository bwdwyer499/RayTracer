Release\RayMarcherAss2.exe -runs 1 -blockSize 8 -size 8 8 -samples 1 -threads 1 -output Outputs/base_01.bmp -input Scenes/cornell.txt           
Release\RayMarcherAss2.exe -runs 1 -blockSize 8 -size 8 8 -samples 1 -threads 1 -output Outputs/base_02.bmp -input Scenes/juggler.txt 
Release\RayMarcherAss2.exe -runs 1 -blockSize 8 -size 8 8 -samples 1 -threads 1 -output Outputs/base_03.bmp -input Scenes/5000spheres.txt 
Release\RayMarcherAss2.exe -runs 1 -blockSize 8 -size 8 8 -samples 1 -threads 1 -output Outputs/base_04.bmp -input Scenes/all.txt 
Release\RayMarcherAss2.exe -runs 1 -blockSize 8 -size 8 8 -samples 1 -threads 1 -output Outputs/base_05.bmp -input Scenes/spiral.txt
Release\RayMarcherAss2.exe -runs 1 -blockSize 8 -size 8 8 -samples 1 -threads 1 -output Outputs/base_06.bmp -input Scenes/sponge.txt

magick compare -metric mae Outputs\base_01.bmp Outputs_REFERENCE\stage0_01.bmp Outputs\basediff_01.bmp
magick compare -metric mae Outputs\base_02.bmp Outputs_REFERENCE\stage0_02.bmp Outputs\basediff_02.bmp
magick compare -metric mae Outputs\base_03.bmp Outputs_REFERENCE\stage0_03.bmp Outputs\basediff_03.bmp
magick compare -metric mae Outputs\base_04.bmp Outputs_REFERENCE\stage0_04.bmp Outputs\basediff_04.bmp
magick compare -metric mae Outputs\base_05.bmp Outputs_REFERENCE\stage0_05.bmp Outputs\basediff_05.bmp
magick compare -metric mae Outputs\base_06.bmp Outputs_REFERENCE\stage0_06.bmp Outputs\basediff_06.bmp

Release\RayMarcherAss2.exe -runs 1 -blockSize 16 -size 256 256 -samples 1 -output Outputs/base_11.bmp -input Scenes/cornell.txt     
Release\RayMarcherAss2.exe -runs 1 -blockSize 16 -size 256 256 -samples 1 -output Outputs/base_12.bmp -input Scenes/juggler.txt 
Release\RayMarcherAss2.exe -runs 1 -blockSize 16 -size 256 256 -samples 1 -output Outputs/base_13.bmp -input Scenes/5000spheres.txt 
Release\RayMarcherAss2.exe -runs 1 -blockSize 16 -size 256 256 -samples 1 -output Outputs/base_14.bmp -input Scenes/all.txt 
Release\RayMarcherAss2.exe -runs 1 -blockSize 16 -size 256 256 -samples 1 -output Outputs/base_15.bmp -input Scenes/spiral.txt
Release\RayMarcherAss2.exe -runs 1 -blockSize 16 -size 256 256 -samples 1 -output Outputs/base_16.bmp -input Scenes/sponge.txt

magick compare -metric mae Outputs\base_11.bmp Outputs_REFERENCE\stage0_11.bmp Outputs\basediff_11.bmp
magick compare -metric mae Outputs\base_12.bmp Outputs_REFERENCE\stage0_12.bmp Outputs\basediff_12.bmp
magick compare -metric mae Outputs\base_13.bmp Outputs_REFERENCE\stage0_13.bmp Outputs\basediff_13.bmp
magick compare -metric mae Outputs\base_14.bmp Outputs_REFERENCE\stage0_14.bmp Outputs\basediff_14.bmp
magick compare -metric mae Outputs\base_15.bmp Outputs_REFERENCE\stage0_15.bmp Outputs\basediff_15.bmp
magick compare -metric mae Outputs\base_16.bmp Outputs_REFERENCE\stage0_16.bmp Outputs\basediff_16.bmp

Release\RayMarcherAss2.exe -runs 1 -blockSize 16 -size 256 256 -samples 2 -output Outputs/base_21.bmp -input Scenes/cornell.txt      
Release\RayMarcherAss2.exe -runs 1 -blockSize 16 -size 256 256 -samples 2 -output Outputs/base_22.bmp -input Scenes/juggler.txt 
Release\RayMarcherAss2.exe -runs 1 -blockSize 16 -size 256 256 -samples 2 -output Outputs/base_23.bmp -input Scenes/5000spheres.txt 
Release\RayMarcherAss2.exe -runs 1 -blockSize 16 -size 256 256 -samples 2 -output Outputs/base_24.bmp -input Scenes/all.txt 
Release\RayMarcherAss2.exe -runs 1 -blockSize 16 -size 256 256 -samples 2 -output Outputs/base_25.bmp -input Scenes/spiral.txt
Release\RayMarcherAss2.exe -runs 1 -blockSize 16 -size 256 256 -samples 2 -output Outputs/base_26.bmp -input Scenes/sponge.txt
                                                                                
magick compare -metric mae Outputs\base_21.bmp Outputs_REFERENCE\stage0_21.bmp Outputs\basediff_21.bmp
magick compare -metric mae Outputs\base_22.bmp Outputs_REFERENCE\stage0_22.bmp Outputs\basediff_22.bmp
magick compare -metric mae Outputs\base_23.bmp Outputs_REFERENCE\stage0_23.bmp Outputs\basediff_23.bmp
magick compare -metric mae Outputs\base_24.bmp Outputs_REFERENCE\stage0_24.bmp Outputs\basediff_24.bmp
magick compare -metric mae Outputs\base_25.bmp Outputs_REFERENCE\stage0_25.bmp Outputs\basediff_25.bmp
magick compare -metric mae Outputs\base_26.bmp Outputs_REFERENCE\stage0_26.bmp Outputs\basediff_26.bmp