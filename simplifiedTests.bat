Release\RayMarcherAss2Simplified.exe -runs 1 -blockSize 8 -size 8 8 -samples 1 -threads 1 -output Outputs/simplified_01.bmp -input Scenes/cornell.txt           
Release\RayMarcherAss2Simplified.exe -runs 1 -blockSize 8 -size 8 8 -samples 1 -threads 1 -output Outputs/simplified_02.bmp -input Scenes/juggler.txt 
Release\RayMarcherAss2Simplified.exe -runs 1 -blockSize 8 -size 8 8 -samples 1 -threads 1 -output Outputs/simplified_03.bmp -input Scenes/5000spheres.txt 

magick compare -metric mae Outputs\simplified_01.bmp Outputs_REFERENCE\stage0_01.bmp Outputs\simplifieddiff_01.bmp
magick compare -metric mae Outputs\simplified_02.bmp Outputs_REFERENCE\stage0_02.bmp Outputs\simplifieddiff_02.bmp
magick compare -metric mae Outputs\simplified_03.bmp Outputs_REFERENCE\stage0_03.bmp Outputs\simplifieddiff_03.bmp

Release\RayMarcherAss2Simplified.exe -runs 1 -blockSize 16 -size 256 256 -samples 1 -output Outputs/simplified_11.bmp -input Scenes/cornell.txt     
Release\RayMarcherAss2Simplified.exe -runs 1 -blockSize 16 -size 256 256 -samples 1 -output Outputs/simplified_12.bmp -input Scenes/juggler.txt 
Release\RayMarcherAss2Simplified.exe -runs 1 -blockSize 16 -size 256 256 -samples 1 -output Outputs/simplified_13.bmp -input Scenes/5000spheres.txt 

magick compare -metric mae Outputs\simplified_11.bmp Outputs_REFERENCE\stage0_11.bmp Outputs\simplifieddiff_11.bmp
magick compare -metric mae Outputs\simplified_12.bmp Outputs_REFERENCE\stage0_12.bmp Outputs\simplifieddiff_12.bmp
magick compare -metric mae Outputs\simplified_13.bmp Outputs_REFERENCE\stage0_13.bmp Outputs\simplifieddiff_13.bmp

Release\RayMarcherAss2Simplified.exe -runs 1 -blockSize 16 -size 256 256 -samples 2 -output Outputs/simplified_21.bmp -input Scenes/cornell.txt      
Release\RayMarcherAss2Simplified.exe -runs 1 -blockSize 16 -size 256 256 -samples 2 -output Outputs/simplified_22.bmp -input Scenes/juggler.txt 
Release\RayMarcherAss2Simplified.exe -runs 1 -blockSize 16 -size 256 256 -samples 2 -output Outputs/simplified_23.bmp -input Scenes/5000spheres.txt 
                                                                                
magick compare -metric mae Outputs\simplified_21.bmp Outputs_REFERENCE\stage0_21.bmp Outputs\simplifieddiff_21.bmp
magick compare -metric mae Outputs\simplified_22.bmp Outputs_REFERENCE\stage0_22.bmp Outputs\simplifieddiff_22.bmp
magick compare -metric mae Outputs\simplified_23.bmp Outputs_REFERENCE\stage0_23.bmp Outputs\simplifieddiff_23.bmp
