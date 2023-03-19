# Face Extraction from Videos

This program extracts faces from videos and saves them as individual images in an output directory. 
It uses OpenCV for face detection and Laplacian matrix sorting for quality control. 
You might find it helpful when training Stable Diffusion LoRA models.

## Usage

To run the program, use the following command:

```shell
python face.py [-h] [-s SAMPLE_RATE] [-n NUM_BEST] [-r RESOLUTION] input_dir output_dir
```

### Positional Arguments

- `input_dir`: input directory containing the video files.
- `output_dir`: output directory where the extracted face images will be saved.

### Options

- `-h`, `--help`: show the help message and exit.
- `-s SAMPLE_RATE`, `--sample-rate SAMPLE_RATE`: sample rate for frame sampling. Default is 6 (every 6 frames).
- `-n NUM_BEST`, `--num-best NUM_BEST`: max number of output images sorted using Laplacian matrix. Default is 20.
- `-r RESOLUTION`, `--resolution RESOLUTION`: resolution of the output image. Default is 512x512 pixels.

## Example

To extract faces from all video files in the `input` directory and save the output images in the ``output` directory with a resolution of 512x512 pixels and a sample rate of 2 (every 2nd frame), use the following command:

```shell
python face.py -r 512 -s 2 input output
```

## Requirements

```text
colorama==0.4.6
imutils==0.5.4
numpy==1.24.2
opencv-python==4.7.0.72
tqdm==4.65.0
```
Using slightly different versions should also be fine. 

## License
This project is licensed under the MIT License - see the `LICENSE` file for details.