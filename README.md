## Quad----

Image compression based on quadtrees.

The program targets an input image. The input image is split into four quadrants. Each quadrant is assigned an averaged color based on the colors in the input image. The quadrant with the largest error is split into its four children quadrants to refine the image. This process is repeated N times.

### Architecture
1. I develop a front-end GUI using `Python3`, with a `C++` back-end implementing a Quadtree library.
2.  `Pybind11` is used for integrating the front-end and back-end. Utilize `OpenCV` for image cropping and drawing.
3. The GUI is implemented using `TKinter`, and a `Makefile` is written for convenient maintenance of the program.
### Animation

The first animation shows the natural iterative process of the algorithm.
| ![Animation](http://i.imgur.com/UE2eOkx.gif) |
|:-----------------------------------:|
| **Animation 1** |


The second animation shows a top-down, breadth-first traversal of the final quadtree.
| ![Animation](http://i.imgur.com/l3sv0In.gif) |
|:-----------------------------------:|
| **Animation 2** |

### GUI
The followomg picture is my GUI design. Here is some explanation about it:
1. The `ITERATION` parameter controls the number of quadtree nodes.
2. `DRAW_MODE` determines the pattern drawn for each quadtree leaf. For example, the line in the GUI below illustrates the rectangle mode.
3. `STATIC` displays the statistics of this image processing.
4. You can load an input image and download the processed image using the respective buttons."

| ![image](https://github.com/CodeStone1125/Quad----/assets/72511296/613d0223-8e51-4f73-923e-bf3e99304ae4) |
|:-----------------------------------:|
| **GUI** |


### Samples
| ![image](https://github.com/CodeStone1125/Quad----/assets/72511296/19341642-50be-41cb-8aa8-814aa4ac3508) |
|:-----------------------------------:|
| **Output(left) vs Input(right)** |


| ![image](https://github.com/CodeStone1125/Quad----/assets/72511296/4d12decc-4e64-4b97-aefa-53d4a529c6b4) |
|:-----------------------------------:|
| **Patrick star(ellipse)** |
