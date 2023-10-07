\# 2D Collision Detector

\#\# Basic Information

Simulate how different types of shapes collide in a 2D map.

-   GitHub repository:
    \[<https://github.com/CodeStone1125/2DCollisionDetector>\](<https://github.com/CodeStone1125/2DCollisionDetector>)

\#\# Problem to Solve

The problem this system aims to solve is to calculate collisions between
the most common shapes in a 2D map, such as rectangles, circles, and
convex polygons.

There are several algorithms that can be implemented to solve this
problem. Before delving deeper into the topic, let\'s first discuss the
\"Axis-Aligned Bounding Box\" (AABB) algorithm. While it\'s not a single
algorithm but a framework for collision detection, it serves as the
basis for many collision detection systems in 2D graphics and game
development due to its simplicity and efficiency.

-   Bounding Boxes: Each object in a 2D scene is enclosed by a rectangle
    (a bounding box) aligned with the axes of the coordinate system.
    These rectangles are often referred to as \"AABBs\" because they are
    axis-aligned.

However, a 2D map may not only contain rectangles and circles but also
various types of polygons. To address this, objects are divided into two
types: Convex Polygons and concave polygons. The Delaunay Triangulation
and Hertel-Mehlhorn algorithm are used to improve efficiency.

\* Delauney Triangulation: The Delauney sweepline triangulation
algorithm provides a triangulation with the maximum minimum internal
angle. In simpler terms, it produces a triangulation with fewer thin
strips. Implementing Delaunay triangulation from scratch is quite
complex, typically involving advanced algorithms like the Bowyer-Watson
algorithm or incremental algorithms.

-   Hertel-Mehlhorn: The simplest approach to this is the
    Hertel-Mehlhorn algorithm, which promises to produce no more than 4
    times the number of polygons of the optimal solution. In practice,
    for simple concave polygons, this algorithm often produces the
    optimal solution.

The algorithm is straightforward: it iterates over the internal edges
(diagonals) created by triangulation and removes non-essential
diagonals. A diagonal is considered non-essential if, at either end of
the diagonal, the linked points would form a convex shape. This is
determined by testing the orientation of the points.

\#\# Prospective Users

-   Game Developers: Determining whether objects in a game world are in
    contact with a character is common in game development. Some games
    even make this a central feature, like \[Super
    Mario\](<https://en.wikipedia.org/wiki/Super_Mario>) and
    \[Pac-Man\](<https://en.wikipedia.org/wiki/Pac-Man>).

| !\[Super Mario\](./pictures/Mario.png) \|
  !\[Pac-Man\](./pictures/pacman.png) \|

:\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\--:\|
\| **Super Mario**

-   Collision Prevention for Automation Equipment: Robots and automation
    devices used in industrial automation need to ensure that they do
    not collide or interfere with each other while performing tasks. 2D
    collision detection can monitor the positions of individual machine
    components to prevent unnecessary collisions, enhance production
    efficiency, and safeguard equipment.

\#\# System Architecture

The system should provide a graphical user interface that allows users
to design a 2D map, place walls, and move a light source to interact
with the environment. Additionally, when the user interacts with any
objects on the map, the system needs to highlight the area reachable
from the light source in real-time.

| !\[among-us\](./assets/system\_arch.jpg) \|

\| **System Flow Chart** \|

\#\# API Description

-   Python API:
    -   \`addWall(start, end)\`: Create a wall on the map and return its
        ID.
    -   \`rmWall(wall\_id)\`: Remove the wall by its ID.
    -   \`moveLightSource(dest)\`: Move the light source to the
        destination coordinate.
    -   \`lightArea(walls, light\_source)\`: Call the C++ API and
        display the lit area on the map.
-   C++ API:
    -   \`lightArea(walls, light\_source)\`: Calculate all the areas
        where light from a given single light source can reach on a 2D
        map and return the coordinates of the lit places and their area.

\#\# Engineering Infrastructure

-   Automatic build system: [CMake]{.title-ref}
-   Version control: [Git]{.title-ref}
-   Testing framework: [Pytest]{.title-ref}
-   Documentation: GitHub [README.md]{.title-ref}

\#\# Schedule

Planning phase (6 weeks from 9/19 to 10/31): Setup the environment and
become familiar with the algorithm.

Week 1 (10/31): Implement the algorithm with C++.

Week 2 (11/7): Create Python wrappers for C++ with [pybind]{.title-ref}.

Week 3 (11/14): Finish C++, and start creating the interactive map in
Python.

Week 4 (11/21): Implement features of the interactive map.

Week 5 (11/28): Test all features with [Pytest]{.title-ref}.

Week 6 (12/5): Finish up, debug, and write the documentation.

Week 7 (12/12): Buffer time for further testing and debugging.

Week 8 (12/19): Prepare slides and materials for the presentation.

\#\# References

-   \[Red Blob
    Games\](<https://www.redblobgames.com/articles/visibility/>)
-   \[Sight and Light\](<https://ncase.me/sight-and-light/>)
