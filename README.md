# Parallel Black Hole Rendering with CUDA
![Black Hole](assets/blackhole.jpg)
*Background image courtesy of [Greg Zaal](https://polyhaven.com/a/rogland_clear_night)*

## Engulf Anything
Render a [Schwarzchild black hole](https://en.wikipedia.org/wiki/Schwarzschild_metric) onto any equirectangular image in real-time. Rendering respects physics using numerical integration of geodesic equations and raytracing.

## How to Use It
1. Download any equirectangular image.
2. Point to the downloaded image path in `config.txt`.
3. Build and run the executable.
```
make setup && make
./blackhole config.txt
```
4. See your rendered image at the output path.

## How it Works
1. A camera and black hole are simulated in a scene.
2. Each GPU thread shoots a light ray out into space.
3. As the light rays travel near the black hole, they "accelerate" and follow geodesics through warped spacetime. This is simulated from the geodesic equations using numerical integration (Runge-Kutta methods).
4. As light rays leave the warped region of spacetime and settle, they travel to infinity and point sample from an infinitely-far equirectangular image.

## Performance Benchmark
This is the actual runtime to generate the same image you see in the readme. Averaged over 100 renders.

| Resolution | Kernel Runtime (ms) |  FPS |
| ---------- | ------------------- | ---- |
| 1080p      |               13.38 | 74.7 |
| 4K         |               55.94 | 17.9 |

The following parameters were used for the test.

| Param | Value |
|-------|-------|
| GPU   | NVIDIA L4 |
| Steps| 1000 |
| Step Size  |  0.1 $r_s$ |

*Note: $r_s$ is the Schwarzchild radius of the black hole*
