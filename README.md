# cuda-blackhole
![Black Hole](assets/blackhole.jpg)

## Engulf Anything
Render a [Schwarzchild black hole](https://en.wikipedia.org/wiki/Schwarzschild_metric) onto any equirectangular image in real-time. Rendering respects physics using numerical integration of geodesic equations and raytracing.

## How it Works
1. A camera and black hole are simulated in a scene.
2. Each GPU thread shoots a light ray out into space.
3. As the light rays travel near the black hole, they "accelerate" and follow geodesic paths through the warped spacetime. This is simulated from the geodesic equations using numerical integration (Runge-Kutta methods).
4. As light rays leave the warped region of spacetime and settle, they travel to infinity and point sample from an infinitely-far equirectangular image.

## Performance Benchmark
```
T4 GPU 1080p: 300ms
```

