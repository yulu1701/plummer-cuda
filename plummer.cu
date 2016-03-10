#include <stdio.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

// Definitions
#define NUM_PARTICLES 10000
#define GM (1.0/NUM_PARTICLES)
#define PI 3.14159265
#define BLOCK_SIZE 256

// Structs
typedef struct { double x, y, z; } vector3;
typedef struct { vector3 *p, *v; } particles_t;

// Headers
double unirandom();
void new_particle(vector3 *p, vector3 *v);
__global__ void integrate_position(vector3 *p, vector3 *v, double dt);
__global__ void leapint(vector3 *p, vector3 *v, double dt);
__global__ void print_position(vector3 *p);

// Main function
int main(const int argc, const char** argv) {
  int num_blocks = (NUM_PARTICLES + BLOCK_SIZE - 1) / BLOCK_SIZE;
  double dt = 0.1;
  int nstep = 120;
  int nout = 1;

  int num_bytes = 2 * NUM_PARTICLES * sizeof(vector3);
  double *host_memory = (double *) malloc(num_bytes);
  double *device_memory;
  cudaMalloc(&device_memory, num_bytes);

  // Assign host memory
  particles_t host_particles;
  host_particles.p = (vector3*) host_memory;
  host_particles.v = ((vector3*) host_memory) + NUM_PARTICLES;
  // Assign device memory
  particles_t device_particles;
  device_particles.p = (vector3*) device_memory;
  device_particles.v = ((vector3*) device_memory) + NUM_PARTICLES;

  // Initialize bodies
  srand(time(0));
  for (int i = 0; i < NUM_PARTICLES; i++) {
    new_particle(&host_particles.p[i], &host_particles.v[i]);
  }

  // Needed to print in paralel
  cudaDeviceSetLimit(cudaLimitPrintfFifoSize, NUM_PARTICLES * 400);
  // Copy host to device
  cudaMemcpy(device_memory, host_memory, num_bytes, cudaMemcpyHostToDevice);

  // Loop leapfrog
  for (int i = 0; i < nstep; i++) {
    if (i % nout == 0) {
      print_position<<<num_blocks, BLOCK_SIZE>>>(device_particles.p);
      cudaDeviceSynchronize();
    }

    leapint<<<num_blocks, BLOCK_SIZE>>>(device_particles.p, device_particles.v, dt);
    cudaDeviceSynchronize();
    integrate_position<<<num_blocks, BLOCK_SIZE>>>(device_particles.p, device_particles.v, dt);
    cudaDeviceSynchronize();
    leapint<<<num_blocks, BLOCK_SIZE>>>(device_particles.p, device_particles.v, dt);
    cudaDeviceSynchronize();
  }
  if (nstep % nout == 0) {
    print_position<<<num_blocks, BLOCK_SIZE>>>(device_particles.p);
    cudaDeviceSynchronize();
  }

  free(host_memory);
  cudaFree(device_memory);
}

double unirandom() {
  return ((double) rand())/RAND_MAX;
}

void new_particle(vector3 *p, vector3 *v) {
  double X1 = unirandom();
  double r = pow(pow(X1, -2.0/3) - 1, -0.5);
  double X2 = unirandom();
  double X3 = unirandom();
  p->z = (1 - 2 * X2) * r;
  p->x = sqrt((r * r - p->z * p->z)) * cos(2 * PI * X3);
  p->y = sqrt((r * r - p->z * p->z)) * sin(2 * PI * X3);

  #ifdef STABLE
    double V_e = sqrt(2) * pow((1 + r * r), -0.25);
    double X4 = unirandom();
    double X5 = unirandom();
    while ((0.1 * X5) >= (X4 * X4 * pow((1 - X4 * X4), 3.5))) {
      X4 = unirandom();
      X5 = unirandom();
    }
    double V = X4 * V_e;
    double X6 = unirandom();
    double X7 = unirandom();
    v->z = (1 - 2 * X6) * V;
    v->x = sqrt((V * V - v->z * v->z)) * cos(2 * PI * X7);
    v->y = sqrt((V * V - v->z * v->z)) * sin(2 * PI * X7);
  #else
    (*v) = (vector3) {0.0, 0.0, 0.0};
  #endif
}

// Integrate position
__global__ void integrate_position(vector3 *p, vector3 *v, double dt) {
  unsigned long i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < NUM_PARTICLES) {
    p[i] = (vector3) {p[i].x + v[i].x * dt, p[i].y + v[i].y * dt, p[i].z + v[i].z * dt};
  }
}

// Leapfrog integrator
__global__ void leapint(vector3 *p, vector3 *v, double dt) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < NUM_PARTICLES) {
    double ax = 0.0;
    double ay = 0.0;
    double az = 0.0;

    for (int j = 0; j < NUM_PARTICLES; j++) {
      if(i == j) {
        continue;
      }
      double dx = p[j].x - p[i].x;
      double dy = p[j].y - p[i].y;
      double dz = p[j].z - p[i].z;
      double accel = pow(rsqrt(dx * dx + dy * dy + dz * dz), 3);
      ax += dx * accel;
      ay += dy * accel;
      az += dz * accel;
    }

    v[i] = (vector3) {v[i].x + 0.5 * GM * dt * ax, v[i].y + 0.5 * GM * dt * ay, v[i].z + 0.5 * GM * dt * az};
  }
}

// Print positions
__global__ void print_position(vector3 *p) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < NUM_PARTICLES) {
    printf("%lu,%.6lf,%.6lf,%.6lf\n", i, p[i].x, p[i].y, p[i].z);
  }
}
