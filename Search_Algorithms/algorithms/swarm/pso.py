import numpy as np
from algorithms.base.optimizer_base import Optimizer

class PSO(Optimizer):
    def run(self):
        # Config
        n_particles = self.config.get("n_particles", 30)
        w = self.config.get("w", 0.7)    # Inertia weight
        c1 = self.config.get("c1", 1.5)  # Cognitive (Personal)
        c2 = self.config.get("c2", 1.5)  # Social (Global)
        max_iters = self.config.get("max_iters", 1000)

        dim = self.problem.get_dimension()
        low, high = self.problem.get_bounds()

        # Init Swarm
        particles_x = np.random.uniform(low, high, (n_particles, dim))
        particles_v = np.random.uniform(-1, 1, (n_particles, dim)) # Vận tốc khởi tạo nhỏ
        
        # PBest & GBest
        pbest_x = particles_x.copy()
        pbest_f = np.full(n_particles, float('inf'))
        
        gbest_x = None
        gbest_f = float('inf')

        # Đánh giá khởi tạo
        for i in range(n_particles):
            f = self.evaluate(particles_x[i])
            if f is None: return self._build_result()
            
            pbest_f[i] = f
            if f < gbest_f:
                gbest_f = f
                gbest_x = particles_x[i].copy()

        self.best_solution = gbest_x
        self.best_fitness = gbest_f
        self.history.append(gbest_f)

        # Loop
        for iteration in range(max_iters):
            if self.evaluations >= self.max_evals:
                break

            # Vector hóa tính toán vận tốc cho nhanh (NumPy power)
            r1 = np.random.rand(n_particles, dim)
            r2 = np.random.rand(n_particles, dim)

            # v_new = w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x)
            particles_v = (w * particles_v) + \
                          (c1 * r1 * (pbest_x - particles_x)) + \
                          (c2 * r2 * (gbest_x - particles_x)) # gbest_x broadcast xuống

            # Velocity clamping to prevent explosion
            v_max = self.config.get("v_max", (high - low) * 0.5)
            particles_v = np.clip(particles_v, -v_max, v_max)

            # Update Position
            particles_x = particles_x + particles_v
            
            # Clip bounds (Phản xạ hoặc gán biên - ở đây dùng gán biên)
            particles_x = np.clip(particles_x, low, high)

            # Evaluate
            for i in range(n_particles):
                f = self.evaluate(particles_x[i])
                if f is None: break

                # Update Personal Best
                if f < pbest_f[i]:
                    pbest_f[i] = f
                    pbest_x[i] = particles_x[i].copy()

                    # Update Global Best
                    if f < gbest_f:
                        gbest_f = f
                        gbest_x = particles_x[i].copy()

            self.best_fitness = gbest_f
            self.best_solution = gbest_x
            self.history.append(gbest_f)

        return self._build_result()