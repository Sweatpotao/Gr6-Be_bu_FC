# Kế hoạch Xây dựng Biểu đồ cho Bài toán RờI rạc

## Tổng quan

Xây dựng các biểu đồ visualization cho thuật toán tối ưu rờI rạc bao gồm:
1. So sánh hiệu năng tổng thể
2. Phân tích độ nhạy tham số
3. Biểu đồ 3D bề mặt hàm mục tiêu

## Các bài toán hỗ trợ
- TSP (Traveling Salesman Problem)
- Knapsack (Bài toán cái túi)
- N-Queens
- Grid Pathfinding

## Thuật toán Metaheuristic
- GA_TSP (Genetic Algorithm)
- HillClimbingTSP
- SimulatedAnnealingTSP
- ABC_Knapsack (Artificial Bee Colony)
- TLBO_Knapsack (Teaching-Learning-Based Optimization)
- ACO_Discrete (Ant Colony Optimization)

---

## 1. Biểu đồ So sánh Hiệu năng Tổng thể

### 1.1 Các tiêu chí đánh giá

| Tiêu chí | Mô tả | Metrics |
|----------|-------|---------|
| Chất lượng nghiệm tốt nhất | Best score đạt được | `best_score` |
| Chất lượng nghiệm trung bình | Mean score qua nhiều lần chạy | `mean_score` |
| Độ phức tạp tính toán | ThờI gian và số lần đánh giá | `mean_time`, `mean_effort` |
| Độ ổn định | Độ lệch chuẩn và success rate | `std_score`, `success_rate` |
| Khả năng mở rộng | Hiệu năng với kích thước bài toán tăng | scalability test |
| Cân bằng khai phá-khai thác | Exploration vs Exploitation | convergence analysis |

### 1.2 Các loại biểu đồ

#### a) Bar Chart Comparison
- So sánh trực tiếp các metrics giữa các thuật toán
- Grouped bar chart cho nhiều tiêu chí
- Horizontal bar cho readability

#### b) Box Plot
- Phân bố kết quả qua nhiều lần chạy
- Thể hiện median, quartiles, outliers
- Đánh giá độ ổn định

#### c) Radar Chart (đã có - mở rộng)
- Thêm metrics mới: scalability, exploration-exploitation balance
- Tích hợp vào existing radar chart

#### d) Convergence Plot
- Tiến trình tối ưu theo thế hệ/lần lặp
- So sánh tốc độ hội tụ

---

## 2. Phân tích Độ nhạy Tham số

### 2.1 Tham số cần phân tích theo thuật toán

#### GA_TSP
| Tham số | Range | Ý nghĩa |
|---------|-------|---------|
| population_size | [20, 50, 100, 200] | Kích thước quần thể |
| mutation_rate | [0.01, 0.05, 0.1, 0.2] | Tỷ lệ đột biến |
| crossover_rate | [0.6, 0.7, 0.8, 0.9] | Tỷ lệ lai ghép |
| max_iters | [100, 500, 1000, 2000] | Số thế hệ tối đa |

#### SimulatedAnnealingTSP
| Tham số | Range | Ý nghĩa |
|---------|-------|---------|
| initial_temp | [100, 500, 1000, 2000] | Nhiệt độ ban đầu |
| cooling_rate | [0.95, 0.98, 0.995, 0.999] | Tốc độ làm nguội |
| max_iterations | [1000, 5000, 10000] | Số lần lặp tối đa |

#### ABC_Knapsack
| Tham số | Range | Ý nghĩa |
|---------|-------|---------|
| num_employed_bees | [20, 50, 100] | Số ong thợ |
| limit | [50, 100, 200] | Giới hạn scout |
| max_cycles | [100, 500, 1000] | Số chu kỳ tối đa |

#### TLBO_Knapsack
| Tham số | Range | Ý nghĩa |
|---------|-------|---------|
| population_size | [20, 50, 100] | Kích thước lớp học |
| max_iterations | [100, 500, 1000] | Số lần lặp |

### 2.2 Các loại biểu đồ phân tích độ nhạy

#### a) Line Plot (1D Sensitivity)
- Thay đổi 1 tham số, giữ cố định các tham số khác
- X-axis: giá trị tham số
- Y-axis: best_score hoặc mean_score

#### b) Heatmap (2D Sensitivity)
- Thay đổi 2 tham số đồng thờI
- Color: best_score hoặc mean_score
- X, Y: 2 tham số khác nhau

#### c) Contour Plot
- Đường đồng mức của hiệu năng
- Thể hiện vùng tham số tối ưu

---

## 3. Biểu đồ 3D Bề mặt Hàm mục tiêu

### 3.1 Mục đích
- Trực quan hóa không gian tìm kiếm
- Hiểu cấu trúc bài toán (local minima, global minimum)
- So sánh độ phức tạp của các bài toán

### 3.2 Cách thức hiển thị cho bài toán rờI rạc

Vì bài toán rờI rạc có không gian không liên tục, ta sẽ:
- TSP: Hiển thị fitness landscape theo 2 chiều (ví dụ: hoán vị dạng 2D)
- Knapsack: Heatmap 3D của giá trị theo tổ hợp vật phẩm
- N-Queens: Visualization của conflicts theo vị trí quân hậu

### 3.3 Các loại biểu đồ
- 3D Surface Plot (cho continuous approximation)
- 3D Scatter Plot (cho discrete points)
- Parallel Coordinates Plot (cho high-dimensional)

---

## 4. Cấu trúc Module

```
Search_Algorithms/visualization/
├── __init__.py                    # Export functions
├── base.py                        # Shared utilities (existing)
├── discrete_comparison.py         # Existing spider charts
├── continuous_comparison.py       # Existing radar charts
├── performance_charts.py          # NEW: Bar, box, convergence
├── sensitivity_analysis.py        # NEW: Parameter sensitivity
└── objective_surface_3d.py        # NEW: 3D surface plots
```

---

## 5. API Design

### 5.1 Performance Charts
```python
# Bar chart so sánh
def plot_performance_bar(results_dict, metrics, save_path=None)

# Box plot phân bố
def plot_performance_box(results_dict, save_path=None)

# Convergence plot
def plot_convergence_comparison(histories_dict, save_path=None)

# Combined dashboard
def create_performance_dashboard(results_dict, problem_name, output_dir)
```

### 5.2 Sensitivity Analysis
```python
# Line plot 1D sensitivity
def plot_parameter_sensitivity_1d(param_name, param_values, 
                                   results, save_path=None)

# Heatmap 2D sensitivity
def plot_parameter_sensitivity_2d(param1_name, param1_values,
                                   param2_name, param2_values,
                                   results_matrix, save_path=None)

# Full sensitivity analysis
def run_sensitivity_analysis(algorithm_class, problem, 
                             param_ranges, output_dir)
```

### 5.3 3D Surface
```python
# 3D surface cho TSP
def plot_tsp_objective_surface(distance_matrix, save_path=None)

# 3D surface cho Knapsack
def plot_knapsack_objective_surface(weights, values, capacity, save_path=None)

# General 3D visualization
def plot_discrete_objective_3d(problem, sample_points=1000, save_path=None)
```

---

## 6. Triển khai

### Phase 1: Performance Charts
- Tạo `performance_charts.py`
- Implement bar chart, box plot
- Implement convergence plot

### Phase 2: Sensitivity Analysis
- Tạo `sensitivity_analysis.py`
- Implement 1D sensitivity
- Implement 2D heatmap
- Tạo runner script

### Phase 3: 3D Visualization
- Tạo `objective_surface_3d.py`
- Implement 3D surface plots
- Adapter cho từng bài toán

### Phase 4: Integration
- Cập nhật `__init__.py`
- Tạo demo script
- Testing

---

## 7. Dependencies

```
matplotlib >= 3.5.0
seaborn >= 0.11.0
numpy >= 1.21.0
pandas >= 1.3.0  (optional, for data manipulation)
```
