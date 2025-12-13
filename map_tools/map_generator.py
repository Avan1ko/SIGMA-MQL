"""
Map Generator for SIGMA and MQL
All maps use consistent format: 0 = free space, 1 = obstacle
"""

import numpy as np
import random
import skimage.measure
from skimage import morphology
from enum import Enum
from typing import Tuple, List, Optional


class MapType(Enum):
    """Different types of maps that can be generated"""
    # Original types (renamed from SIGMA)
    MAZE = "maze"
    HOUSE = "house"
    WAREHOUSE = "warehouse"
    RANDOM = "random"

    # MQL additional types - reduced to well-differentiated set
    TUNNELS = "tunnels"

class MapGenerator:
    """
    Generator for all map types.
    All maps use consistent format: 0 = free space, 1 = obstacle
    """
    
    def __init__(self, min_size: int = 8, max_size: int = 20):
        self.min_size = min_size
        self.max_size = max_size
    
    def generate(self, map_type: MapType, size: Optional[int] = None, **kwargs) -> np.ndarray:
        """
        Generate a map of specified type.
        
        Args:
            map_type: Type of map to generate
            size: Map size (if None, random between min_size and max_size)
            **kwargs: Additional parameters for specific map types
        
        Returns:
            Binary map where 0 = free space, 1 = obstacle
        """
        if size is None:
            size = np.random.randint(self.min_size, self.max_size + 1)
        
        generator_map = {
            # Original generators
            MapType.MAZE: self._generate_maze,
            MapType.HOUSE: self._generate_house,
            MapType.WAREHOUSE: self._generate_warehouse,
            MapType.RANDOM: self._generate_random,
            
            # MQL generators - reduced set
            MapType.TUNNELS: self._generate_tunnels,
        }
        
        if map_type not in generator_map:
            raise ValueError(f"Unknown map type: {map_type}")
        
        grid_map = generator_map[map_type](size, **kwargs)
        # Ensure all white cells are pairwise reachable
        return self.ensure_connectivity(grid_map)
    
    def ensure_connectivity(self, grid_map: np.ndarray) -> np.ndarray:
        """
        Ensure ALL white cells (free space) are pairwise reachable.
        Connects all isolated components to ensure full connectivity.
        """
        labeled, num_features = skimage.measure.label(1 - grid_map, connectivity=2, return_num=True)
        
        if num_features == 0:
            # All obstacles, create a small free space
            center = grid_map.shape[0] // 2
            grid_map[center, center] = 0
            return grid_map
        
        if num_features == 1:
            return grid_map
        
        # Find all connected components
        components = []
        for label_id in range(1, num_features + 1):
            component = np.argwhere(labeled == label_id)
            if len(component) > 0:
                components.append(component)
        
        # Sort by size, largest first
        components.sort(key=lambda x: len(x), reverse=True)
        main_component = components[0]
        
        # Connect all other components to the main one
        for component in components[1:]:
            # Find closest points between main and this component
            min_dist = float('inf')
            best_main_point = None
            best_other_point = None
            
            for mp in main_component:
                for op in component:
                    dist = abs(mp[0] - op[0]) + abs(mp[1] - op[1])
                    if dist < min_dist:
                        min_dist = dist
                        best_main_point = mp
                        best_other_point = op
            
            # Create L-shaped path between them
            if best_main_point is not None and best_other_point is not None:
                x1, y1 = best_main_point[0], best_main_point[1]
                x2, y2 = best_other_point[0], best_other_point[1]
                
                # Horizontal then vertical
                for x in range(min(x1, x2), max(x1, x2) + 1):
                    if 0 <= x < grid_map.shape[0]:
                        grid_map[x, y1] = 0
                for y in range(min(y1, y2), max(y1, y2) + 1):
                    if 0 <= y < grid_map.shape[1]:
                        grid_map[x2, y] = 0
        
        return grid_map
    
    def get_map_statistics(self, grid_map: np.ndarray) -> dict:
        """Get statistics about a generated map"""
        free_space = np.sum(grid_map == 0)
        obstacles = np.sum(grid_map == 1)
        total = grid_map.size
        
        labeled, num_components = skimage.measure.label(1 - grid_map, connectivity=2, return_num=True)
        
        return {
            'size': grid_map.shape[0],
            'free_space_ratio': free_space / total,
            'obstacle_ratio': obstacles / total,
            'num_connected_components': num_components,
            'density': obstacles / total
        }
    
    def _generate_random(self, size: int, sparsity_level: Optional[float] = None) -> np.ndarray:
        """SIGMA random generator with sparsity levels
        
        Args:
            size: Map size
            sparsity_level: A float (0.0 to 1.0) representing obstacle density value.
                          If None, defaults to 0.2 (20%)
        """
        if sparsity_level is None:
            obstacle_density = 0.2
        else:
            # Direct float value: treat as obstacle density
            obstacle_density = max(0.0, min(1.0, sparsity_level))  # Clamp to [0, 1]
        
        # Generate random obstacles
        grid_map = (np.random.rand(size, size) < obstacle_density).astype(np.int32)
        
        # Ensure all white cells (free space) are connected
        labeled, num_features = skimage.measure.label(1 - grid_map, connectivity=2, return_num=True)
        
        if num_features > 1:
            # Find the largest connected component
            component_sizes = []
            for label_id in range(1, num_features + 1):
                size_component = np.sum(labeled == label_id)
                component_sizes.append((label_id, size_component))
            
            component_sizes.sort(key=lambda x: x[1], reverse=True)
            largest_label = component_sizes[0][0]
            
            # Keep only the largest component, fill others with obstacles
            grid_map = (labeled != largest_label).astype(np.int32)
        
        # If all cells are obstacles, create a small free space
        if np.sum(grid_map == 0) == 0:
            center = size // 2
            grid_map[center, center] = 0
            if center + 1 < size:
                grid_map[center + 1, center] = 0
            if center - 1 >= 0:
                grid_map[center - 1, center] = 0
        
        return grid_map
    
    def _generate_maze(self, size: int, wall_components: Tuple[int, int] = (1, 8),
                             obstacle_density: Optional[Tuple[float, float]] = None,
                             go_straight: float = 0.8) -> np.ndarray:
        """Maze generator - original maze algorithm, returns exact size"""
        if obstacle_density is None:
            obstacle_density = (0.0, 1.0)
        
        min_component, max_component = wall_components
        num_components = np.random.randint(min_component, max_component + 1)
        
        # Ensure odd size for maze
        target_size = size
        if size % 2 == 0:
            size += 1
        
        shape = ((size // 2) * 2 + 3, (size // 2) * 2 + 3)
        total_density = np.random.uniform(obstacle_density[0], obstacle_density[1])
        density = int(shape[0] * shape[1] * total_density // num_components) if num_components != 0 else 0

        # Build maze
        Z = np.zeros(shape, dtype=np.int32)
        Z[0, :] = Z[-1, :] = 1
        Z[:, 0] = Z[:, -1] = 1
        
        # Make aisles
        for _ in range(density):
            x = np.random.randint(0, shape[1] // 2) * 2
            y = np.random.randint(0, shape[0] // 2) * 2
            Z[y, x] = 1
            last_dir = None
            
            for _ in range(num_components):
                neighbors = []
                if x > 1:
                    neighbors.append((y, x - 2))
                if x < shape[1] - 2:
                    neighbors.append((y, x + 2))
                if y > 1:
                    neighbors.append((y - 2, x))
                if y < shape[0] - 2:
                    neighbors.append((y + 2, x))
                
                if neighbors:
                    if last_dir is None:
                        idx = np.random.randint(0, len(neighbors))
                        y_, x_ = neighbors[int(idx)]
                        if Z[y_, x_] == 0:
                            last_dir = (y_ - y, x_ - x)
                            Z[y_, x_] = 1
                            Z[y_ + (y - y_) // 2, x_ + (x - x_) // 2] = 1
                            x, y = x_, y_
                    else:
                        index_F = -1
                        index_B = -1
                        diff = []
                        for y_, x_ in neighbors:
                            diff.append((y_ - y, x_ - x))
                            if diff[-1] == last_dir:
                                index_F = len(diff) - 1
                            elif diff[-1][0] + last_dir[0] == 0 and diff[-1][1] + last_dir[1] == 0:
                                index_B = len(diff) - 1
                        
                        assert index_B >= 0
                        if index_F + 1:
                            p = (1 - go_straight) * np.ones(len(neighbors)) / (len(neighbors) - 2)
                            p[index_B] = 0
                            p[index_F] = go_straight
                        else:
                            if len(neighbors) == 1:
                                p = np.array([1.0])
                            else:
                                p = np.ones(len(neighbors)) / (len(neighbors) - 1)
                                p[index_B] = 0

                        idx = np.random.choice(range(len(neighbors)), p=p)
                        y_, x_ = neighbors[idx]
                        if Z[y_, x_] == 0:
                            last_dir = (y_ - y, x_ - x)
                            Z[y_, x_] = 1
                            Z[y_ + (y - y_) // 2, x_ + (x - x_) // 2] = 1
                            x, y = x_, y_
        
        # Convert to standard format (0=free, 1=obstacle)
        grid_map = (1 - Z).astype(np.int32)
        
        # Crop/resize to exact target size
        if grid_map.shape[0] != target_size or grid_map.shape[1] != target_size:
            # Center crop to target size
            h, w = grid_map.shape
            start_h = (h - target_size) // 2
            start_w = (w - target_size) // 2
            grid_map = grid_map[start_h:start_h + target_size, start_w:start_w + target_size]
        
        return grid_map
    
    def _generate_house(self, size: int, obstacle_ratio: int = 10,
                              remove_edge_ratio: int = 6) -> np.ndarray:
        """House generator - original house algorithm"""
        grid_map = np.zeros((size, size), dtype=np.int32)
        all_x = list(range(2, size - 2))
        all_y = list(range(2, size - 2))
        obs_edge = []
        obs_corner_x = []
        
        while len(obs_corner_x) < size // obstacle_ratio:
            corn_x = random.sample(all_x, 1)[0]
            near_flag = any(abs(i - corn_x) == 1 for i in obs_corner_x)
            if not near_flag:
                obs_corner_x.append(corn_x)
        
        obs_corner_y = []
        while len(obs_corner_y) < size // obstacle_ratio:
            corn_y = random.sample(all_y, 1)[0]
            near_flag = any(abs(i - corn_y) == 1 for i in obs_corner_y)
            if not near_flag:
                obs_corner_y.append(corn_y)
        
        obs_corner_x.extend([0, size - 1])
        obs_corner_y.extend([0, size - 1])
        
        # Create walls
        for i in obs_corner_x:
            edge = []
            for j in range(size):
                grid_map[i, j] = 1
                if j not in obs_corner_y:
                    edge.append([i, j])
                if j in obs_corner_y and edge:
                    obs_edge.append(edge)
                    edge = []
        
        for i in obs_corner_y:
            edge = []
            for j in range(size):
                grid_map[j, i] = 1
                if j not in obs_corner_x:
                    edge.append([j, i])
                if j in obs_corner_x and edge:
                    obs_edge.append(edge)
                    edge = []
        
        # Remove some edges (doors)
        all_edge_list = list(range(len(obs_edge)))
        remove_edge = random.sample(all_edge_list, len(obs_edge) // remove_edge_ratio)
        for edge_number in remove_edge:
            for current_edge in obs_edge[edge_number]:
                grid_map[current_edge[0], current_edge[1]] = 0
        
        # Remove small edges
        for edges in obs_edge:
            if len(edges) == 1 or len(edges) <= size // 20:
                for coordinates in edges:
                    grid_map[coordinates[0], coordinates[1]] = 0
        
        # Ensure connectivity
        _, count = skimage.measure.label(grid_map, background=1, connectivity=1, return_num=True)
        while count != 1 and len(obs_edge) > 0:
            door_edge_index = random.sample(range(len(obs_edge)), 1)[0]
            door_edge = obs_edge[door_edge_index]
            door_index = random.sample(range(len(door_edge)), 1)[0]
            door = door_edge[door_index]
            grid_map[door[0], door[1]] = 0
            _, count = skimage.measure.label(grid_map, background=1, connectivity=1, return_num=True)
            obs_edge.remove(door_edge)
        
        # Ensure borders
        grid_map[:, -1] = grid_map[:, 0] = 1
        grid_map[-1, :] = grid_map[0, :] = 1
        
        return grid_map
    
    def _generate_warehouse(self, size: int, num_block: Tuple[int, int] = (2, 2),
                                  num_shelves: Tuple[int, int] = (4, 2)) -> np.ndarray:
        """Warehouse generator - returns exact size"""
        # Calculate appropriate num_block and num_shelves to fit size
        # Adjust parameters to create a warehouse that fits the target size
        target_size = size
        if size < 15:
            num_block = (1, 1)
            num_shelves = (2, 2)
        elif size < 25:
            num_block = (2, 2)
            num_shelves = (2, 2)
        else:
            num_block = (2, 2)
            num_shelves = (3, 2)
        
        shelf_size = (2 * num_shelves[0] + num_shelves[0] - 1, 5 * num_shelves[1] + num_shelves[1] - 1)
        block_size = (shelf_size[0] + 4, shelf_size[1] + 4)
        env_size = (block_size[0] * num_block[0] + 2, block_size[1] * num_block[1] + 2)
        
        # Build warehouse
        grid_map = np.zeros(env_size, dtype=np.int32)
        grid_map[0, :] = grid_map[-1, :] = 1
        grid_map[:, 0] = grid_map[:, -1] = 1
        
        for i in range(1, num_block[0] * num_shelves[0] + 1):
            for j in range(1, num_block[1] * num_shelves[1] + 1):
                counter_row = int((i - 1) // num_shelves[0])
                counter_column = int((j - 1) // num_shelves[1])
                grid_map[3 * i + counter_row * 3:3 * i + counter_row * 3 + 2,
                3 + 6 * (j - 1) + counter_column * 3:3 + 6 * (j - 1) + 5 + counter_column * 3] = 1
        
        # Resize to exact target size
        if grid_map.shape[0] != target_size or grid_map.shape[1] != target_size:
            # Center crop to target size
            h, w = grid_map.shape
            start_h = (h - target_size) // 2
            start_w = (w - target_size) // 2
            grid_map = grid_map[start_h:start_h + target_size, start_w:start_w + target_size]
        
        return grid_map
    
    def _generate_tunnels(self, size: int) -> np.ndarray:
        """Generate random tunnel network - horizontal and vertical lines at random positions"""
        grid_map = np.ones((size, size), dtype=np.int32)
        
        # Number of horizontal and vertical lines
        num_horizontal = max(2, size // 6)
        num_vertical = max(2, size // 6)
        
        # Create random horizontal lines (full width)
        horizontal_positions = set()
        for _ in range(num_horizontal):
            y = np.random.randint(1, size - 1)
            horizontal_positions.add(y)
        
        for y in horizontal_positions:
            for x in range(1, size - 1):
                grid_map[x, y] = 0
        
        # Create random vertical lines (full height)
        vertical_positions = set()
        for _ in range(num_vertical):
            x = np.random.randint(1, size - 1)
            vertical_positions.add(x)
        
        for x in vertical_positions:
            for y in range(1, size - 1):
                grid_map[x, y] = 0
        
        # Connectivity will be ensured by ensure_connectivity() called in generate()
        return grid_map
    
def random_generator(size=(10, 40), density=0.3):
    """Random generator
    
    Args:
        size: Size range tuple (min, max) for map dimensions
        density: A float (0.0 to 1.0) representing obstacle density value
    """
    generator = MapGenerator()
    size = np.random.randint(size[0], size[1] + 1)
    grid_map = generator._generate_random(size, density)
    return generator.ensure_connectivity(grid_map)

def maze_generator(env_size=(10, 70), wall_components=(1, 8), obstacle_density=None, go_straight=0.8):
    """Maze generator"""
    generator = MapGenerator()
    size = np.random.randint(env_size[0], env_size[1] + 1)
    grid_map = generator._generate_maze(size, wall_components, obstacle_density, go_straight)
    return generator.ensure_connectivity(grid_map)


def house_generator(env_size=10, obstacle_ratio=10, remove_edge_ratio=6):
    """House generator"""
    generator = MapGenerator()
    if isinstance(env_size, tuple):
        size = np.random.randint(env_size[0], env_size[1] + 1)
    else:
        size = env_size
    grid_map = generator._generate_house(size, obstacle_ratio, remove_edge_ratio)
    grid_map = generator.ensure_connectivity(grid_map)
    # Return format compatible with original (map, nodes)
    nodes = get_map_nodes(grid_map)
    return grid_map, nodes


def warehouse_generator(num_block, num_shelves=(10, 5), ent_location=10, ent_position='right', entrance_size=1):
    """Warehouse generator"""
    generator = MapGenerator()
    # Estimate size from num_block
    size = max(20, num_block[0] * 10)
    grid_map = generator._generate_warehouse(size, num_block, num_shelves)
    return generator.ensure_connectivity(grid_map)


def get_map_nodes(world):
    """Get map nodes for pathfinding - original SIGMA function"""
    def neighbor(x, y, image):
        num_free_cell = 0
        x_1, y_1, x1, y1 = x - 1, y - 1, x + 1, y + 1
        if x_1 >= 0 and y_1 >= 0 and x1 < image.shape[0] and y1 < image.shape[1]:
            if image[x, y] == 0:
                for i in range(x_1, x1 + 1):
                    for j in range(y_1, y1 + 1):
                        if image[i, j] == 0:
                            num_free_cell = num_free_cell + 1
        return num_free_cell - 1

    def end_branch_point(image):
        ed_points = []
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                num_neighbors = neighbor(i, j, image)
                if image[i, j] == 0:
                    if num_neighbors == 1 or num_neighbors >= 3:
                        ed_points.append([i, j])
        return ed_points

    def mask_ebpoints(image, eb_points):
        # Simplified masking logic
        return eb_points

    world_for_ske = 1 - (-1 * world)
    skeleton, distance = morphology.medial_axis(world_for_ske, return_distance=True)
    ske_needed = skeleton.astype(int) - 1
    eb_points = end_branch_point(ske_needed)
    nodes = mask_ebpoints(ske_needed, eb_points)
    return nodes


def generate_task_maps(num_tasks: int, size_range: Tuple[int, int] = (10, 20),
                      map_types: Optional[List[MapType]] = None) -> List[Tuple[np.ndarray, MapType, dict]]:
    """
    Generate a set of diverse maps for training.
    
    Args:
        num_tasks: Number of different map tasks to generate
        size_range: (min_size, max_size) for map dimensions
        map_types: Specific map types to use (None = use all types)
    
    Returns:
        List of (map, map_type, statistics) tuples
    """
    generator = MapGenerator(min_size=size_range[0], max_size=size_range[1])
    
    if map_types is None:
        map_types = list(MapType)
    
    task_list = []
    for task_idx in range(num_tasks):
        task_map_type = map_types[task_idx % len(map_types)]
        task_size = np.random.randint(size_range[0], size_range[1] + 1)
        
        task_map = generator.generate(task_map_type, task_size)
        task_map = generator.ensure_connectivity(task_map)
        task_stats = generator.get_map_statistics(task_map)
        
        task_list.append((task_map, task_map_type, task_stats))
    
    return task_list
