import numpy as np
from PIL import Image
import random
from skimage import io
import os
from collections import deque
import matplotlib.pyplot as plt
from typing import Tuple, List

FREE = 255         
START = 208        
END = 232
OBSTACLE = 127

# # training data seed
# random.seed(42)
# np.random.seed(42)

# inpainting data seed
random.seed(1)
np.random.seed(1)

# # testing data seed
# random.seed(0)
# np.random.seed(0)

class MapGenerator:
    def __init__(self,
                 grid_size: int = 25, 
                 num_grids: int = 20, 
                 num_rooms: tuple = (5, 6),
                 room_min_size: int = 3, 
                 room_max_size: int = 6,
                 corridor_width: int = (1, 1),
                 corridor_density: float = 0.2,
                 draw_goal: bool = False,
                 prefix_name: str = 'map'):
        """
        初始化地图生成器。

        参数：
        - grid_size: 每个网格的像素大小
        - num_grids: 地图的网格数量（n x n）
        - num_rooms: 房间的数量
        - room_min_size: 房间的最小尺寸（边长）
        - room_max_size: 房间的最大尺寸（边长）
        - corridor_width: 通道的宽度（网格数）
        - corridor_density: 通道密度，0到1之间的浮点数，表示在MST基础上添加额外通道的概率
        - draw_goal: 是否在地图上绘制终点位置
        """
        self.grid_size = grid_size
        self.num_grids = num_grids
        self.num_rooms = random.randint(*num_rooms)
        self.room_min_size = room_min_size
        self.room_max_size = room_max_size
        self.corridor_width = random.randint(*corridor_width)
        self.corridor_density = corridor_density
        self.draw_goal = draw_goal
        self.prefix_name = prefix_name

    def initialize_map(self) -> np.ndarray:

        grid_status = np.zeros((self.num_grids, self.num_grids), dtype=int)
        return grid_status

    def place_rooms(self, grid_status: np.ndarray) -> List[Tuple[int, int, int, int]]:

        rooms = []
        max_attempts = self.num_rooms * 5
        attempts = 0

        while len(rooms) < self.num_rooms and attempts < max_attempts:
            room_width = random.randint(self.room_min_size, self.room_max_size)
            room_height = random.randint(self.room_min_size, self.room_max_size)
            x = random.randint(1, self.num_grids - room_width - 1)
            y = random.randint(1, self.num_grids - room_height - 1)

            new_room = (x, y, x + room_width, y + room_height)
            overlap = False
            for other_room in rooms:
                if self.rooms_overlap(new_room, other_room):
                    overlap = True
                    break
            if not overlap:
                self.carve_room(grid_status, new_room)
                rooms.append(new_room)
            attempts += 1

        if len(rooms) < self.num_rooms:
            print(f"警告：仅成功生成了 {len(rooms)} 个房间。")
        return rooms

    def rooms_overlap(self, room1: Tuple[int, int, int, int], room2: Tuple[int, int, int, int]) -> bool:

        return (room1[0] <= room2[2] and room1[2] >= room2[0] and
                room1[1] <= room2[3] and room1[3] >= room2[1])

    def carve_room(self, grid_status: np.ndarray, room: Tuple[int, int, int, int]):

        x1, y1, x2, y2 = room
        grid_status[x1:x2, y1:y2] = 1

    def connect_rooms(self, grid_status: np.ndarray, rooms: List[Tuple[int, int, int, int]]) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:

        if not rooms:
            return []

        centers = [( (room[0] + room[2]) // 2, (room[1] + room[3]) // 2 ) for room in rooms]

       
        mst_connections = self.minimum_spanning_tree(centers)

        for (start, end) in mst_connections:
            self.carve_corridor(grid_status, start, end)

        return mst_connections

    def minimum_spanning_tree(self, centers: List[Tuple[int, int]]) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:

        if not centers:
            return []

        connections = []
        connected = set()
        connected.add(centers[0])

        while len(connected) < len(centers):
            min_dist = float('inf')
            closest_pair = None
            for c1 in connected:
                for c2 in centers:
                    if c2 not in connected:
                        dist = abs(c1[0] - c2[0]) + abs(c1[1] - c2[1])
                        if dist < min_dist:
                            min_dist = dist
                            closest_pair = (c1, c2)
            if closest_pair:
                connections.append(closest_pair)
                connected.add(closest_pair[1])

        return connections

    def carve_corridor(self, grid_status: np.ndarray, start: Tuple[int, int], end: Tuple[int, int]):

        x1, y1 = start
        x2, y2 = end

        if random.choice([True, False]):
           
            for x in range(min(x1, x2), max(x1, x2) + 1):
                self.carve_corridor_width(grid_status, x, y1)
            for y in range(min(y1, y2), max(y1, y2) + 1):
                self.carve_corridor_width(grid_status, x2, y)
        else:
           
            for y in range(min(y1, y2), max(y1, y2) + 1):
                self.carve_corridor_width(grid_status, x1, y)
            for x in range(min(x1, x2), max(x1, x2) + 1):
                self.carve_corridor_width(grid_status, x, y2)

    def carve_corridor_width(self, grid_status: np.ndarray, x: int, y: int):

        half_width = self.corridor_width // 2
        for dx in range(-half_width, half_width + 1):
            for dy in range(-half_width, half_width + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.num_grids and 0 <= ny < self.num_grids:
                    grid_status[nx][ny] = 1

    def add_extra_corridors(self, grid_status: np.ndarray, rooms: List[Tuple[int, int, int, int]], mst_connections: List[Tuple[Tuple[int, int], Tuple[int, int]]]):

        centers = [( (room[0] + room[2]) // 2, (room[1] + room[3]) // 2 ) for room in rooms]
        connected_pairs = set([tuple(sorted([conn[0], conn[1]])) for conn in mst_connections])

        all_pairs = []
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                pair = tuple(sorted([centers[i], centers[j]]))
                if pair not in connected_pairs:
                    all_pairs.append(pair)

        total_possible_extra = len(all_pairs)
        num_extra_corridors = int(self.corridor_density * total_possible_extra)

        extra_corridors = random.sample(all_pairs, min(num_extra_corridors, total_possible_extra))

        for (start, end) in extra_corridors:
            self.carve_corridor(grid_status, start, end)

    def convert_to_image(self, grid_status: np.ndarray, start: Tuple[int, int], draw_goal: bool = False) -> np.ndarray:

        image = np.ones((self.grid_size * self.num_grids, self.grid_size * self.num_grids), dtype=np.uint8) * OBSTACLE
        for x in range(self.num_grids): 
            for y in range(self.num_grids):
                if grid_status[x][y] == 1:
                    x_start_pix = x * self.grid_size
                    y_start_pix = y * self.grid_size
                    image[x_start_pix:x_start_pix + self.grid_size, y_start_pix:y_start_pix + self.grid_size] = FREE
        x_start_pix = start[0] * self.grid_size
        y_start_pix = start[1] * self.grid_size
        image[x_start_pix:x_start_pix + self.grid_size, y_start_pix:y_start_pix + self.grid_size] = START
        
        if draw_goal:
            
            free_positions = np.argwhere(grid_status == 1)
            if len(free_positions) >= 2:
                valid_positions = [
                    pos for pos in free_positions
                    if tuple(pos) != start and np.linalg.norm(np.array(pos) - np.array(start)) > 15
                ]
                if valid_positions:
                    end_pos = tuple(random.choice(valid_positions))
                else:
                    print("没有有效的终点位置可用，随机选择一个")
                    end_pos = tuple(random.choice([pos for pos in free_positions if tuple(pos) != start]))
                x_end_pix = end_pos[0] * self.grid_size
                y_end_pix = end_pos[1] * self.grid_size
                image[x_end_pix:x_end_pix + self.grid_size, y_end_pix:y_end_pix + self.grid_size] = END
        return image

    def contains_subpattern(self, grid_status: np.ndarray) -> bool:
        pattern1 = np.array([[0, 1], [1, 0]])
        pattern2 = np.array([[1, 0], [0, 1]])
        rows, cols = grid_status.shape
        for i in range(rows - 1):
            for j in range(cols - 1):
                subarray = grid_status[i:i + 2, j:j + 2]
                if np.array_equal(subarray, pattern1) or np.array_equal(subarray, pattern2):
                    return True
        return False

    def visualize_map(self, image_array: np.ndarray, img_index: int):

        plt.figure(figsize=(6,6))
        plt.imshow(image_array, cmap='gray', vmin=0, vmax=255)
        plt.title(f'Map {img_index}')
        plt.axis('off')
        plt.show()

    def save_map(self, image_array: np.ndarray, img_index: int):

        img = Image.fromarray(image_array)
        img.save(f'{self.prefix_name}/{self.prefix_name}_{img_index + 1}.png')

    def generate_map(self, img_index: int) -> bool:

        grid_status = self.initialize_map()
        
        rooms = self.place_rooms(grid_status)
        if not rooms:
            return False  
        
        mst_connections = self.connect_rooms(grid_status, rooms)
        
        self.add_extra_corridors(grid_status, rooms, mst_connections)
        
        start_pos = rooms[0]  
        start_center = ( (start_pos[0] + start_pos[2]) // 2, (start_pos[1] + start_pos[3]) // 2 )

        pattern_exist = self.contains_subpattern(grid_status)
        if pattern_exist:
            return False
        grid_status[0, :] = 0
        grid_status[-1, :] = 0
        grid_status[:, 0] = 0
        grid_status[:, -1] = 0
        
        image = self.convert_to_image(grid_status, start_center, self.draw_goal)

        self.save_map(image, img_index)
        
        return True  

def main():
    '''map_params = {
        'grid_size': 10,
        'num_grids': 25,
        'num_rooms': (6, 8),
        'room_min_size': 2,
        'room_max_size': 6,
        'corridor_width': (1, 1),
        'corridor_density': 0.3,
        'draw_goal': True,
        'prefix_name': 'maps'
    }'''
    '''map_params = {
        'grid_size': 10,
        'num_grids': 40,
        'num_rooms': (7, 7),
        'room_min_size': 2,
        'room_max_size': 6,
        'corridor_width': (1, 1),
        'corridor_density': 0.9,
        'draw_goal': False,
        'prefix_name': 'tunnel'
    }'''
    '''map_params = {
        'grid_size': 10,
        'num_grids': 40,
        'num_rooms': (14, 14),
        'room_min_size': 3,
        'room_max_size': 7,
        'corridor_width': (1, 1),
        'corridor_density': 0.1,
        'draw_goal': False,
        'prefix_name': 'room'
    }'''
    map_params = {
        'grid_size': 10,
        'num_grids': 25,
        'num_rooms': (7, 7),
        'room_min_size': 2,
        'room_max_size': 6,
        'corridor_width': (2, 3),
        'corridor_density': 0.5,
        'draw_goal': False,
        'prefix_name': 'outdoor'
    }
    map_nums = 1000
    prefix_name = map_params['prefix_name']
    if not os.path.exists(prefix_name):
        os.makedirs(prefix_name)

    for img_index in range(map_nums):
        generator = MapGenerator(**map_params)
        while not generator.generate_map(img_index):
            print(f"地图 {img_index + 1} 生成失败，正在重试...")
        
        if (img_index + 1) % 500 == 0:
            try:
                ground_truth = io.imread(f'{prefix_name}/{prefix_name}_{img_index + 1}.png', as_gray=True).astype(int)
                print(f"地图 {img_index + 1} 生成成功，形状: {ground_truth.shape}")
                print(f"唯一像素值: {np.unique(ground_truth)}")
            except FileNotFoundError:
                print(f"地图 '{prefix_name}_{img_index + 1}.png' 未找到。")
            generator.visualize_map(ground_truth, img_index + 1)
    
    try:
        ground_truth = io.imread(f'{prefix_name}/{prefix_name}_93.png', as_gray=True).astype(int)
        print("Ground Truth Shape:", ground_truth.shape)
        print("Unique Pixel Values:", np.unique(ground_truth))
    except FileNotFoundError:
        print("地图 'img_93.png' 未找到。请确保已生成至少93张地图。")

if __name__ == "__main__":
    main()
