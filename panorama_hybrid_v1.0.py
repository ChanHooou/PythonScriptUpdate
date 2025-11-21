import numpy as np
from PIL import Image
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Set
import re
import cv2
from collections import defaultdict

@dataclass
class ImageMetadata:
    """이미지 메타데이터"""
    filename: str
    index: int
    front: int
    back: int
    right: int
    left: int
    
    @classmethod
    def parse_filename(cls, filename: str):
        pattern = r'p(\d+)[_\s]+f(\d+)[_\s]+b(\d+)[_\s]+r(\d+)[_\s]+l(\d+)'
        match = re.search(pattern, filename, re.IGNORECASE)
        if not match:
            raise ValueError(f"Invalid filename format: {filename}")
        
        idx, f, b, r, l = match.groups()
        return cls(
            filename=filename,
            index=int(idx),
            front=int(f),
            back=int(b),
            right=int(r),
            left=int(l)
        )


class GlobalOptimizationStitcher:
    """전역 최적화 파노라마 스티칭: 센서 배치 → 인접 이미지 피처 정합"""
    
    def __init__(self, 
                 folder_path: str, 
                 building_width: int = None, 
                 building_height: int = None,
                 image_real_width: int = None,
                 image_real_height: int = None,
                 sensor_mode: str = "BL",
                 movement_direction: str = "forward",
                 use_global_optimization: bool = True,
                 overlap_threshold: float = 0.3,
                 refinement_iterations: int = 3,
                 feature_method: str = "SIFT"):
        """
        Args:
            folder_path: 이미지 폴더 경로
            building_width: 건물 가로 길이 (cm)
            building_height: 건물 세로 길이 (cm)
            image_real_width: 이미지 실제 가로 크기 (cm)
            image_real_height: 이미지 실제 세로 크기 (cm)
            sensor_mode: 센서 조합 ("FL", "FR", "BL", "BR")
            movement_direction: 이동 방향
            use_global_optimization: 전역 최적화 사용 여부
            overlap_threshold: 겹침 판단 임계값 (0.0~1.0)
            refinement_iterations: 정밀화 반복 횟수
            feature_method: 피처 추출 방법 ("SIFT", "ORB")
        """
        self.folder_path = Path(folder_path)
        self.metadata_list: List[ImageMetadata] = []
        self.images: List[np.ndarray] = []
        self.building_width = building_width
        self.building_height = building_height
        
        self.sensor_mode = sensor_mode.upper()
        self.movement_direction = movement_direction.lower()
        
        if self.sensor_mode not in ["FL", "FR", "BL", "BR"]:
            raise ValueError(f"Invalid sensor_mode: {sensor_mode}")
        
        if self.movement_direction not in ["forward", "backward", "left", "right"]:
            raise ValueError(f"Invalid movement_direction: {movement_direction}")
        
        self.vertical_sensor = self.sensor_mode[0]
        self.horizontal_sensor = self.sensor_mode[1]
        
        self.use_global_optimization = use_global_optimization
        self.overlap_threshold = overlap_threshold
        self.refinement_iterations = refinement_iterations
        self.feature_method = feature_method.upper()
        
        self.IMAGE_REAL_WIDTH = image_real_width if image_real_width else 125
        self.IMAGE_REAL_HEIGHT = image_real_height if image_real_height else 87
        
        self.IMAGE_PIXEL_WIDTH = None
        self.IMAGE_PIXEL_HEIGHT = None
        
        self.CM_PER_PIXEL_X = None
        self.CM_PER_PIXEL_Y = None
        self.PIXEL_PER_CM_X = None
        self.PIXEL_PER_CM_Y = None
        
        # 이미지 위치 저장 (전역 최적화용)
        self.positions: List[Tuple[int, int]] = []
        
        # 피처 디텍터
        self.feature_detector = None
        self.feature_matcher = None
        if self.use_global_optimization:
            self._initialize_feature_detector()
        
        print(f"\n{'='*60}")
        print("Global Optimization Panorama Stitcher")
        print(f"{'='*60}")
        sensor_names = {"F": "Front", "B": "Back", "L": "Left", "R": "Right"}
        print(f"Sensor Mode: {self.sensor_mode} ({sensor_names[self.vertical_sensor]}/{sensor_names[self.horizontal_sensor]})")
        print(f"Movement Direction: {self.movement_direction}")
        if self.use_global_optimization:
            print(f"Global Optimization: Enabled")
            print(f"  Feature Method: {self.feature_method}")
            print(f"  Overlap Threshold: {self.overlap_threshold}")
            print(f"  Refinement Iterations: {self.refinement_iterations}")
        else:
            print(f"Global Optimization: Disabled (Sensor-only)")
        
    def _initialize_feature_detector(self):
        """피처 디텍터 초기화"""
        try:
            if self.feature_method == "SIFT":
                self.feature_detector = cv2.SIFT_create(nfeatures=2000)
                self.feature_matcher = cv2.FlannBasedMatcher(
                    dict(algorithm=1, trees=5),
                    dict(checks=50)
                )
            else:
                self.feature_detector = cv2.ORB_create(nfeatures=2000)
                self.feature_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            
            print(f"  ✓ {self.feature_method} feature detector initialized")
                
        except Exception as e:
            print(f"  ⚠ Feature detector initialization failed: {e}")
            print(f"  Falling back to sensor-only mode")
            self.use_global_optimization = False
    
    def _calculate_scale(self):
        """스케일 계산"""
        if self.IMAGE_PIXEL_WIDTH and self.IMAGE_PIXEL_HEIGHT:
            self.CM_PER_PIXEL_X = self.IMAGE_REAL_WIDTH / self.IMAGE_PIXEL_WIDTH
            self.CM_PER_PIXEL_Y = self.IMAGE_REAL_HEIGHT / self.IMAGE_PIXEL_HEIGHT
            self.PIXEL_PER_CM_X = self.IMAGE_PIXEL_WIDTH / self.IMAGE_REAL_WIDTH
            self.PIXEL_PER_CM_Y = self.IMAGE_PIXEL_HEIGHT / self.IMAGE_REAL_HEIGHT
            
            print(f"Image size: {self.IMAGE_PIXEL_WIDTH} x {self.IMAGE_PIXEL_HEIGHT} px")
            print(f"Real size: {self.IMAGE_REAL_WIDTH} x {self.IMAGE_REAL_HEIGHT} cm")
            print(f"Scale: {self.PIXEL_PER_CM_X:.3f} px/cm (X), {self.PIXEL_PER_CM_Y:.3f} px/cm (Y)")
            print(f"{'='*60}")
    
    def _get_sort_key(self, meta: ImageMetadata):
        """정렬 키 반환"""
        if self.movement_direction == "forward":
            return -meta.front if self.vertical_sensor == "F" else meta.back
        elif self.movement_direction == "backward":
            return meta.front if self.vertical_sensor == "F" else -meta.back
        elif self.movement_direction == "left":
            return -meta.left if self.horizontal_sensor == "L" else meta.right
        elif self.movement_direction == "right":
            return meta.left if self.horizontal_sensor == "L" else -meta.right
    
    def load_images_from_folder(self):
        """폴더에서 이미지 로드"""
        image_files_set = set()
        for ext in ['*.jpg', '*.jpeg', '*.JPG', '*.JPEG', '*.png', '*.PNG']:
            image_files_set.update(self.folder_path.glob(ext))
        
        image_files = list(image_files_set)
        
        if not image_files:
            raise FileNotFoundError(f"No images found in {self.folder_path}")
        
        print(f"\nFound {len(image_files)} images")
        
        temp_data = []
        for img_path in image_files:
            try:
                meta = ImageMetadata.parse_filename(img_path.name)
                temp_data.append((meta, img_path))
            except ValueError as e:
                print(f"  ⚠ Skipping {img_path.name}: {e}")
        
        if len(temp_data) == 0:
            raise ValueError("No valid images found")
        
        temp_data.sort(key=lambda x: self._get_sort_key(x[0]))
        
        with Image.open(str(temp_data[0][1])) as first_img:
            self.IMAGE_PIXEL_WIDTH, self.IMAGE_PIXEL_HEIGHT = first_img.size
        
        self._calculate_scale()
        
        for idx, (meta, img_path) in enumerate(temp_data):
            self.metadata_list.append(meta)
            
            with Image.open(str(img_path)) as img:
                img_array = np.array(img)
            
            h, w = img_array.shape[:2]
            if h != self.IMAGE_PIXEL_HEIGHT or w != self.IMAGE_PIXEL_WIDTH:
                with Image.open(str(img_path)) as img:
                    img_resized = img.resize((self.IMAGE_PIXEL_WIDTH, self.IMAGE_PIXEL_HEIGHT), Image.LANCZOS)
                    img_array = np.array(img_resized)
            
            self.images.append(img_array)
            
            if idx < 5:
                print(f"  [{idx:03d}] F:{meta.front:05d} B:{meta.back:05d} R:{meta.right:05d} L:{meta.left:05d}")
        
        print(f"\n✓ Successfully loaded {len(self.images)} images")
    
    def calculate_sensor_offset(self, idx: int) -> Tuple[int, int]:
        """센서 기반 오프셋 계산"""
        if idx == 0:
            return (0, 0)
        
        prev = self.metadata_list[idx - 1]
        curr = self.metadata_list[idx]
        
        # 세로축(Y)
        if self.vertical_sensor == "F":
            front_diff = prev.front - curr.front
            dy = -int(front_diff * self.PIXEL_PER_CM_Y)
        else:
            back_diff = curr.back - prev.back
            dy = -int(back_diff * self.PIXEL_PER_CM_Y)
        
        # 가로축(X)
        if self.horizontal_sensor == "L":
            left_diff = curr.left - prev.left
            dx = int(left_diff * self.PIXEL_PER_CM_X)
        else:
            right_diff = curr.right - prev.right
            dx = -int(right_diff * self.PIXEL_PER_CM_X)
        
        # 부 이동축 제한
        if self.movement_direction in ["forward", "backward"]:
            max_dx = int(self.IMAGE_PIXEL_WIDTH * 0.15)
            dx = max(-max_dx, min(max_dx, dx))
        else:
            max_dy = int(self.IMAGE_PIXEL_HEIGHT * 0.15)
            dy = max(-max_dy, min(max_dy, dy))
        
        return (dx, dy)
    
    def build_initial_layout_sensor(self):
        """Phase 1: 센서 데이터로 초기 배치 구축"""
        print(f"\n{'='*60}")
        print("Phase 1: Building Initial Layout from Sensor Data")
        print(f"{'='*60}")
        
        h = self.IMAGE_PIXEL_HEIGHT
        w = self.IMAGE_PIXEL_WIDTH
        
        self.positions = [(0, 0)]
        
        for i in range(1, len(self.images)):
            dx, dy = self.calculate_sensor_offset(i)
            prev_x, prev_y = self.positions[-1]
            new_x = prev_x + dx
            new_y = prev_y + dy
            self.positions.append((new_x, new_y))
            
            if i < 5 or i % 10 == 0:
                direction_v = "↑" if dy < 0 else "↓" if dy > 0 else "—"
                direction_h = "←" if dx < 0 else "→" if dx > 0 else ""
                print(f"Image {i:3d}: offset=({dx:+5d}, {dy:+5d}) {direction_h}{direction_v} → pos=({new_x:6d}, {new_y:6d}) [sensor]")
        
        print(f"\n✓ Initial layout completed with {len(self.positions)} images")
    
    def _create_overlap_masks(self, shape1: Tuple[int, int], shape2: Tuple[int, int],
                              idx1: int, idx2: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """이동 방향에 따라 겹침 영역에 대한 마스크 생성

        Args:
            shape1: 첫 번째 이미지 shape (h, w)
            shape2: 두 번째 이미지 shape (h, w)
            idx1: 첫 번째 이미지 인덱스
            idx2: 두 번째 이미지 인덱스

        Returns:
            (mask1, mask2): 각 이미지에 대한 마스크 (None이면 전체 이미지 사용)
        """
        h1, w1 = shape1
        h2, w2 = shape2

        # 이동 방향이 전진/후진일 때만 특수 처리
        if self.movement_direction not in ["forward", "backward"]:
            return None, None

        # 위치 관계 파악
        y1, y2 = self.positions[idx1][1], self.positions[idx2][1]

        # 겹침 영역 비율 (이미지 높이의 50% - 충분한 특징점 확보)
        overlap_ratio = 0.50
        overlap_height = int(h1 * overlap_ratio)

        mask1 = None
        mask2 = None

        if self.movement_direction == "forward":
            # Forward: 아래쪽 이미지가 위쪽 이미지보다 y 값이 작음 (위로 이동)
            if y2 < y1:  # idx2가 idx1보다 위에 있음
                # idx1의 상단 영역, idx2의 하단 영역 매칭
                mask1 = np.zeros((h1, w1), dtype=np.uint8)
                mask1[0:overlap_height, :] = 255

                mask2 = np.zeros((h2, w2), dtype=np.uint8)
                mask2[h2-overlap_height:h2, :] = 255
            else:  # idx2가 idx1보다 아래에 있음
                # idx1의 하단 영역, idx2의 상단 영역 매칭
                mask1 = np.zeros((h1, w1), dtype=np.uint8)
                mask1[h1-overlap_height:h1, :] = 255

                mask2 = np.zeros((h2, w2), dtype=np.uint8)
                mask2[0:overlap_height, :] = 255

        elif self.movement_direction == "backward":
            # Backward: 위쪽 이미지가 아래쪽 이미지보다 y 값이 크음 (아래로 이동)
            if y2 > y1:  # idx2가 idx1보다 아래에 있음
                # idx1의 하단 영역, idx2의 상단 영역 매칭
                mask1 = np.zeros((h1, w1), dtype=np.uint8)
                mask1[h1-overlap_height:h1, :] = 255

                mask2 = np.zeros((h2, w2), dtype=np.uint8)
                mask2[0:overlap_height, :] = 255
            else:  # idx2가 idx1보다 위에 있음
                # idx1의 상단 영역, idx2의 하단 영역 매칭
                mask1 = np.zeros((h1, w1), dtype=np.uint8)
                mask1[0:overlap_height, :] = 255

                mask2 = np.zeros((h2, w2), dtype=np.uint8)
                mask2[h2-overlap_height:h2, :] = 255

        return mask1, mask2

    def find_overlapping_neighbors(self, idx: int, max_distance: int = None) -> List[int]:
        """Phase 2: 특정 이미지와 겹치는 인접 이미지들 찾기
        
        Args:
            idx: 대상 이미지 인덱스
            max_distance: 최대 거리 (픽셀), None이면 자동 계산
            
        Returns:
            겹치는 이미지들의 인덱스 리스트
        """
        if max_distance is None:
            # 이미지 크기의 1.5배 이내
            max_distance = int(max(self.IMAGE_PIXEL_WIDTH, self.IMAGE_PIXEL_HEIGHT) * 1.5)
        
        x, y = self.positions[idx]
        w = self.IMAGE_PIXEL_WIDTH
        h = self.IMAGE_PIXEL_HEIGHT
        
        neighbors = []
        
        for i in range(len(self.images)):
            if i == idx:
                continue
            
            xi, yi = self.positions[i]
            
            # 중심점 간 거리 확인
            dx = abs(xi - x)
            dy = abs(yi - y)
            
            if dx < max_distance and dy < max_distance:
                # 실제 겹침 계산
                overlap_x = max(0, min(x + w, xi + w) - max(x, xi))
                overlap_y = max(0, min(y + h, yi + h) - max(y, yi))
                
                if overlap_x > 0 and overlap_y > 0:
                    overlap_area = overlap_x * overlap_y
                    total_area = w * h
                    overlap_ratio = overlap_area / total_area
                    
                    if overlap_ratio >= self.overlap_threshold:
                        neighbors.append(i)
        
        return neighbors
    
    def match_features_between_images(self, idx1: int, idx2: int) -> Optional[Tuple[int, int]]:
        """두 이미지 간 피처 매칭으로 상대 오프셋 계산

        Args:
            idx1: 기준 이미지 인덱스
            idx2: 비교 이미지 인덱스

        Returns:
            (dx, dy): idx2가 idx1 대비 이동해야 할 오프셋, 실패시 None
        """
        if not self.use_global_optimization:
            return None

        try:
            img1 = self.images[idx1]
            img2 = self.images[idx2]

            # 그레이스케일 변환
            gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY) if len(img1.shape) == 3 else img1
            gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY) if len(img2.shape) == 3 else img2

            # ROI 설정: 이동 방향에 따라 겹치는 영역에 집중
            mask1, mask2 = self._create_overlap_masks(gray1.shape, gray2.shape, idx1, idx2)

            # 피처 검출 (마스크 적용)
            kp1, des1 = self.feature_detector.detectAndCompute(gray1, mask1)
            kp2, des2 = self.feature_detector.detectAndCompute(gray2, mask2)
            
            if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
                return None
            
            # 매칭
            if self.feature_method == "SIFT":
                matches = self.feature_matcher.knnMatch(des1, des2, k=2)
                good_matches = []
                for m_n in matches:
                    if len(m_n) == 2:
                        m, n = m_n
                        if m.distance < 0.7 * n.distance:
                            good_matches.append(m)
            else:
                matches = self.feature_matcher.knnMatch(des1, des2, k=2)
                good_matches = []
                for m_n in matches:
                    if len(m_n) == 2:
                        m, n = m_n
                        if m.distance < 0.75 * n.distance:
                            good_matches.append(m)
            
            if len(good_matches) < 10:
                return None
            
            # 현재 위치 차이 (센서 기반)
            x1, y1 = self.positions[idx1]
            x2, y2 = self.positions[idx2]
            sensor_dx = x2 - x1
            sensor_dy = y2 - y1
            
            # 매칭된 점들의 변위
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches])
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches])
            
            displacements = dst_pts - src_pts
            
            # 센서 예측 주변 필터링
            search_radius = 100  # 픽셀
            
            # 예상되는 피처 이동: img1의 점이 img2에서 어디에 나타날지
            # img2는 img1 대비 (sensor_dx, sensor_dy) 만큼 이동
            # 따라서 img1의 피처는 img2에서 (-sensor_dx, -sensor_dy) 방향으로 보여야 함
            expected_dx = -sensor_dx
            expected_dy = -sensor_dy
            
            valid_mask = (
                (np.abs(displacements[:, 0] - expected_dx) < search_radius) &
                (np.abs(displacements[:, 1] - expected_dy) < search_radius)
            )
            
            if np.sum(valid_mask) < 5:
                return None
            
            # 중앙값으로 로버스트 추정
            dx_feature = int(np.median(displacements[valid_mask, 0]))
            dy_feature = int(np.median(displacements[valid_mask, 1]))
            
            # 피처 매칭 결과: img1의 점이 img2에서 (dx_feature, dy_feature) 이동
            # 이는 img2가 img1 대비 (-dx_feature, -dy_feature) 위치에 있다는 의미
            correction_dx = -dx_feature - sensor_dx
            correction_dy = -dy_feature - sensor_dy
            
            return (correction_dx, correction_dy)
            
        except Exception as e:
            return None
    
    def refine_positions_global(self):
        """Phase 3: 인접 이미지들과의 피처 매칭으로 위치 정밀화"""
        print(f"\n{'='*60}")
        print("Phase 2: Finding Overlapping Neighbors")
        print(f"{'='*60}")
        
        # 각 이미지의 인접 이미지 찾기
        neighbor_map: Dict[int, List[int]] = {}
        total_pairs = 0
        
        for i in range(len(self.images)):
            neighbors = self.find_overlapping_neighbors(i)
            neighbor_map[i] = neighbors
            total_pairs += len(neighbors)
            
            if i < 5 or len(neighbors) > 0 and i % 10 == 0:
                print(f"Image {i:3d}: {len(neighbors)} neighbors {neighbors[:5]}{'...' if len(neighbors) > 5 else ''}")
        
        print(f"\n✓ Found {total_pairs} overlapping pairs")
        
        if total_pairs == 0:
            print("⚠ No overlapping images found, skipping refinement")
            return
        
        print(f"\n{'='*60}")
        print(f"Phase 3: Refining Positions with Feature Matching")
        print(f"{'='*60}")
        
        # 반복적 정밀화
        for iteration in range(self.refinement_iterations):
            print(f"\nIteration {iteration + 1}/{self.refinement_iterations}")
            
            # 각 이미지에 대한 보정값 수집
            corrections: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
            successful_matches = 0
            failed_matches = 0
            
            # 모든 인접 쌍에 대해 피처 매칭
            processed_pairs = set()
            
            for i in range(len(self.images)):
                if i % 10 == 0 or i < 5:
                    print(f"  Processing image {i}/{len(self.images)}...", end='\r')
                
                for j in neighbor_map[i]:
                    # 중복 처리 방지
                    pair = tuple(sorted([i, j]))
                    if pair in processed_pairs:
                        continue
                    processed_pairs.add(pair)
                    
                    # 피처 매칭
                    correction = self.match_features_between_images(i, j)
                    
                    if correction is not None:
                        dx, dy = correction
                        # j를 보정
                        corrections[j].append((dx, dy))
                        successful_matches += 1
                    else:
                        failed_matches += 1
            
            print(f"  Processing image {len(self.images)}/{len(self.images)}... Done")
            print(f"  Feature matching: {successful_matches} success, {failed_matches} failed")
            
            if successful_matches == 0:
                print("  No successful matches, stopping refinement")
                break
            
            # 보정 적용 (평균값 사용)
            max_correction = 0
            corrected_count = 0
            
            for i, correction_list in corrections.items():
                if len(correction_list) > 0:
                    # 중앙값으로 로버스트하게
                    dx_corrections = [c[0] for c in correction_list]
                    dy_corrections = [c[1] for c in correction_list]
                    
                    dx_median = int(np.median(dx_corrections))
                    dy_median = int(np.median(dy_corrections))
                    
                    # 보정 적용
                    x, y = self.positions[i]
                    self.positions[i] = (x + dx_median, y + dy_median)
                    
                    correction_magnitude = np.sqrt(dx_median**2 + dy_median**2)
                    max_correction = max(max_correction, correction_magnitude)
                    corrected_count += 1
            
            print(f"  Applied corrections to {corrected_count} images")
            print(f"  Max correction: {max_correction:.1f} pixels")
            
            # 수렴 판단
            if max_correction < 2.0:  # 2픽셀 이하면 수렴
                print(f"  Converged (max correction < 2 pixels)")
                break
        
        print(f"\n✓ Position refinement completed")
    
    def create_panorama(self) -> np.ndarray:
        """파노라마 생성"""
        if len(self.images) == 0:
            raise ValueError("No images loaded")
        
        # Phase 1: 센서 기반 초기 배치
        self.build_initial_layout_sensor()
        
        # Phase 2 & 3: 전역 최적화
        if self.use_global_optimization:
            self.refine_positions_global()
        
        # 캔버스 생성
        print(f"\n{'='*60}")
        print("Phase 4: Generating Final Panorama")
        print(f"{'='*60}")
        
        h = self.IMAGE_PIXEL_HEIGHT
        w = self.IMAGE_PIXEL_WIDTH
        
        min_x = min(pos[0] for pos in self.positions)
        max_x = max(pos[0] for pos in self.positions) + w
        min_y = min(pos[1] for pos in self.positions)
        max_y = max(pos[1] for pos in self.positions) + h
        
        canvas_w = max_x - min_x + 400
        canvas_h = max_y - min_y + 400
        
        print(f"Canvas size: {canvas_w} x {canvas_h}")
        
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        
        offset_x = -min_x + 200
        offset_y = -min_y + 200
        
        # 이미지 배치
        for i, (x, y) in enumerate(self.positions):
            abs_x = x + offset_x
            abs_y = y + offset_y
            
            y1 = max(0, min(abs_y, canvas_h - h))
            x1 = max(0, min(abs_x, canvas_w - w))
            y2 = y1 + h
            x2 = x1 + w
            
            img_y1, img_y2 = 0, h
            img_x1, img_x2 = 0, w

            if y1 <= 0:
                img_y1 = -y1
                y1 = 0
            if x1 <= 0:
                img_x1 = -x1
                x1 = 0
            if y2 > canvas_h:
                img_y2 = h - (y2 - canvas_h)
                y2 = canvas_h
            if x2 > canvas_w:
                img_x2 = w - (x2 - canvas_w)
                x2 = canvas_w

            if y2 > y1 and x2 > x1:
                canvas[y1:y2, x1:x2] = self.images[i][img_y1:img_y2, img_x1:img_x2]
        
        canvas = self._crop_canvas(canvas)
        
        print(f"\nFinal panorama: {canvas.shape[1]} x {canvas.shape[0]} px")
        real_w = canvas.shape[1] * self.CM_PER_PIXEL_X
        real_h = canvas.shape[0] * self.CM_PER_PIXEL_Y
        print(f"Real size: {real_w:.1f} x {real_h:.1f} cm")
        
        return canvas
    
    def _crop_canvas(self, canvas: np.ndarray) -> np.ndarray:
        """빈 영역 제거"""
        gray = canvas.mean(axis=2)
        rows = np.any(gray > 0, axis=1)
        cols = np.any(gray > 0, axis=0)
        
        if not np.any(rows) or not np.any(cols):
            return canvas
        
        y1, y2 = np.where(rows)[0][[0, -1]]
        x1, x2 = np.where(cols)[0][[0, -1]]
        
        return canvas[y1:y2+1, x1:x2+1]
    
    def calculate_coverage_range(self) -> Tuple[int, int, int, int]:
        """촬영 범위 계산"""
        if not self.building_width or not self.building_height:
            return None, None, None, None
        
        if self.vertical_sensor == "F":
            f_values = [meta.front for meta in self.metadata_list]
            y_min = self.building_height - max(f_values)
            y_max = self.building_height - min(f_values)
        else:
            b_values = [meta.back for meta in self.metadata_list]
            y_min = min(b_values)
            y_max = max(b_values)
        
        if self.horizontal_sensor == "L":
            l_values = [meta.left for meta in self.metadata_list]
            x_min = min(l_values)
            x_max = max(l_values)
        else:
            r_values = [meta.right for meta in self.metadata_list]
            x_min = self.building_width - max(r_values)
            x_max = self.building_width - min(r_values)
        
        return int(y_min), int(y_max), int(x_min), int(x_max)
    
    def save_panorama(self, panorama: np.ndarray, output_path: str):
        """파노라마 저장"""
        h, w = panorama.shape[:2]
        max_dimension = 65000
        
        input_folder_name = self.folder_path.name
        
        if self.building_width and self.building_height:
            y_min, y_max, x_min, x_max = self.calculate_coverage_range()
            
            if y_min is not None:
                print(f"\n📍 Coverage Range (Building Coordinate):")
                print(f"   Y: {y_min} ~ {y_max} cm")
                print(f"   X: {x_min} ~ {x_max} cm")
                
                base_path = Path(output_path)
                mode_str = "global" if self.use_global_optimization else "sensor"
                sensor_str = self.sensor_mode.lower()
                direction_str = self.movement_direction
                new_filename = f"{input_folder_name}_{mode_str}_{sensor_str}_{direction_str}_Ymin{y_min}_Ymax{y_max}_Xmin{x_min}_Xmax{x_max}.jpg"
                output_path = str(base_path.parent / new_filename)
        else:
            base_path = Path(output_path)
            mode_str = "global" if self.use_global_optimization else "sensor"
            sensor_str = self.sensor_mode.lower()
            direction_str = self.movement_direction
            new_filename = f"{input_folder_name}_{mode_str}_{sensor_str}_{direction_str}.jpg"
            output_path = str(base_path.parent / new_filename)
        
        if w > max_dimension or h > max_dimension:
            print(f"\n⚠ Image too large ({w}x{h}), resizing for JPG...")
            
            if w > h:
                new_w = max_dimension
                new_h = int(h * (max_dimension / w))
            else:
                new_h = max_dimension
                new_w = int(w * (max_dimension / h))
            
            img_pil = Image.fromarray(panorama)
            img_pil_resized = img_pil.resize((new_w, new_h), Image.LANCZOS)
            
            png_path = output_path.replace('.jpg', '_full.png')
            try:
                img_pil.save(png_path, compress_level=3)
                print(f"✓ Full size (PNG): {png_path}")
            except Exception as e:
                print(f"  Failed to save full size PNG: {e}")
            
            img_to_save = img_pil_resized
        else:
            img_to_save = Image.fromarray(panorama)
        
        try:
            img_to_save.save(output_path, quality=95)
            print(f"✓ Saved: {output_path}")
        except Exception as e:
            try:
                png_path = output_path.replace('.jpg', '.png')
                img_to_save.save(png_path, compress_level=6)
                print(f"✓ Saved (PNG): {png_path}")
            except Exception as e2:
                print(f"❌ Failed to save image: {e2}")
    
    def process(self, output_dir: str = "./output"):
        """전체 프로세스 실행"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        self.load_images_from_folder()
        
        panorama = self.create_panorama()
        
        input_folder_name = self.folder_path.name
        mode_str = "global" if self.use_global_optimization else "sensor"
        sensor_str = self.sensor_mode.lower()
        direction_str = self.movement_direction
        output_filename = f"{input_folder_name}_{mode_str}_{sensor_str}_{direction_str}.jpg"
        self.save_panorama(panorama, str(output_path / output_filename))
        
        print(f"\n{'='*60}")
        print("✓ Panorama generation completed!")
        print(f"{'='*60}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python panorama_global.py <image_folder> [output_folder] [sensor_mode] [movement_direction] [use_global] [overlap_threshold] [iterations] [building_width] [building_height] [image_real_width] [image_real_height]")
        print("\nExample:")
        print("  # Global optimization (default)")
        print("  python panorama_global.py ./images")
        print("  python panorama_global.py ./images ./output BL forward 1 0.3 3")
        print("\n  # Sensor-only mode")
        print("  python panorama_global.py ./images ./output BL forward 0")
        print("\n  # With building size")
        print("  python panorama_global.py ./images ./output BL forward 1 0.3 3 1620 810 125 87")
        print("\nArguments:")
        print("  sensor_mode        : 'FL', 'FR', 'BL', 'BR' (default: BL)")
        print("  movement_direction : 'forward', 'backward', 'left', 'right' (default: forward)")
        print("  use_global         : 0 (sensor-only) or 1 (global optimization) (default: 1)")
        print("  overlap_threshold  : 0.0~1.0, 겹침 판단 임계값 (default: 0.3)")
        print("  iterations         : 1~10, 정밀화 반복 횟수 (default: 3)")
        print("  building_width     : 건물 전체 가로(X) 길이 (cm)")
        print("  building_height    : 건물 전체 세로(Y) 길이 (cm)")
        print("  image_real_width   : 이미지 1장의 실제 가로 크기 (cm, default: 125)")
        print("  image_real_height  : 이미지 1장의 실제 세로 크기 (cm, default: 87)")
        print("\nGlobal Optimization:")
        print("  Phase 1: Build initial layout from sensor data")
        print("  Phase 2: Find overlapping neighbor images")
        print("  Phase 3: Refine positions with feature matching")
        print("  Phase 4: Generate final panorama")
        
        sys.exit(0)
    
    folder_path = sys.argv[1]
    output_folder = sys.argv[2] if len(sys.argv) > 2 else "./output"
    sensor_mode = sys.argv[3] if len(sys.argv) > 3 else "BL"
    movement_direction = sys.argv[4] if len(sys.argv) > 4 else "forward"
    use_global = int(sys.argv[5]) if len(sys.argv) > 5 else 1
    overlap_threshold = float(sys.argv[6]) if len(sys.argv) > 6 else 0.3
    iterations = int(sys.argv[7]) if len(sys.argv) > 7 else 3
    building_width = int(sys.argv[8]) if len(sys.argv) > 8 else None
    building_height = int(sys.argv[9]) if len(sys.argv) > 9 else None
    image_real_width = int(sys.argv[10]) if len(sys.argv) > 10 else None
    image_real_height = int(sys.argv[11]) if len(sys.argv) > 11 else None
    
    try:
        stitcher = GlobalOptimizationStitcher(
            folder_path, 
            building_width=building_width,
            building_height=building_height,
            image_real_width=image_real_width,
            image_real_height=image_real_height,
            sensor_mode=sensor_mode,
            movement_direction=movement_direction,
            use_global_optimization=bool(use_global),
            overlap_threshold=overlap_threshold,
            refinement_iterations=iterations
        )
        stitcher.process(output_dir=output_folder)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()