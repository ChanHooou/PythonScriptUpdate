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
    """전역 최적화 파노라마 스티칭 (v2.0 - 상대 좌표 + Median 기반 Robust 트렌드)"""

    def __init__(self,
                 folder_path: str,
                 building_width: int = 810,
                 building_height: int = 1620,
                 image_real_width: int = 150,
                 image_real_height: int = 87,
                 sensor_mode: str = "BL",
                 movement_direction: str = "forward",
                 use_global_optimization: bool = False,
                 overlap_threshold: float = 0.3,
                 refinement_iterations: int = 3,
                 feature_method: str = "SIFT",
                 min_bf_sensor_diff: int = 0,
                 rotate_output: bool = True):
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
            min_bf_sensor_diff: B/F 센서 최소 차이 (cm) - 이보다 작으면 중복으로 간주 (0이면 비활성화)
            rotate_output: 최종 결과를 시계방향 90도 회전 및 x/y축 swap 여부
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
        self.min_bf_sensor_diff = min_bf_sensor_diff
        self.rotate_output = rotate_output
        
        self.IMAGE_REAL_WIDTH = image_real_width if image_real_width else 150
        self.IMAGE_REAL_HEIGHT = image_real_height if image_real_height else 87
        
        self.IMAGE_PIXEL_WIDTH = None
        self.IMAGE_PIXEL_HEIGHT = None
        
        self.CM_PER_PIXEL_X = None
        self.CM_PER_PIXEL_Y = None
        self.PIXEL_PER_CM_X = None
        self.PIXEL_PER_CM_Y = None
        
        # 이미지 위치 및 회전 저장 (전역 최적화용)
        self.positions: List[Tuple[int, int]] = []
        self.rotations: List[float] = []  # 각 이미지의 회전 각도 (도)

        # 보정된 센서 값 저장 (누적 추세 유지용)
        self.corrected_vertical_values: List[int] = []  # 보정된 F 또는 B 값
        self.corrected_horizontal_values: List[int] = []  # 보정된 L 또는 R 값

        # 트렌드 기반 보정을 위한 변수 (v2.0: median 기반)
        self.expected_vertical_diff: Optional[float] = None  # median 증감량
        self.expected_horizontal_diff: Optional[float] = None
        self.v_outlier_threshold: float = 50.0  # 세로축 이상치 threshold
        self.h_outlier_threshold: float = 4.0   # 가로축 이상치 threshold

        # 피처 디텍터
        self.feature_detector = None
        self.feature_matcher = None
        if self.use_global_optimization:
            self._initialize_feature_detector()
        
        print(f"\n{'='*60}")
        print("Global Optimization Panorama Stitcher v2.0")
        print("Strategy: Median-based Robust Trend + Outlier Detection")
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

        if self.min_bf_sensor_diff > 0:
            print(f"Sensor-based Duplicate Removal: Enabled")
            print(f"  Min B/F Sensor Diff: {self.min_bf_sensor_diff} cm")
        else:
            print(f"Sensor-based Duplicate Removal: Disabled")
        
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
            
            print(f"  OK {self.feature_method} feature detector initialized")
                
        except Exception as e:
            print(f"  WARNING Feature detector initialization failed: {e}")
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

    def remove_sensor_duplicates(self, temp_data: List[Tuple[ImageMetadata, Path]], min_sensor_diff: int = 20) -> List[Tuple[ImageMetadata, Path]]:
        """센서 값 기반 중복 이미지 제거 - b 또는 f 센서 값이 min_sensor_diff cm 미만 차이 시 제거"""
        if len(temp_data) < 2:
            return temp_data

        print(f"\n{'='*60}")
        print(f"Removing sensor-based duplicates (min diff: {min_sensor_diff} cm)...")
        print(f"{'='*60}")

        filtered_data = []
        removed_count = 0

        # 첫 이미지는 항상 포함
        filtered_data.append(temp_data[0])
        prev_meta = temp_data[0][0]

        for i in range(1, len(temp_data)):
            curr_meta = temp_data[i][0]

            # 선택된 센서(b 또는 f) 값 비교
            if self.vertical_sensor == "F":
                sensor_diff = abs(curr_meta.front - prev_meta.front)
                sensor_name = "F"
                curr_val = curr_meta.front
                prev_val = prev_meta.front
            else:  # "B"
                sensor_diff = abs(curr_meta.back - prev_meta.back)
                sensor_name = "B"
                curr_val = curr_meta.back
                prev_val = prev_meta.back

            if sensor_diff < min_sensor_diff:
                print(f"  REJECT p{curr_meta.index:03d}: {sensor_name} sensor diff too small ({sensor_diff} cm < {min_sensor_diff} cm) - {sensor_name}={curr_val} vs prev {sensor_name}={prev_val}")
                removed_count += 1
            else:
                filtered_data.append(temp_data[i])
                prev_meta = curr_meta

        print(f"\n  OK Sensor-based duplicates removed: {removed_count}")
        print(f"  Kept {len(filtered_data)} images")

        return filtered_data

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
                print(f"  WARNING Skipping {img_path.name}: {e}")
        
        if len(temp_data) == 0:
            raise ValueError("No valid images found")

        temp_data.sort(key=lambda x: self._get_sort_key(x[0]))

        # 센서 값 기반 중복 제거 (활성화된 경우에만)
        if self.min_bf_sensor_diff > 0:
            temp_data = self.remove_sensor_duplicates(temp_data, min_sensor_diff=self.min_bf_sensor_diff)

        if len(temp_data) < 2:
            raise ValueError(f"Not enough valid images after filtering (need >= 2, got {len(temp_data)})")

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
        
        print(f"\nOK Successfully loaded {len(self.images)} images")

    def establish_sensor_trend(self):
        """Median 기반 robust 트렌드 추정 (이상치에 강함)"""
        from statistics import median

        if len(self.metadata_list) < 3:
            print(f"Warning: Not enough images for trend estimation")
            return

        vertical_diffs = []
        horizontal_diffs = []

        # 모든 이미지 간 diff 계산
        for i in range(1, len(self.metadata_list)):
            prev = self.metadata_list[i-1]
            curr = self.metadata_list[i]

            # 세로축 센서 차이
            if self.vertical_sensor == "F":
                v_diff = prev.front - curr.front
            else:
                v_diff = curr.back - prev.back

            # 가로축 센서 차이
            if self.horizontal_sensor == "L":
                h_diff = curr.left - prev.left
            else:
                h_diff = curr.right - prev.right

            vertical_diffs.append(v_diff)
            horizontal_diffs.append(h_diff)

        # Median 사용 (평균 대신 - 이상치에 robust)
        self.expected_vertical_diff = median(vertical_diffs)
        self.expected_horizontal_diff = median(horizontal_diffs)

        # MAD (Median Absolute Deviation) 기반 이상치 threshold
        v_deviations = [abs(d - self.expected_vertical_diff) for d in vertical_diffs]
        h_deviations = [abs(d - self.expected_horizontal_diff) for d in horizontal_diffs]

        v_mad = median(v_deviations) if v_deviations else 0
        h_mad = median(h_deviations) if h_deviations else 0

        # 3*MAD 규칙 (최소값 적용)
        self.v_outlier_threshold = max(50, 3 * v_mad)
        self.h_outlier_threshold = max(4, 3 * h_mad)

        print(f"\nMedian-based Trend Estimation:")
        print(f"  Vertical (median): {self.expected_vertical_diff:+.1f} cm/image")
        print(f"  Horizontal (median): {self.expected_horizontal_diff:+.1f} cm/image")
        print(f"  Outlier thresholds: V={self.v_outlier_threshold:.1f}cm, H={self.h_outlier_threshold:.1f}cm")

    def calculate_sensor_offset(self, idx: int) -> Tuple[int, int]:
        """센서 기반 오프셋 계산 (Diff 기반 이상치 감지 및 보정)"""
        curr = self.metadata_list[idx]

        # 첫 이미지: 보정된 값 초기화
        if idx == 0:
            if self.vertical_sensor == "F":
                self.corrected_vertical_values.append(curr.front)
            else:
                self.corrected_vertical_values.append(curr.back)

            if self.horizontal_sensor == "L":
                self.corrected_horizontal_values.append(curr.left)
            else:
                self.corrected_horizontal_values.append(curr.right)

            return (0, 0)

        # 이전 이미지 가져오기
        prev = self.metadata_list[idx - 1]
        prev_corrected_vertical = self.corrected_vertical_values[idx - 1]
        prev_corrected_horizontal = self.corrected_horizontal_values[idx - 1]

        # 세로축(Y) - Diff 기반 보정
        if self.vertical_sensor == "F":
            # 원본 diff 계산
            vertical_diff_raw = prev.front - curr.front

            # 이상치 감지: expected_vertical_diff와 비교
            if self.expected_vertical_diff is not None:
                deviation = abs(vertical_diff_raw - self.expected_vertical_diff)

                if deviation > self.v_outlier_threshold:
                    # 이상치 -> median diff 사용
                    vertical_diff = self.expected_vertical_diff
                    if idx < 20 or deviation > 200:
                        print(f"  [p{curr.index:03d}] V outlier: {vertical_diff_raw:+.0f} -> {vertical_diff:+.0f} (dev={deviation:.0f})")
                else:
                    # 정상 -> 원본 diff 사용
                    vertical_diff = vertical_diff_raw
            else:
                vertical_diff = vertical_diff_raw

            # 보정된 절대값 계산 (누적)
            curr_corrected_vertical = prev_corrected_vertical - vertical_diff
            self.corrected_vertical_values.append(curr_corrected_vertical)
            dy = -int(vertical_diff * self.PIXEL_PER_CM_Y)

        else:  # "B"
            # 원본 diff 계산
            vertical_diff_raw = curr.back - prev.back

            # 이상치 감지
            if self.expected_vertical_diff is not None:
                deviation = abs(vertical_diff_raw - self.expected_vertical_diff)

                if deviation > self.v_outlier_threshold:
                    # 이상치 -> median diff 사용
                    vertical_diff = self.expected_vertical_diff
                    if idx < 20 or deviation > 200:
                        print(f"  [p{curr.index:03d}] V outlier: {vertical_diff_raw:+.0f} -> {vertical_diff:+.0f} (dev={deviation:.0f})")
                else:
                    # 정상 -> 원본 diff 사용
                    vertical_diff = vertical_diff_raw
            else:
                vertical_diff = vertical_diff_raw

            # 보정된 절대값 계산 (누적)
            curr_corrected_vertical = prev_corrected_vertical + vertical_diff
            self.corrected_vertical_values.append(curr_corrected_vertical)
            dy = -int(vertical_diff * self.PIXEL_PER_CM_Y)

        # 가로축(X) - Diff 기반 보정
        if self.horizontal_sensor == "L":
            # 원본 diff 계산
            horizontal_diff_raw = curr.left - prev.left

            # 이상치 감지
            if self.expected_horizontal_diff is not None:
                deviation = abs(horizontal_diff_raw - self.expected_horizontal_diff)

                if deviation > self.h_outlier_threshold:
                    horizontal_diff = self.expected_horizontal_diff
                    if idx < 20:
                        print(f"  [p{curr.index:03d}] H outlier: {horizontal_diff_raw:+.0f} -> {horizontal_diff:+.0f}")
                else:
                    horizontal_diff = horizontal_diff_raw
            else:
                horizontal_diff = horizontal_diff_raw

            curr_corrected_horizontal = prev_corrected_horizontal + horizontal_diff
            self.corrected_horizontal_values.append(curr_corrected_horizontal)
            dx = int(horizontal_diff * self.PIXEL_PER_CM_X)

        else:  # "R"
            # 원본 diff 계산
            horizontal_diff_raw = curr.right - prev.right

            # 이상치 감지
            if self.expected_horizontal_diff is not None:
                deviation = abs(horizontal_diff_raw - self.expected_horizontal_diff)

                if deviation > self.h_outlier_threshold:
                    horizontal_diff = self.expected_horizontal_diff
                    if idx < 20:
                        print(f"  [p{curr.index:03d}] H outlier: {horizontal_diff_raw:+.0f} -> {horizontal_diff:+.0f}")
                else:
                    horizontal_diff = horizontal_diff_raw
            else:
                horizontal_diff = horizontal_diff_raw

            curr_corrected_horizontal = prev_corrected_horizontal + horizontal_diff
            self.corrected_horizontal_values.append(curr_corrected_horizontal)
            dx = -int(horizontal_diff * self.PIXEL_PER_CM_X)

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

        # 센서 트렌드 확립 (첫 N개 이미지 분석)
        self.establish_sensor_trend()

        # 첫 번째 이미지 초기화 (보정 값도 함께 초기화됨)
        self.calculate_sensor_offset(0)
        self.positions = [(0, 0)]
        self.rotations = [0.0]  # 첫 이미지는 회전 없음

        for i in range(1, len(self.images)):
            dx, dy = self.calculate_sensor_offset(i)
            prev_x, prev_y = self.positions[-1]
            new_x = prev_x + dx
            new_y = prev_y + dy
            self.positions.append((new_x, new_y))
            self.rotations.append(0.0)  # 초기값은 회전 없음

            if i < 5 or i % 10 == 0:
                direction_v = "^" if dy < 0 else "v" if dy > 0 else "-"
                direction_h = "<" if dx < 0 else ">" if dx > 0 else ""
                print(f"Image {i:3d}: offset=({dx:+5d}, {dy:+5d}) {direction_h}{direction_v} > pos=({new_x:6d}, {new_y:6d}) [sensor]")

        print(f"\nOK Initial layout completed with {len(self.positions)} images")
    
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
    
    def match_features_between_images(self, idx1: int, idx2: int) -> Optional[Tuple[int, int, float]]:
        """두 이미지 간 피처 매칭으로 상대 오프셋 및 회전 계산

        Args:
            idx1: 기준 이미지 인덱스
            idx2: 비교 이미지 인덱스

        Returns:
            (dx, dy, rotation): idx2가 idx1 대비 이동해야 할 오프셋과 회전 각도(도), 실패시 None
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

            # 회전 각도 추정 (유효한 매칭점들만 사용)
            valid_src_pts = src_pts[valid_mask]
            valid_dst_pts = dst_pts[valid_mask]

            rotation_angle = 0.0
            if len(valid_src_pts) >= 3:
                # 아핀 변환 행렬 추정 (회전 + 이동)
                affine_matrix, inliers = cv2.estimateAffinePartial2D(
                    valid_src_pts,
                    valid_dst_pts,
                    method=cv2.RANSAC,
                    ransacReprojThreshold=5.0
                )

                if affine_matrix is not None and inliers is not None and np.sum(inliers) >= 3:
                    # 회전 각도 추출 (라디안 -> 도)
                    rotation_angle = np.arctan2(affine_matrix[1, 0], affine_matrix[0, 0]) * 180.0 / np.pi

                    # 너무 큰 회전은 무시 (5도 이상)
                    if abs(rotation_angle) > 5.0:
                        rotation_angle = 0.0

            # 피처 매칭 결과: img1의 점이 img2에서 (dx_feature, dy_feature) 이동
            # 이는 img2가 img1 대비 (-dx_feature, -dy_feature) 위치에 있다는 의미
            correction_dx = -dx_feature - sensor_dx
            correction_dy = -dy_feature - sensor_dy

            return (correction_dx, correction_dy, rotation_angle)
            
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
        
        print(f"\nOK Found {total_pairs} overlapping pairs")
        
        if total_pairs == 0:
            print("WARNING No overlapping images found, skipping refinement")
            return
        
        print(f"\n{'='*60}")
        print(f"Phase 3: Refining Positions with Feature Matching")
        print(f"{'='*60}")
        
        # 반복적 정밀화
        for iteration in range(self.refinement_iterations):
            print(f"\nIteration {iteration + 1}/{self.refinement_iterations}")
            
            # 각 이미지에 대한 보정값 및 회전 수집
            corrections: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
            rotation_corrections: Dict[int, List[float]] = defaultdict(list)
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
                        dx, dy, rotation = correction
                        # j를 보정
                        corrections[j].append((dx, dy))
                        if abs(rotation) > 0.1:  # 0.1도 이상의 회전만 기록
                            rotation_corrections[j].append(rotation)
                        successful_matches += 1
                    else:
                        failed_matches += 1
            
            print(f"  Processing image {len(self.images)}/{len(self.images)}... Done")
            print(f"  Feature matching: {successful_matches} success, {failed_matches} failed")
            
            if successful_matches == 0:
                print("  No successful matches, stopping refinement")
                break
            
            # 보정 적용 (중앙값 사용)
            max_correction = 0
            corrected_count = 0
            rotation_count = 0

            for i, correction_list in corrections.items():
                if len(correction_list) > 0:
                    # 중앙값으로 로버스트하게
                    dx_corrections = [c[0] for c in correction_list]
                    dy_corrections = [c[1] for c in correction_list]

                    dx_median = int(np.median(dx_corrections))
                    dy_median = int(np.median(dy_corrections))

                    # 위치 보정 적용
                    x, y = self.positions[i]
                    self.positions[i] = (x + dx_median, y + dy_median)

                    correction_magnitude = np.sqrt(dx_median**2 + dy_median**2)
                    max_correction = max(max_correction, correction_magnitude)
                    corrected_count += 1

                    # 회전 보정 적용
                    if i in rotation_corrections and len(rotation_corrections[i]) > 0:
                        rotation_median = float(np.median(rotation_corrections[i]))
                        self.rotations[i] += rotation_median
                        rotation_count += 1

            print(f"  Applied corrections to {corrected_count} images")
            if rotation_count > 0:
                print(f"  Applied rotation corrections to {rotation_count} images")
            print(f"  Max correction: {max_correction:.1f} pixels")
            
            # 수렴 판단
            if max_correction < 2.0:  # 2픽셀 이하면 수렴
                print(f"  Converged (max correction < 2 pixels)")
                break
        
        print(f"\nOK Position refinement completed")
    
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
            # 회전 적용
            img = self.images[i].copy()
            rotation_angle = self.rotations[i]
            mask = None

            if abs(rotation_angle) > 0.1:  # 0.1도 이상의 회전만 적용
                # 이미지 중심을 기준으로 회전
                center = (w // 2, h // 2)
                rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
                img = cv2.warpAffine(img, rotation_matrix, (w, h),
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=(0, 0, 0))

                # 회전으로 생긴 검은색 영역을 마스크로 생성
                mask = cv2.warpAffine(
                    np.ones((h, w), dtype=np.uint8) * 255,
                    rotation_matrix,
                    (w, h),
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0
                )

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
                img_crop = img[img_y1:img_y2, img_x1:img_x2]

                if mask is not None:
                    # 마스크 적용: 검은색 영역은 기존 캔버스 유지
                    mask_crop = mask[img_y1:img_y2, img_x1:img_x2]
                    # 마스크가 있는 부분만 새 이미지로 덮어쓰기
                    canvas_region = canvas[y1:y2, x1:x2]
                    canvas[y1:y2, x1:x2] = np.where(
                        mask_crop[:, :, np.newaxis] > 0,
                        img_crop,
                        canvas_region
                    )
                else:
                    # 회전 없는 경우 그냥 덮어쓰기
                    canvas[y1:y2, x1:x2] = img_crop
        
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
        """촬영 범위 계산 (이미지 물리적 크기 포함)"""
        if not self.building_width or not self.building_height:
            return None, None, None, None

        # Y축 범위 계산 (이미지 높이 포함)
        if self.vertical_sensor == "F":
            f_values = [meta.front for meta in self.metadata_list]
            # F센서: building_height 기준 역방향
            y_min = self.building_height - max(f_values) - self.IMAGE_REAL_HEIGHT
            y_max = self.building_height - min(f_values)
        else:
            b_values = [meta.back for meta in self.metadata_list]
            # B센서: 0 기준 정방향
            y_min = min(b_values)
            y_max = max(b_values) + self.IMAGE_REAL_HEIGHT

        # X축 범위 계산 (이미지 너비 포함)
        if self.horizontal_sensor == "L":
            l_values = [meta.left for meta in self.metadata_list]
            # L센서: 0 기준 정방향
            x_min = min(l_values)
            x_max = max(l_values) + self.IMAGE_REAL_WIDTH
        else:
            r_values = [meta.right for meta in self.metadata_list]
            # R센서: building_width 기준 역방향
            x_min = self.building_width - max(r_values) - self.IMAGE_REAL_WIDTH
            x_max = self.building_width - min(r_values)

        return int(y_min), int(y_max), int(x_min), int(x_max)
    
    def save_panorama(self, panorama: np.ndarray, output_path: str):
        """파노라마 저장"""
        h, w = panorama.shape[:2]
        max_dimension = 65000

        input_folder_name = self.folder_path.name

        # 회전 옵션 적용 전 coverage range 계산
        if self.building_width and self.building_height:
            y_min, y_max, x_min, x_max = self.calculate_coverage_range()

            # rotate_output이 True면 x/y축 swap
            if self.rotate_output:
                y_min, y_max, x_min, x_max = x_min, x_max, y_min, y_max

            if y_min is not None:
                print(f"\nCoverage Range (Building Coordinate):")
                if self.rotate_output:
                    print(f"   Y: {y_min} ~ {y_max} cm (rotated - original X)")
                    print(f"   X: {x_min} ~ {x_max} cm (rotated - original Y)")
                else:
                    print(f"   Y: {y_min} ~ {y_max} cm")
                    print(f"   X: {x_min} ~ {x_max} cm")

                base_path = Path(output_path)
                mode_str = "global" if self.use_global_optimization else "sensor"
                sensor_str = self.sensor_mode.lower()
                direction_str = self.movement_direction
                rotate_str = "_rotated" if self.rotate_output else ""
                new_filename = f"result_{x_min},{y_min}-{x_max},{y_max}.jpg"
                output_path = str(base_path.parent / new_filename)
        else:
            base_path = Path(output_path)
            mode_str = "global" if self.use_global_optimization else "sensor"
            sensor_str = self.sensor_mode.lower()
            direction_str = self.movement_direction
            rotate_str = "_rotated" if self.rotate_output else ""
            new_filename = f"result_{input_folder_name}.jpg"
            output_path = str(base_path.parent / new_filename)

        # 시계방향 90도 회전 적용
        if self.rotate_output:
            print(f"\nRotating output 90 degrees clockwise...")
            panorama = cv2.rotate(panorama, cv2.ROTATE_90_CLOCKWISE)
            h, w = panorama.shape[:2]
            print(f"   Rotated size: {w}x{h}")
        
        if w > max_dimension or h > max_dimension:
            print(f"\nWARNING Image too large ({w}x{h}), resizing for JPG...")
            
            if w > h:
                new_w = max_dimension
                new_h = int(h * (max_dimension / w))
            else:
                new_h = max_dimension
                new_w = int(w * (max_dimension / h))
            
            img_pil = Image.fromarray(panorama)
            img_pil_resized = img_pil.resize((new_w, new_h), Image.LANCZOS)
            
            # v2.0: PNG 출력 제거, JPG만 생성
            img_to_save = img_pil_resized
        else:
            img_to_save = Image.fromarray(panorama)
        
        try:
            img_to_save.save(output_path, quality=95)
            print(f"OK Saved: {output_path}")
        except Exception as e:
            print(f"ERROR Failed to save JPG: {e}")
    
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
        print("OK Panorama generation completed!")
        print(f"{'='*60}")


if __name__ == "__main__":
    import sys

    print("Global Optimization Panorama Stitcher v2.0")
    print("=" * 60)

    if len(sys.argv) < 2:
        print("\nUsage: python panorama_hybrid_v2.0_full.py <image_folder> [output_folder] [options...]")
        print("\nBasic Example:")
        print("  python panorama_hybrid_v2.0_full.py ./images")
        print("\nWith sensor filtering and rotation:")
        print("  python panorama_hybrid_v2.0_full.py ./images ./output BL backward 0 0.3 3 1620 810 125 87 20 True")
        print("\nArguments:")
        print("  1.  image_folder        : 입력 이미지 폴더")
        print("  2.  output_folder       : 출력 폴더 (default: ./output)")
        print("  3.  sensor_mode         : FL/FR/BL/BR (default: BL)")
        print("  4.  movement_direction  : forward/backward/left/right (default: forward)")
        print("  5.  use_global          : 0/1 (default: 1)")
        print("  6.  overlap_threshold   : 0.0~1.0 (default: 0.3)")
        print("  7.  iterations          : 1~10 (default: 3)")
        print("  8.  building_width      : 건물 가로 길이 (cm, default: None, 0이면 None)")
        print("  9.  building_height     : 건물 세로 길이 (cm, default: None, 0이면 None)")
        print("  10. image_real_width    : 이미지 실제 가로 (cm, default: 125)")
        print("  11. image_real_height   : 이미지 실제 세로 (cm, default: 87)")
        print("  12. min_bf_sensor_diff  : B/F 센서 최소 차이 cm - 중복 제거 (default: 0, 비활성화)")
        print("  13. rotate_output       : True/False - 시계방향 90도 회전 (default: False)")
        print("\n✨ v2.0 NEW FEATURES:")
        print("  - Median-based robust trend estimation (이상치에 강함)")
        print("  - Diff-based outlier detection (절대값 offset 영향 제거)")
        print("  - MAD (Median Absolute Deviation) 자동 threshold")
        print("  - 전체 offset 타입과 부분 점프 타입 모두 처리 가능")

        sys.exit(0)
    
    folder_path = sys.argv[1]
    output_folder = sys.argv[2] if len(sys.argv) > 2 else "./output"
    sensor_mode = sys.argv[3] if len(sys.argv) > 3 else "BL"
    movement_direction = sys.argv[4] if len(sys.argv) > 4 else "forward"
    use_global = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    overlap_threshold = float(sys.argv[6]) if len(sys.argv) > 6 else 0.3
    iterations = int(sys.argv[7]) if len(sys.argv) > 7 else 3
    building_width = int(sys.argv[8]) if len(sys.argv) > 8 and sys.argv[8] != '0' else 810
    building_height = int(sys.argv[9]) if len(sys.argv) > 9 and sys.argv[9] != '0' else 1620
    image_real_width = int(sys.argv[10]) if len(sys.argv) > 10 else None
    image_real_height = int(sys.argv[11]) if len(sys.argv) > 11 else None
    min_bf_sensor_diff = int(sys.argv[12]) if len(sys.argv) > 12 else 20
    rotate_output = sys.argv[13].lower() == 'true' if len(sys.argv) > 13 else True

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
            refinement_iterations=iterations,
            min_bf_sensor_diff=min_bf_sensor_diff,
            rotate_output=rotate_output
        )
        stitcher.process(output_dir=output_folder)
        
    except Exception as e:
        print(f"\nERROR Error: {e}")
        import traceback
        traceback.print_exc()