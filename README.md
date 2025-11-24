# PythonScriptUpdate

파노라마 이미지 스티칭 스크립트 - ROI 기반 피처 매칭 버전

## 개요

센서 데이터와 피처 매칭을 결합한 파노라마 이미지 생성 도구입니다.
전진/후진 이동 시 이미지의 겹치는 가장자리 영역에 집중하여 피처 매칭의 정확도를 향상시킵니다.

## 주요 기능

### 1. ROI(Region of Interest) 기반 피처 매칭
- 이동 방향에 따라 겹치는 영역에만 집중하여 특징점 검출
- **Forward 이동**: 상단 이미지의 하단 50% ↔ 하단 이미지의 상단 50%
- **Backward 이동**: 하단 이미지의 상단 50% ↔ 상단 이미지의 하단 50%
- Left/Right 이동: 전체 이미지 사용

### 2. 센서 데이터 기반 초기 배치
- Front, Back, Right, Left 센서 값을 사용한 정확한 초기 위치 설정
- 이미지 파일명 형식: `p001 f00089 b00478 r00558 l00246.jpg`

### 3. 회전/기울기 자동 보정 (NEW!)
- 아핀 변환 기반 회전 각도 자동 추정
- RANSAC을 사용한 robust 회전 각도 계산
- 5도 이상의 큰 회전은 노이즈로 필터링
- 마스크 기반 처리로 회전 영역의 검은색 부분이 아래 이미지를 가리지 않음
- 테스트 결과: **최대 46%의 이미지에 회전 보정 적용**

### 4. 전역 최적화
- SIFT 또는 ORB 피처 디텍터 지원
- 반복적 정밀화를 통한 위치 보정
- 로버스트한 중앙값 기반 오프셋 계산

## 성능 향상

ROI 기반 매칭 + 회전 보정 적용 결과:
- **피처 매칭 성공률**: 22.3% → 42.1% (**+89% 향상**, BEST CASE)
- **보정된 이미지 수**: 33개 → 55개 (**+67% 증가**)
- **회전 보정 적용**: 최대 46%의 이미지에 자동 회전 보정
- **이미지 품질**: 수평 정렬 및 이미지 연결 부드러움 향상

## 설치

```bash
pip install numpy pillow opencv-python
```

## 사용법

### 기본 사용

```bash
python panorama_hybrid_v1.0.py <image_folder> [output_folder] [sensor_mode] [movement_direction] [use_global] [overlap_threshold] [iterations]
```

### 예제

```bash
# ROI 기반 전역 최적화 (기본값)
python panorama_hybrid_v1.0.py ./images ./output BL forward 1 0.3 3

# 센서 전용 모드
python panorama_hybrid_v1.0.py ./images ./output BL forward 0

# 건물 크기 정보 포함
python panorama_hybrid_v1.0.py ./images ./output BL forward 1 0.3 3 1620 810 125 87
```

### 매개변수 설명

- `image_folder`: 이미지가 있는 폴더 경로
- `output_folder`: 결과 저장 폴더 (기본값: ./output)
- `sensor_mode`: 센서 조합 - 'FL', 'FR', 'BL', 'BR' (기본값: BL)
- `movement_direction`: 이동 방향 - 'forward', 'backward', 'left', 'right' (기본값: forward)
- `use_global`: 전역 최적화 사용 - 0 또는 1 (기본값: 1)
- `overlap_threshold`: 겹침 판단 임계값 - 0.0~1.0 (기본값: 0.3)
- `iterations`: 정밀화 반복 횟수 - 1~10 (기본값: 3)
- `building_width`: 건물 전체 가로 길이 (cm, 선택)
- `building_height`: 건물 전체 세로 길이 (cm, 선택)
- `image_real_width`: 이미지 실제 가로 크기 (cm, 기본값: 125)
- `image_real_height`: 이미지 실제 세로 크기 (cm, 기본값: 87)

## 처리 단계

### Phase 1: 센서 기반 초기 배치
- 센서 데이터를 사용하여 각 이미지의 초기 위치 계산
- 이동 방향과 센서 모드에 따라 자동 정렬

### Phase 2: 인접 이미지 탐색
- 겹침 영역이 있는 인접 이미지 쌍 탐색
- 겹침 임계값 기반 필터링

### Phase 3: ROI 기반 피처 매칭 및 위치 정밀화
- 마스크를 적용하여 겹치는 영역에서만 특징점 검출
- 아핀 변환으로 회전 각도 추정 및 보정
- 반복적 최적화로 위치 보정
- 수렴 조건: 최대 보정 < 2 픽셀

### Phase 4: 최종 파노라마 생성
- 회전 보정이 적용된 이미지를 보정된 위치에 배치
- 마스크 기반 블렌딩으로 회전 영역의 검은색 부분 처리
- 빈 영역 자동 제거
- JPG 또는 PNG 형식으로 저장

## 기술 세부사항

### ROI 마스크 생성
```python
# 이미지 높이의 50% 영역을 마스크로 사용
overlap_ratio = 0.50
overlap_height = int(image_height * overlap_ratio)

# Forward 이동 예시
if movement_direction == "forward":
    if idx2_is_above_idx1:
        # idx1의 상단 영역, idx2의 하단 영역
        mask1[0:overlap_height, :] = 255
        mask2[h-overlap_height:h, :] = 255
```

### 특징점 요구사항 (ROI 최적화)
- 최소 특징점: 4개 (기존 10개에서 완화)
- 최소 매칭: 4개 (기존 10개에서 완화)
- 최소 유효 매칭: 3개 (기존 5개에서 완화)

### 피처 디텍터
- **SIFT**: 높은 정확도, 느린 속도, 상용 라이선스 필요
- **ORB**: 빠른 속도, 무료, 약간 낮은 정확도

## 파일명 형식

이미지 파일명은 다음 형식을 따라야 합니다:

```
p<인덱스> f<전방센서> b<후방센서> r<우측센서> l<좌측센서>.jpg
```

예시:
```
p001 f00089 b00478 r00558 l00246.jpg
p002 f00089 b00497 r00823 l00267.jpg
```

## 출력 파일명

결과 파일명은 자동으로 생성됩니다:

```
<폴더명>_<모드>_<센서>_<방향>_Ymin<값>_Ymax<값>_Xmin<값>_Xmax<값>.jpg
```

예시:
```
2025-11-13_Auto_Final_4-1_global_bl_forward_Ymin478_Ymax1592_Xmin243_Xmax823.jpg
```

## 라이선스

MIT License

## 변경 이력

### v1.1 (2025-11-24)
- **회전/기울기 자동 보정 기능 추가**
- 아핀 변환 기반 회전 각도 추정
- 마스크 기반 블렌딩으로 회전 영역 처리
- 최대 46%의 이미지에 회전 보정 적용
- 피처 매칭 성공률 22.3% → 42.1% (+89% 향상)

### v1.0 (2025-11-21)
- ROI 기반 피처 매칭 구현
- 이동 방향별 마스크 자동 생성
- 특징점 요구사항 완화
- 67% 피처 매칭 성공률 향상

## 문의

이슈나 개선 제안은 GitHub Issues를 통해 제출해 주세요.
