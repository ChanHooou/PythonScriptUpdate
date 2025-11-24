# 파노라마 스티칭 - 파라미터 실행 가이드

## 기본 명령어 형식

```bash
python panorama_hybrid_v1.0.py <이미지_폴더> [출력_폴더] [센서_모드] [이동_방향] [전역_최적화] [겹침_임계값] [반복_횟수] [건물_가로] [건물_세로] [이미지_실제_가로] [이미지_실제_세로]
```

## 필수 파라미터

### 1. `<이미지_폴더>` (필수)
- **설명**: 센서 데이터가 포함된 이미지 파일들이 있는 폴더 경로
- **형식**: `p<번호> f<전방> b<후방> r<우측> l<좌측>.jpg`
- **예시**: `D:\CH\dataTest\2025-11-13_auto_test_2`

```bash
# 예시 파일명
p001 f00089 b00478 r00558 l00246.jpg
p002 f00089 b00497 r00823 l00267.jpg
```

## 선택 파라미터

### 2. `[출력_폴더]` (선택, 기본값: ./output)
- **설명**: 생성된 파노라마 이미지를 저장할 폴더
- **예시**: `D:\CH\pythonUtil\output`

### 3. `[센서_모드]` (선택, 기본값: BL)
- **설명**: 이미지 배치에 사용할 센서 조합
- **옵션**:
  - `FL`: Front + Left (전방 + 좌측)
  - `FR`: Front + Right (전방 + 우측)
  - `BL`: Back + Left (후방 + 좌측) ⭐ **권장**
  - `BR`: Back + Right (후방 + 우측)

### 4. `[이동_방향]` (선택, 기본값: forward)
- **설명**: 로봇/카메라의 이동 방향
- **옵션**:
  - `forward`: 전진 이동 ⭐ **권장**
  - `backward`: 후진 이동
  - `left`: 좌측 이동
  - `right`: 우측 이동

### 5. `[전역_최적화]` (선택, 기본값: 1)
- **설명**: ROI 기반 피처 매칭 및 회전 보정 사용 여부
- **옵션**:
  - `1`: 사용 (피처 매칭 + 회전 보정 + 위치 정밀화) ⭐ **권장**
  - `0`: 미사용 (센서 데이터만 사용)

### 6. `[겹침_임계값]` (선택, 기본값: 0.3)
- **설명**: 인접 이미지 판단 기준 (0.0 ~ 1.0)
- **권장값**:
  - `0.2`: 느슨한 기준 (더 많은 이미지 쌍 매칭)
  - `0.3`: 일반적 기준 ⭐ **권장**
  - `0.5`: 엄격한 기준 (확실한 겹침만 매칭)

### 7. `[반복_횟수]` (선택, 기본값: 3)
- **설명**: 위치 정밀화 반복 횟수
- **권장값**:
  - `1`: 빠른 처리
  - `3`: 일반적 품질 ⭐ **권장**
  - `5`: 높은 품질 (처리 시간 증가)

### 8-11. 건물 크기 파라미터 (선택)
- `[건물_가로]`: 건물 전체 가로 길이 (cm)
- `[건물_세로]`: 건물 전체 세로 길이 (cm)
- `[이미지_실제_가로]`: 이미지 실제 가로 크기 (cm, 기본값: 125)
- `[이미지_실제_세로]`: 이미지 실제 세로 크기 (cm, 기본값: 87)

## 실행 예시

### 예시 1: 기본 실행 (권장 설정)
```bash
python panorama_hybrid_v1.0.py D:\CH\dataTest\2025-11-13_auto_test_2
```
- 센서 모드: BL (Back + Left)
- 이동 방향: forward
- 전역 최적화: 사용
- 겹침 임계값: 0.3
- 반복 횟수: 3회
- 출력: ./output 폴더

### 예시 2: 모든 파라미터 명시
```bash
python panorama_hybrid_v1.0.py D:\CH\dataTest\2025-11-13_auto_test_2 D:\CH\pythonUtil\output BL forward 1 0.3 3
```
- 명시적으로 모든 주요 설정을 지정
- 출력 폴더: D:\CH\pythonUtil\output

### 예시 3: 센서 전용 모드 (피처 매칭 없음)
```bash
python panorama_hybrid_v1.0.py D:\CH\dataTest\2025-11-13_auto_test_2 ./output BL forward 0
```
- 전역 최적화 미사용 (센서 데이터만 사용)
- 빠른 처리, 낮은 정확도

### 예시 4: 다른 센서 조합 테스트
```bash
# Front + Right 센서 사용
python panorama_hybrid_v1.0.py D:\CH\dataTest\2025-11-13_auto_test_2 ./output FR forward 1 0.3 3
```

### 예시 5: 건물 크기 정보 포함
```bash
python panorama_hybrid_v1.0.py D:\CH\dataTest\2025-11-13_auto_test_2 ./output BL forward 1 0.3 3 1620 810 125 87
```
- 건물 크기: 1620cm × 810cm
- 이미지 크기: 125cm × 87cm

### 예시 6: 후진 이동 시나리오
```bash
python panorama_hybrid_v1.0.py D:\CH\dataTest\backward_test ./output BL backward 1 0.3 3
```
- 이동 방향: backward (후진)
- ROI 영역이 자동으로 반전됨

### 예시 7: 높은 품질 설정 (처리 시간 증가)
```bash
python panorama_hybrid_v1.0.py D:\CH\dataTest\2025-11-13_auto_test_2 ./output BL forward 1 0.2 5
```
- 겹침 임계값: 0.2 (더 많은 이미지 쌍 매칭)
- 반복 횟수: 5회 (더 정밀한 보정)

## 실행 결과

### 처리 단계
1. **Phase 1**: 센서 기반 초기 배치
2. **Phase 2**: 인접 이미지 탐색
3. **Phase 3**: ROI 기반 피처 매칭 및 위치 정밀화
   - 반복마다 회전 보정 및 위치 보정 적용
   - 수렴 조건: 최대 보정 < 2 픽셀
4. **Phase 4**: 최종 파노라마 생성
   - 회전 보정 적용 (마스크 기반 블렌딩)
   - 빈 영역 제거

### 출력 파일명 형식
```
<폴더명>_<모드>_<센서>_<방향>_Ymin<값>_Ymax<값>_Xmin<값>_Xmax<값>.jpg
```

**예시**:
```
2025-11-13_auto_test_2_global_bl_forward_Ymin478_Ymax1592_Xmin243_Xmax823.jpg
```

### 콘솔 출력 예시
```
======================================================================
Panorama Stitching with ROI-based Feature Matching + Rotation Correction
======================================================================
Images folder: D:\CH\dataTest\2025-11-13_auto_test_2
Sensor mode: BL (Back + Left)
Movement direction: forward
Use global optimization: True
Overlap threshold: 0.3
Refinement iterations: 3

Phase 1: Building initial layout using sensor data...
  Loaded 131 images

Phase 2: Finding neighboring image pairs...
  Found 124 image pairs with overlap

Phase 3: Refining positions using feature matching...
Iteration 1/3:
  Matched 52 pairs (success rate: 41.9%)
  Applied rotation corrections to 35 images
  Max correction: 8.5 pixels

Iteration 2/3:
  Matched 54 pairs (success rate: 43.5%)
  Applied rotation corrections to 37 images
  Max correction: 3.2 pixels

Iteration 3/3:
  Matched 55 pairs (success rate: 44.4%)
  Applied rotation corrections to 37 images
  Max correction: 1.8 pixels
  Converged! (Max correction < 2.0 pixels)

Phase 4: Creating final panorama...
  Canvas size: 1116 x 1351
  Saved: output/2025-11-13_auto_test_2_global_bl_forward_Ymin478_Ymax1592_Xmin243_Xmax823.jpg

Done!
```

## 파라미터 선택 가이드

### 센서 모드 선택
- **BL (Back + Left)**: 대부분의 경우 권장
  - 후방 센서: Y축 (세로) 위치
  - 좌측 센서: X축 (가로) 위치
- **BR (Back + Right)**: 우측 벽면 촬영 시
- **FL/FR**: 전방 센서 사용 시 (덜 일반적)

### 이동 방향 선택
- **forward**: 로봇이 전진하며 촬영 (대부분의 경우)
- **backward**: 로봇이 후진하며 촬영
- ROI 영역이 이동 방향에 따라 자동으로 조정됨

### 겹침 임계값 조정
- **낮을수록** (0.2): 더 많은 이미지 쌍을 매칭 시도
  - 장점: 더 많은 연결
  - 단점: 처리 시간 증가, 잘못된 매칭 가능성
- **높을수록** (0.5): 확실한 겹침만 매칭
  - 장점: 빠른 처리, 정확한 매칭
  - 단점: 일부 연결 누락 가능

### 반복 횟수 조정
- **1회**: 빠른 처리, 기본 보정
- **3회**: 일반적 품질 (권장)
- **5회 이상**: 정밀한 보정, 처리 시간 증가
- 수렴 시 자동으로 조기 종료됨

## 성능 팁

### 최적 성능을 위한 권장 설정
```bash
python panorama_hybrid_v1.0.py <이미지_폴더> ./output BL forward 1 0.3 3
```

### 빠른 테스트를 위한 설정
```bash
python panorama_hybrid_v1.0.py <이미지_폴더> ./output BL forward 1 0.3 1
```

### 최고 품질을 위한 설정
```bash
python panorama_hybrid_v1.0.py <이미지_폴더> ./output BL forward 1 0.2 5
```

## 문제 해결

### 피처 매칭 성공률이 낮은 경우
1. 겹침 임계값을 낮춰보기 (0.3 → 0.2)
2. 반복 횟수 증가 (3 → 5)
3. 다른 센서 모드 시도 (BL → BR)

### 이미지가 비정상적으로 배치되는 경우
1. 이동 방향 확인 (forward ↔ backward)
2. 센서 모드 변경 시도
3. 센서 전용 모드로 테스트 (전역_최적화=0)

### 처리 시간이 너무 긴 경우
1. 겹침 임계값 증가 (0.3 → 0.5)
2. 반복 횟수 감소 (3 → 1)
3. 이미지 수 확인

## 추가 정보

- **소스 코드**: https://github.com/ChanHooou/PythonScriptUpdate
- **버전**: v1.1 (2025-11-24)
- **라이선스**: MIT
- **피처 매칭 성공률**: 22.3% → 42.1% (+89% 향상, BEST CASE)
- **회전 보정 적용**: 최대 46%의 이미지에 자동 회전 보정
