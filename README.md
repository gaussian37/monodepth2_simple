### **Monodepth2의 Simple 버전으로 아래 조건을 전제로 코드를 정리하였습니다.**

<br>

#### 1. Monocular Video 사용 (Stereo 배제)
#### 2. Monodepth2 이외의 논문 모듈 제거
#### 3. Monodepth2 지표의 최고 성능을 낼 수 있는 기법을 사용함
  - `minimum reprojection loss`, `auto-masking loss`, `full-resolution multi-scale sampling`, `SSIM`
#### 4. Pose Network는 Seperate Network를 사용함
#### 5. multi scale은 (0, 1, 2, 3) 즉, (2^0, 2^1, 2^2, 2^3) 만큼 down sampling 하도록 사용함
#### 6. disparity를 구하기 위하여 현재 frame (I_(t))을 기준으로 직전 프레임(I_(t-1))과 직후 프레임(I_(t+1))을 사용하는 것을 가정함

<br>

- Monodepth2 리뷰 : [https://gaussian37.github.io/vision-depth-monodepth2/](https://gaussian37.github.io/vision-depth-monodepth2/)
- Monodepth2 원본 깃헙 링크 : https://github.com/nianticlabs/monodepth2
- Monodepth2 논문 링크 : https://arxiv.org/abs/1806.01260